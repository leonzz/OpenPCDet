# Improving Mid-to-Long Range 3D Object Detection

This document serves as a description of our implementation for our course project for [CSC413/2516 at UofT Winter 2021](https://csc413-uoft.github.io/2021/).

## Contributions on top of OpenPCDet

On top of OpenPCDet, we have committed (or changed) the following files:

- `pcdet/models/dense_heads/anchor_head_template.py`: the logic that enables weighted loss function based on range.
- `tools/cfgs/range_weighted_loss/pointpillar.yaml`: the configuration file to train and evaluate model for range weighted loss function.
- `tools/cfgs/kitti_models/range_training/*`: a set of configurations files that trains 2 models for the Split Model method.
- `tools/merge_predictions/merge_predictions.py`: Script to merge prediction results from the Split Model method.
- `tools/merge_predictions/evaluate_kitti_range.py`: Script to evaluate a model's recognition accuracy for KITTI dataset.
- `tools/analyze_kitti_distribution/analyze_gt_points.py`: Script to produce number of car distribution over range.

## Running the Experiments

### Prerequisites

A workstation with at least one NVIDIA GPU that supports CUDA is required.

Follow the [OpenPCDet instruction](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) to install the package. We created a slightly detailed instruction that installs dependencies on a fresh GCP instance with Ubuntu 18.04 LTS OS.

1. Make sure python version `>= 3.6`. Here we do a fresh install of Python 3.7:
    ```
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.7 python3.7-dev
    sudo rm /usr/bin/python
    sudo ln -s /usr/bin/python3.7 /usr/bin/python
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
    ```
2. Install some dependencies (numpy, pytorch, etc.):
    ```
    git clone https://github.com/open-mmlab/OpenPCDet.git
    cd OpenPCDet/
    pip install -r requirements.txt 
    ```
3. Install CUDA 10.2 (An underlying dependency `numba` is not compatible with latest CUDA 11):
    ```
    CUDA_REPO_PKG=cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
    wget -O /tmp/${CUDA_REPO_PKG} https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/${CUDA_REPO_PKG} 
    sudo dpkg -i /tmp/${CUDA_REPO_PKG}
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub 
    rm -f /tmp/${CUDA_REPO_PKG}
    sudo apt-get update
    sudo apt-get install cuda-10-2
    sudo apt-get install nvidia-smi
    sudo reboot
    nvidia-smi # to check cuda version
    ```
4. Install `cudnn` (required to build `spconv`): Follow [NVIDIA's guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#download). Make sure CUDNN version matches CUDA version.

5. Install `cmake` (required bo build `spconv`):
    ```
    wget https://github.com/Kitware/CMake/releases/download/v3.19.5/cmake-3.19.5-Linux-x86_64.sh
    chmod +x cmake-3.19.5-Linux-x86_64.sh 
    ./cmake-3.19.5-Linux-x86_64.sh 
    mv cmake-3.19.5-Linux-x86_64 ~/.local/
    ln -s ~/.local/cmake-3.19.5-Linux-x86_64/bin/cmake ~/.local/bin/cmake
    ```
6. Install `spconv`:
    ```
    git clone https://github.com/traveller59/spconv.git --recursive
    cd spconv
    # add `set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.2/bin/nvcc")` to `CMakeLists.txt`
    (echo '\set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.2/bin/nvcc")' && cat CMakeLists.txt) > CMakeLists.txt.new && mv CMakeLists.txt.new CMakeLists.txt
    python setup.py bdist_wheel
    cd dist/
    pip install spconv-1.2.1-cp37-cp37m-linux_x86_64.whl 
    ```
7. Install `OpenPCDet`:
    ```
    cd OpenPCDet/
    python setup.py develop
    ```
8. Download KITTI dataset: Follow [KITTI instruction](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) to download at least the following:
    ```
    training label (5MB)
    Velodyne point cloud (29 GB)
    camera calibration matrices of object data set (16 MB)
    left color images of object data set (12 GB)
    ```
9. Prepare KITTI info in OpenPCDet format. Follow this [folder structure](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md#kitti-dataset) and run `python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml`

### Train and Evaluate

Checkout csc2516 branch.
```
cd tools
git checkout csc2516
```

- Baseline PointPillars
    ```
    cd tools
    python train.py --cfg_file cfgs/kitti_models/range_training/car/pointpillar_full_range.yaml --save_to_file
    ```
    The result file will be auto saved at `OpenPCDet/output/kitti_models/pointpillar/default/eval/eval_with_train/`, run evaluation script against the result from the last epoch:
    ```
    python tools/merge_predictions/evaluate_kitti_range.py --full_range_result path/to/result.pkl
    ```
- Method 1: Split Model
    ```
    # Train short range model
    python train.py --cfg_file cfgs/kitti_models/range_training/car/pointpillar_0_35.yaml --save_to_file
    # Train long range model
    python train.py --cfg_file cfgs/kitti_models/range_training/car/pointpillar_30_70.yaml --save_to_file
    ```
    Again, find the last epoch result files for the 2 models in `OpenPCDet/output/kitti_models/pointpillar/`, and run the merge / evaluation script:
    ```
    python tools/merge_predictions/evaluate_kitti_range.py --short_range_result=path/to/result_short.pkl --long_range_result=path/to/result_long.pkl --save_merge_result
    ```

- Method 2: Range Weighted Loss, Edit Line 64 and Line 190 of `pcdet/models/dense_heads/anchor_head_template.py` to switch among the 3 cases:
    - Case 1 Range Weight: `weight_anchor = self.range_to_anchors.unsqueeze(dim=0)`
    - Case 2 1/NSample: `weight_anchor = self.train_dist_weight.unsqueeze(dim=0)`, `LOG = False`
    - Case 3 1/log(Nsample) `weight_anchor = self.train_dist_weight.unsqueeze(dim=0)`, `LOG = True`

    ```
    python train.py --cfg_file cfgs/kitti_models/range_weighted_loss/pointpillar.yaml --save_to_file

    python tools/merge_predictions/evaluate_kitti_range.py --full_range_result path/to/result.pkl
    ```