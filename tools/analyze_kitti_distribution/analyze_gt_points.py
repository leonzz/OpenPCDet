import os
import pickle
import matplotlib.pyplot as plt
import numpy as np




def get_objPix_objPts_objRange(class_name):
	obj_pixels = []
	obj_points = []
	obj_range = []
	for sample in dataset_info:
		num_of_objects = sample['annos']['name'].size
		for i in range(0, num_of_objects):
			if sample['annos']['name'][i] == class_name:
				bbox = sample['annos']['bbox'][i, :]
				gt_box = sample['annos']['gt_boxes_lidar'][i, :]
				curr_range = np.sqrt(gt_box[0]*gt_box[0] + gt_box[1]*gt_box[1] + gt_box[3]*gt_box[3])
				curr_pixels = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
				curr_points = sample['annos']['num_points_in_gt'][i]
				obj_pixels.append(curr_pixels)
				obj_points.append(curr_points)
				obj_range.append(curr_range)
	return obj_pixels, obj_points, obj_range

def plot_range(obj_range):
	obj_range_arr = np.array(obj_range)
	num_20 = ((0 < obj_range_arr) & (obj_range_arr <= 20)).sum()
	num_40 = ((20 < obj_range_arr) & (obj_range_arr <= 40)).sum()
	num_60 = ((40 < obj_range_arr) & (obj_range_arr <= 60)).sum()
	num_80 = ((60 < obj_range_arr)).sum()
	total_num_obj = obj_range_arr.shape[0]
	# figure i
	plt.figure(2)
	plt.hist(obj_range, bins=16)
	plt.grid(0.01)
	plt.xlabel('Range Bins')
	plt.ylabel('Number of {}s'.format(class_name))
	plt.title('Num of Objects 0.0 to 20.0 m: {} - {:.2f}% \n Num of Objects 20.0 to 40.0 m: {} - {:.2f}% \n Num of Objects 40.0 to 60.0 m: {} - {:.2f}% \n Num of Objects 60.0 to 80.0 m: {} - {:.2f}% \n'
		.format(num_20 , (num_20/total_num_obj)*100, num_40, (num_40/total_num_obj)*100, num_60, (num_60/total_num_obj)*100, num_80, (num_80/total_num_obj)*100))
	plt.show()
	#plt.savefig('{}.png'.format(class_name))

def plot_range(obj_range_train, obj_range_val):    
	# figure i
	plt.figure(2)
	plt.hist(obj_range_train, bins=16)
	plt.hist(obj_range_val, bins=16, alpha=0.5)
	plt.grid(0.01)
	plt.xlabel('Range Bins')
	plt.ylabel('Number of {}s'.format(class_name))
	plt.legend(['Training set','Validation set'])
	plt.show()

if __name__ == "__main__":
	class_name = 'Car'
	# extract train data
	dataset_name = 'kitti'
	pkl_file = '{}_infos_train.pkl'.format(dataset_name)
	data_folder = '/home/nas/OpenPCDet-Traffic/data/kitti/'
	data_full_path = os.path.join(data_folder, pkl_file)
	with open(data_full_path, 'rb') as f:
		dataset_info = pickle.load(f)
	obj_pixels_train, obj_points_train, obj_range_train = get_objPix_objPts_objRange(class_name)
	# extract val data
	dataset_name = 'kitti'
	pkl_file = '{}_infos_val.pkl'.format(dataset_name)
	data_folder = '/home/nas/OpenPCDet-Traffic/data/kitti/'
	data_full_path = os.path.join(data_folder, pkl_file)
	with open(data_full_path, 'rb') as f:
		dataset_info = pickle.load(f)
	obj_pixels_val, obj_points_val, obj_range_val = get_objPix_objPts_objRange(class_name)
	
	plot_range(obj_range_train, obj_range_val)
	


