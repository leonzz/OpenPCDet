import copy
import argparse
import pickle 
import numpy as np 
import sys
import os
import argparse

from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
from merge_predictions import merge_results_all, merge_results_within_range

GROUND_TRUTH_FILE = '../../data/kitti/kitti_infos_val.pkl'

def parse_config():
    parser = argparse.ArgumentParser(description='Evaluate kitti results based on saved prediction results.')
    parser.add_argument('--range_cutoff', type=float, default=34.0, help='The cutoff value to merge or split short and long range result.')
    parser.add_argument('--full_range_result', type=str, default=None, help='The full range validation result pickle file.')
    parser.add_argument('--short_range_result', type=str, default=None, help='The short range validation result pickle file.')
    parser.add_argument('--long_range_result', type=str, default=None, help='The short range validation result pickle file.')
    parser.add_argument('--save_merge_result', action='store_true', default=False, help='If present, save merge result pickle file to working directory.')

    args = parser.parse_args()
    return args, parser

def read_pickle_file(filename):
    with open(filename, 'rb') as f:
        content = pickle.load(f)
        return content

def print_eval_result(ap_result_str, ap_dict):
    # print(ap_result_str)
    # print(ap_dict)
    print('Easy: {:.2f}   Moderate: {:.2f}   Hard: {:.2f}\n'.format( ap_dict['Car_3d/easy_R40'], ap_dict['Car_3d/moderate_R40'], ap_dict['Car_3d/hard_R40'] ))

def split_result_by_range(full_range_result: [], cutoff_distance: float)-> ([],[]):
    short_range_result, long_range_result = [], []
    for det in full_range_result:
        short_detections, long_detections = {}, {}
        indexes_for_short = det['location'][:,2] < cutoff_distance - 0.27
        indexes_for_long = det['location'][:,2] > cutoff_distance - 0.27
        for key, val in det.items():
            if key != 'frame_id' and key != 'gt_boxes_lidar':
                short_detections[key] = det[key][indexes_for_short]
                long_detections[key] = det[key][indexes_for_long]
        if 'frame_id' in det:
            short_detections['frame_id'] = det['frame_id']
            long_detections['frame_id'] = det['frame_id']
        short_range_result.append(short_detections)
        long_range_result.append(long_detections)
    return short_range_result, long_range_result

def main():
    args, parser = parse_config()
    current_folder = os.path.abspath(os.path.dirname(__file__)) + '/'

    groundtruth = read_pickle_file(os.path.join(current_folder,GROUND_TRUTH_FILE))
    eval_gt_full = [copy.deepcopy(info['annos']) for info in groundtruth]
    eval_gt_short, eval_gt_long = split_result_by_range(eval_gt_full, args.range_cutoff)

    show_help = True

    # Full range evaluation
    if args.full_range_result is not None:
        show_help = False
        full_range_result = read_pickle_file(args.full_range_result)
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_full, full_range_result, 'Car')
    
        print('\n******** Model trained with full range data ********')
        print_eval_result(ap_result_str, ap_dict)

        print('  - Splitting result to evaluate on range...')
        split_result_short, split_result_long = split_result_by_range(full_range_result, args.range_cutoff)
        
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_short, split_result_short, 'Car')
        print('  - Accuracy on range 0-{:.1f}:'.format(args.range_cutoff))
        print_eval_result(ap_result_str, ap_dict)

        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_long, split_result_long, 'Car')
        print('  - Accuracy on range {:.1f}-70:'.format(args.range_cutoff))
        print_eval_result(ap_result_str, ap_dict)

    # Evaluation of merged result from models handling separate ranges
    if args.long_range_result is not None and args.short_range_result is not None:
        show_help = False
        short_range_result = read_pickle_file(args.short_range_result)
        long_range_result = read_pickle_file(args.long_range_result)

        merged_result = merge_results_all(short_range_result, long_range_result)
        if args.save_merge_result:
            with open('merge_all_result.pkl', 'wb') as f:
                pickle.dump(merged_result, f)
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_full, merged_result, 'Car')
        print('\n******** Merged result from models handling separate ranges ********')
        print('  - Merge all:')
        print_eval_result(ap_result_str, ap_dict)

        merged_result = merge_results_within_range(short_range_result, long_range_result, args.range_cutoff)
        if args.save_merge_result:
            with open('merge_with_cutoff_{:.0f}_result.pkl'.format(args.range_cutoff), 'wb') as f:
                pickle.dump(merged_result, f)
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_full, merged_result, 'Car')
        print('  - Merge with range cutoff at {}:'.format(args.range_cutoff))
        print_eval_result(ap_result_str, ap_dict)

        split_result_short, split_result_long = split_result_by_range(merged_result, args.range_cutoff)
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_short, split_result_short, 'Car')
        print('  - Accuracy on range 0-{:.1f}:'.format(args.range_cutoff))
        print_eval_result(ap_result_str, ap_dict)

        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_long, split_result_long, 'Car')
        print('  - Accuracy on range {:.1f}-70:'.format(args.range_cutoff))
        print_eval_result(ap_result_str, ap_dict)

    if show_help:
        parser.print_help()

if __name__ == '__main__':
    main()
