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
    parser.add_argument('--full_range_result', type=str, default=None, help='The full range validation result pickle file.')
    parser.add_argument('--short_range_result', type=str, default=None, help='The short range validation result pickle file.')
    parser.add_argument('--long_range_result', type=str, default=None, help='The short range validation result pickle file.')
    parser.add_argument('--merge_range_cutoff', type=float, default=34.0, help='The cutoff value to merge the short and long range result.')

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

def main():
    args, parser = parse_config()
    current_folder = os.path.abspath(os.path.dirname(__file__)) + '/'

    groundtruth = read_pickle_file(os.path.join(current_folder,GROUND_TRUTH_FILE))
    eval_gt_annos = [copy.deepcopy(info['annos']) for info in groundtruth]

    show_help = True

    # Full range evaluation
    if args.full_range_result is not None:
        show_help = False
        full_range_result = read_pickle_file(args.full_range_result)
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, full_range_result, 'Car')
    
        print('\nModel trained with full range data:')
        print_eval_result(ap_result_str, ap_dict)

    # Evaluation of merged result from models handling separate ranges
    if args.long_range_result is not None and args.short_range_result is not None:
        show_help = False
        print('\nMerged result from models handling separate ranges:')
        short_range_result = read_pickle_file(args.short_range_result)
        long_range_result = read_pickle_file(args.long_range_result)

        merged_result = merge_results_all(short_range_result, long_range_result)
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, merged_result, 'Car')
        print('\n- Merge all:')
        print_eval_result(ap_result_str, ap_dict)

        merged_result = merge_results_within_range(short_range_result, long_range_result, args.merge_range_cutoff)
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, merged_result, 'Car')
        print('\n- Merge with range cutoff at {}:'.format(args.merge_range_cutoff))
        print_eval_result(ap_result_str, ap_dict)

    if show_help:
        parser.print_help()

if __name__ == '__main__':
    main()
