import copy
import argparse
import pickle 
import numpy as np 
import sys
import os

from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
from merge_predictions import merge_results_all, merge_results_within_range

PREDICTION_FULL_RANGE_FILE  = 'result_fr.pkl'
PREDICTION_SHORT_RANGE_FILE = 'result_short.pkl'
PREDICTION_LONG_RANGE_FILE  = 'result_long.pkl'

GROUND_TRUTH_FILE = '../../data/kitti/kitti_infos_val.pkl'

def read_pickle_file(filename):
	with open(filename, 'rb') as f:
		content = pickle.load(f)
		return content

def print_eval_result(ap_result_str, ap_dict):
	# print(ap_result_str)
	# print(ap_dict)
	print('Easy: {:.2f}   Moderate: {:.2f}   Hard: {:.2f}\n'.format( ap_dict['Car_3d/easy_R40'], ap_dict['Car_3d/moderate_R40'], ap_dict['Car_3d/hard_R40'] ))

def main():
	current_folder = os.path.abspath(os.path.dirname(__file__)) + '/'

	groundtruth = read_pickle_file(os.path.join(current_folder,GROUND_TRUTH_FILE))
	eval_gt_annos = [copy.deepcopy(info['annos']) for info in groundtruth]    

	# Full range evaluation
	full_range_result = read_pickle_file(os.path.join(current_folder,PREDICTION_FULL_RANGE_FILE))
	ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, full_range_result, 'Car')
	
	print('\nModel trained with full range data:')
	print_eval_result(ap_result_str, ap_dict)

	# Evaluation of merged result from models handling separate ranges
	print('\nMerged result from models handling separate ranges:')
	short_range_result = read_pickle_file(os.path.join(current_folder,PREDICTION_SHORT_RANGE_FILE))
	long_range_result = read_pickle_file(os.path.join(current_folder,PREDICTION_LONG_RANGE_FILE))

	merged_result = merge_results_all(short_range_result, long_range_result)
	ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, merged_result, 'Car')
	print('\n- Merge all:')
	print_eval_result(ap_result_str, ap_dict)

	#for range_cutoff in [30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0]:
	for range_cutoff in [34.0]:
		merged_result = merge_results_within_range(short_range_result, long_range_result, range_cutoff)
		ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, merged_result, 'Car')
		print('\n- Merge with range cutoff at {}:'.format(range_cutoff))
		print_eval_result(ap_result_str, ap_dict)

if __name__ == '__main__':
	main()
