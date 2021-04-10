## naively merge two pickle files
import pickle 
import numpy as np 
import sys

def merge_results_all(short_detections: [], long_detections: [])-> []:
	merged_results = []
	for short_det, long_det in zip(short_detections, long_detections):
		frame_merged_detections = short_det.copy()
		for key, _ in short_det.items():
			if key != 'frame_id':
				frame_merged_detections[key] = np.concatenate([frame_merged_detections[key], long_det[key]], axis=0)

		merged_results.append(frame_merged_detections)
	return merged_results	

def merge_results_within_range(short_detections: [], long_detections: [], cutoff_distance: float)-> []:
	merged_results = []
	for short_det, long_det in zip(short_detections, long_detections):
		frame_merged_detections = {
			'name': [],
			'truncated': [],
			'occluded': [],
			'alpha': [],
			'bbox': [],
			'dimensions': [],
			'location': [],
			'rotation_y': [],
			'score': [],
			'boxes_lidar': [],
			'frame_id': short_det['frame_id'],
		}
		# Z in camera frame + 0.27 = x in LIDAR frame, ref: http://www.cvlibs.net/datasets/kitti/setup.php
		indexes_in_short = short_det['location'][:,2] < cutoff_distance - 0.27
		indexes_in_long = long_det['location'][:,2] > cutoff_distance - 0.27
		for key, val in short_det.items():
			if key != 'frame_id':
				frame_merged_detections[key] = np.concatenate((short_det[key][indexes_in_short], long_det[key][indexes_in_long]))
		merged_results.append(frame_merged_detections)
	return merged_results	

def main():
	prediction_short_range_file = './result_short.pkl'
	prediction_long_range_file = './result_long.pkl'

	with open(prediction_short_range_file, 'rb') as f:
		prediction_short_range = pickle.load(f)

	with open(prediction_long_range_file, 'rb') as f:
		prediction_long_range = pickle.load(f)


	merged_results = merge_results_all(short_detections=prediction_short_range, long_detections=prediction_long_range)
	merged_prediction_file = './result_merged_test.pkl'
	with open(merged_prediction_file, 'wb') as f:
		pickle.dump(merged_results, f)

if __name__ == '__main__':
	main()