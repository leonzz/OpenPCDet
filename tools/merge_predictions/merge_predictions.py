## naively merge two pickle files
import pickle 
import numpy as np 
import sys

def merge_results(short_detections: [], long_detections: [])-> []:
	merged_results = []
	for short_det, long_det in zip(short_detections, long_detections):
		frame_merged_detections = short_det.copy()
		for key,val in short_det.items():
			if key != 'frame_id':
				frame_merged_detections[key] = np.concatenate([frame_merged_detections[key], long_det[key]], axis=0)

		merged_results.append(frame_merged_detections)
	return merged_results	
	
prediction_short_range_file = './result_short.pkl'
prediction_long_range_file = './result_long.pkl'

with open(prediction_short_range_file, 'rb') as f:
	prediction_short_range = pickle.load(f)

with open(prediction_long_range_file, 'rb') as f:
	prediction_long_range = pickle.load(f)


merged_results = merge_results(short_detections=prediction_short_range, long_detections=prediction_long_range)
merged_prediction_file = './result_merged_test.pkl'
with open(merged_prediction_file, 'wb') as f:
	pickle.dump(merged_results, f)
