import pyelsed
import numpy as np
import cv2
import pandas as pd
from elsed_analyzer import SegmentsAnalyzer
from pyswarm import pso

# Define the function to calculate mAP given a set of thresholds
def calculate_map(thresholds):
    marking_grad_th, marking_angle_threshold_deg, marking_min_seg_len = thresholds
    
    analyzer = SegmentsAnalyzer(pyelsed)
    dataset_path = 'annotations/features_and_labels.csv'
    df = pd.read_csv(dataset_path)
    
    mAP, TP_count, FP_count = 0, 0, 0

    for index, row in df.iterrows():
        gx = np.array([row['grad_Bx'], row['grad_Gx'], row['grad_Rx']])
        gy = np.array([row['grad_By'], row['grad_Gy'], row['grad_Ry']])
        segment_length = row['segment_length']
        is_field_marking_gt = row['is_field_marking_gt']
        
        is_field_marking = analyzer.check_marking_classification(g=gy, 
                                                                   l=segment_length,
                                                                   gradient_threshold=marking_grad_th,
                                                                   angle_threshold_deg=marking_angle_threshold_deg,
                                                                   min_segment_length=marking_min_seg_len)
        
        is_true_positive = (is_field_marking and is_field_marking_gt)
        is_false_positive = (is_field_marking and not is_field_marking_gt)
        
        if is_true_positive:
            TP_count += 1
            
        if is_false_positive: 
            FP_count += 1
        
    if TP_count + FP_count > 0:
        mAP = TP_count / (TP_count + FP_count)
    
    print(f'thresholds: {thresholds} | mAP: {mAP}')

    return -(TP_count - FP_count)

# Define the bounds for the thresholds
lb = [5000, 10, 50]  # Lower bounds
ub = [15000, 90, 300]  # Upper bounds

thresholds_path = 'annotations/optimal_marking_thresholds.npy'

# Run PSO to optimize the thresholds
optimal_thresholds, optimal_mAP = pso(calculate_map, lb, ub, swarmsize=30, maxiter=5)

# Print the optimal thresholds and the corresponding mAP
print(f'Optimal thresholds: {optimal_thresholds}')
print(f'Optimal mAP: {-optimal_mAP}')  # Negate again to get the positive mAP

np.save(thresholds_path, optimal_thresholds)
