import pyelsed
import numpy as np
import cv2
import pandas as pd
from elsed_analyzer import SegmentsAnalyzer
from pyswarm import pso

# Define the function to calculate mAP given a set of thresholds
def calculate_map(thresholds):
    boundary_grad_th, boundary_angle_threshold_deg, boundary_min_seg_len = thresholds
    
    analyzer = SegmentsAnalyzer(pyelsed)
    dataset_path = 'annotations/segments_annotations.csv'
    df = pd.read_csv(dataset_path)[:100]
    
    mAP, TP_count, FP_count = 0, 0, 0

    for index, row in df.iterrows():
        gx = np.array([row['grad_Bx'], row['grad_Gx'], row['grad_Rx']])
        gy = -np.array([row['grad_By'], row['grad_Gy'], row['grad_Ry']])
        segment_length = row['segment_length']
        is_field_boundary_gt = row['is_field_boundary']
        
        is_field_boundary = analyzer.check_boundary_classification(g=gy, 
                                                                   l=segment_length,
                                                                   gradient_threshold=boundary_grad_th,
                                                                   angle_threshold_deg=boundary_angle_threshold_deg,
                                                                   min_segment_length=boundary_min_seg_len)
        
        is_true_positive = (is_field_boundary and is_field_boundary_gt)
        is_false_positive = (is_field_boundary and not is_field_boundary_gt)
        
        if is_true_positive:
            TP_count += 1
            
        if is_false_positive: 
            FP_count += 1
        
    if TP_count + FP_count > 0:
        mAP = TP_count / (TP_count + FP_count)
    
    print(f'thresholds: {thresholds} | mAP: {mAP}')

    return -(TP_count - FP_count)

# Define the bounds for the thresholds
lb = [10, 20, 50]  # Lower bounds
ub = [255, 70, 300]  # Upper bounds

thresholds_path = 'annotations/optimal_boundary_thresholds.npy'

# Run PSO to optimize the thresholds
optimal_thresholds, optimal_mAP = pso(calculate_map, lb, ub, swarmsize=50, maxiter=10)

# Print the optimal thresholds and the corresponding mAP
print(f'Optimal thresholds: {optimal_thresholds}')
print(f'Optimal mAP: {-optimal_mAP}')  # Negate again to get the positive mAP

np.save(thresholds_path, optimal_thresholds)
