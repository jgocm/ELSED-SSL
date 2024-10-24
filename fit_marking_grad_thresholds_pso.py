import pyelsed
import numpy as np
import cv2
import pandas as pd
from elsed_analyzer import SegmentsAnalyzer
from pyswarm import pso

# Define the function to calculate mAP given a set of thresholds
def calculate_map(thresholds):    
    analyzer = SegmentsAnalyzer(pyelsed, marking_thresholds=thresholds)
    dataset_path = "annotations/humanoid/segments_annotations.csv"
    df = pd.read_csv(dataset_path)[:]
    
    TP_count, FP_count = 0, 0

    for index, row in df.iterrows():
        gx = np.array([row['grad_Bx'], row['grad_Gx'], row['grad_Rx']], dtype=np.float32)
        gy = np.array([row['grad_By'], row['grad_Gy'], row['grad_Ry']], dtype=np.float32)
        segment_length = row['segment_length']
        is_field_marking_gt = row['is_field_marking']
        
        label = analyzer.classify(gx, -gy, segment_length)
        
        is_field_marking = (label==2)

        is_true_positive = (is_field_marking and is_field_marking_gt)
        is_false_positive = (is_field_marking and not is_field_marking_gt)
        
        if is_true_positive:
            TP_count += 1
            
        if is_false_positive: 
            FP_count += 1
    
    print(f'thresholds: {thresholds} | TP: {TP_count} | FP: {FP_count} | Score: {TP_count - FP_count}')

    return -(TP_count - FP_count)

if __name__ == "__main__":
    # Define the bounds for the thresholds
    lb = [10, 20, 50]  # Lower bounds
    ub = [255, 70, 300]  # Upper bounds

    thresholds_path = "data/humanoid/annotations/optimal_marking_thresholds.npy"

    # Run PSO to optimize the thresholds
    optimal_thresholds, optimal_score = pso(calculate_map, lb, ub, swarmsize=100, maxiter=10, omega=0.1)

    # Print the optimal thresholds and the corresponding mAP
    print(f'Optimal thresholds: {optimal_thresholds}')
    print(f'Optimal score: {-optimal_score}')  # Negate again to get the positive mAP

    #np.save(thresholds_path, optimal_thresholds)
