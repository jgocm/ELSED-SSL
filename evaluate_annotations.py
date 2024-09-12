import pyelsed
import numpy as np
import cv2
import pandas as pd
from elsed_analyzer import SegmentsAnalyzer

if __name__ == "__main__":
    analyzer = SegmentsAnalyzer(pyelsed)
    annotations_path = 'annotations/segments_annotations.csv'
    boundary_thresholds_path = 'annotations/optimal_boundary_thresholds.npy'
    marking_thresholds_path = 'annotations/optimal_marking_thresholds.npy'
    df = pd.read_csv(annotations_path)
    
    boundary_thresholds = np.load(boundary_thresholds_path)
    print(boundary_thresholds)
    marking_thresholds = np.load(marking_thresholds_path)
    print(marking_thresholds)

    # CONFIG THRESHOLDS
    boundary_grad_th, boundary_angle_threshold_deg, boundary_min_seg_len = boundary_thresholds
    markings_grad_th, markings_angle_threshold_deg, markings_min_seg_len = marking_thresholds
    
    mAP, TP_count, FP_count = 0, 0, 0
    
    for index, row in df.iterrows():
        img_path = row['img_path']
        original_img = cv2.imread(img_path)
        dbg_img = original_img.copy()

        segment = np.array([row['x0'], row['y0'], row['x1'], row['y1']], dtype=np.int32)
        is_field_boundary_gt = row['is_field_boundary']
        is_field_marking_gt = row['is_field_marking']
        
        gx = np.array([row['grad_Bx'],row['grad_Gx'],row['grad_Rx']])
        gy = -np.array([row['grad_By'],row['grad_Gy'],row['grad_Ry']])
        segment_length = row['segment_length']

        line_points = analyzer.get_bresenham_line_points(segment)
        
        is_field_boundary = analyzer.check_boundary_classification(g = gy, 
                                                                   l = segment_length,
                                                                   gradient_threshold = boundary_grad_th,
                                                                   angle_threshold_deg = boundary_angle_threshold_deg,
                                                                   min_segment_length = 150)
        is_field_marking = analyzer.check_marking_classification(g = gy, 
                                                                 l = segment_length,
                                                                 gradient_threshold = markings_grad_th,
                                                                 angle_threshold_deg = markings_angle_threshold_deg,
                                                                 min_segment_length = 150)
        
        is_true_positive = (is_field_boundary and is_field_boundary_gt) or (is_field_marking and is_field_marking_gt)
        is_false_positive = (is_field_boundary and not is_field_boundary_gt) or (is_field_marking and not is_field_marking_gt)
        
        draw_color = analyzer.RED
        
        if is_true_positive:
            draw_color = analyzer.GREEN
            TP_count += 1
            #print("True Positive")
            
        if is_false_positive: 
            draw_color = analyzer.RED
            FP_count += 1
            #print("False positive")
        
        for p in line_points:
            x, y = p
            dbg_img[y, x] = draw_color
        
        if TP_count+FP_count > 0:
            mAP = TP_count/(TP_count+FP_count)

        cv2.imshow('elsed segments', dbg_img)
        if is_false_positive:
            print(f'False Positive detected on row nr {index+2}: {segment}')
            print(f'            Boundary  Marking')
            print(f'Inference:    {is_field_boundary},   {is_field_marking}')
            print(f'Ground truth: {is_field_boundary_gt},   {is_field_marking_gt}')
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(1) & 0xFF
        
        if key==ord('q'):
            break
    
    print(f'mAP: {mAP}, TP: {TP_count}, FP: {FP_count}, total lines: {index+1}')
