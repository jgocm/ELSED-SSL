import pyelsed
import numpy as np
import cv2
import pandas as pd
from elsed_analyzer import SegmentsAnalyzer

if __name__ == "__main__":
    analyzer = SegmentsAnalyzer(pyelsed)
    annotations_path = '/home/rc-blackout/Documents/PhD/cadeiras/visao-computacional/ELSED-SSL/segments_annotations.csv'
    df = pd.read_csv(annotations_path)
    
    mAP, TP_count, FP_count = 0, 0, 0
    
    for index, row in df.iterrows():
        img_path = row['img_path']
        original_img = cv2.imread(img_path)
        dbg_img = original_img.copy()
        
        segment = np.array([row['x0'], row['y0'], row['x1'], row['y1']], dtype=np.int32)
        is_field_boundary_gt = row['is_field_boundary']
        is_field_marking_gt = row['is_field_marking']

        line_points = analyzer.get_bresenham_line_points(segment)
        
        gx, gy = analyzer.get_gradients_from_line_points(original_img, line_points)
        if np.linalg.norm(gy)>np.linalg.norm(gx): g = gy
        else: g = gx
        is_field_boundary = analyzer.check_boundary_classification(gy, len(line_points))
        is_field_marking = analyzer.check_marking_classification(g, len(line_points))
        
        is_true_positive = (is_field_boundary and is_field_boundary_gt) or (is_field_marking and is_field_marking_gt)
        is_false_positive = (is_field_boundary and not is_field_boundary_gt) or (is_field_marking and not is_field_marking_gt)
        
        draw_color = analyzer.BLACK
        
        if is_true_positive:
            draw_color = analyzer.GREEN
            TP_count += 1
            print("True Positive")
            
        if is_false_positive: 
            draw_color = analyzer.RED
            FP_count += 1
            print("False positive")
        
        for p in line_points:
            x, y = p
            dbg_img[y, x] = draw_color
        
        if TP_count+FP_count > 0:
            mAP = TP_count/(TP_count+FP_count)
        
        cv2.imshow('elsed segments', dbg_img)
        key = cv2.waitKey(100) & 0xFF
        
        if key==ord('q'):
            print(f'mAP: {mAP}')
            quit()
    
    print(f'mAP: {mAP}')                
