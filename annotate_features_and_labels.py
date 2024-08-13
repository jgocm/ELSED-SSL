import numpy as np
import cv2
import pandas as pd
from elsed_analyzer import SegmentsAnalyzer

if __name__ == "__main__":
    analyzer = SegmentsAnalyzer()
    annotations_path = 'annotations/segments_annotations.csv'
    output_path = 'annotations/features_and_labels.csv'
    df = pd.read_csv(annotations_path)
    
    columns = ['grad_Bx', 
               'grad_Gx', 
               'grad_Rx', 
               'grad_By', 
               'grad_Gy', 
               'grad_Ry', 
               'segment_length', 
               'is_field_boundary_gt', 
               'is_field_marking_gt']
    rows = []
    for index, row in df.iterrows():
        img_path = row['img_path']
        original_img = cv2.imread(img_path)
        
        segment = np.array([row['x0'], row['y0'], row['x1'], row['y1']], dtype=np.int32)
        is_field_boundary_gt = row['is_field_boundary']
        is_field_marking_gt = row['is_field_marking']

        line_points = analyzer.get_bresenham_line_points(segment)
        segment_length = len(line_points)
        
        gx, gy = analyzer.get_gradients_from_line_points(original_img, line_points)
        if gx is None and gy is None:
            import pdb;pdb.set_trace()
            
        grad_Bx, grad_Gx, grad_Rx = gx
        grad_By, grad_Gy, grad_Ry = gy

        row = [grad_Bx, grad_Gx, grad_Rx, grad_By, grad_Gy, grad_Ry, segment_length, is_field_boundary_gt, is_field_marking_gt]        
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path, mode='w', header=True, index=False)
    print(f'finished saving labels for {index+1} line segments')
