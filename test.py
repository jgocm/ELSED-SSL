import pyelsed
import numpy as np
import cv2
import time
import os
from elsed_analyzer import SegmentsAnalyzer

if __name__ == "__main__":
    dataset_path = '/home/joao-dt/humanoid-dataset/kid'
    image_files = [f for f in os.listdir(dataset_path) if f.endswith(('png', 'jpg', 'jpeg'))]

    marking_thresholds_path = 'annotations/humanoid/marking_thresholds_50.npy'
    
    marking_thresholds = np.load(marking_thresholds_path)
    print(f'Markings Thresholds: {marking_thresholds}')

    analyzer = SegmentsAnalyzer(pyelsed, marking_thresholds=marking_thresholds)

    for img_file in image_files:
        img_path = os.path.join(dataset_path, img_file)
        original_img = cv2.imread(img_path)
        gs_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        gs_img = cv2.cvtColor(gs_img, cv2.COLOR_GRAY2BGR)
        dbg_img = original_img.copy()
        print(f"Img: {img_path}")
        
        t0 = time.time()
        segments, scores, labels, grads_x, grads_y = analyzer.detect(original_img, gradientThreshold=30, minLineLen=40)
        print(f'elapsed_time: {time.time() - t0}')
        
        for s, score, label, grad_x, grad_y in zip(segments.astype(np.int32), scores, labels, grads_x, grads_y):
            line_points = analyzer.get_bresenham_line_points(s)

            is_field_boundary = (label==1)
            is_field_marking = (label==2)

            for p in line_points:
                x, y = p
                gs_img[y, x] = analyzer.RED
                
                if is_field_marking:
                    dbg_img[y, x] = analyzer.RED                  
                else:
                    dbg_img[y, x] = analyzer.BLACK
        
        merged_img = np.concatenate((gs_img, dbg_img), axis=1)
        cv2.imshow('ELSED-SSL', merged_img)
        key = cv2.waitKey(0) & 0xFF

        if key==ord('q'):
            break        
