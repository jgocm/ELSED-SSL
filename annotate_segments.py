import numpy as np
import cv2
import pandas as pd
from elsed_analyzer import SegmentsAnalyzer

if __name__ == "__main__":
    analyzer = SegmentsAnalyzer()
    dataset_path = "/home/joao-dt/ssl-navigation-dataset"
    annotations_path = "annotations/segments_annotations.csv"
    scenarios = ["rnd", "sqr", "igs"]
    rounds = 3
    max_img_nr = 2000
    columns = ['img_path', 
               'x0', 
               'y0', 
               'x1', 
               'y1',
               'grad_Bx',
               'grad_Gx',
               'grad_Rx',
               'grad_By',
               'grad_Gy',
               'grad_Ry',
               'segment_length',
               'is_field_boundary', 
               'is_field_marking',
               'is_not_a_field_feature']
    
    annotations = []
    while True:
        original_img, img_path, img_details = analyzer.get_random_img_from_dataset(dataset_path, scenarios, rounds, max_img_nr)
        print(f"Img: {img_path}")
        
        segments, _, _, grads_x, grads_y = analyzer.segments_detector.detect(original_img, 1, 30, 40)
        
        for s, grad_x, grad_y in zip(segments.astype(np.int32), grads_x, grads_y):
            dbg_img = original_img.copy()

            x0, y0, x1, y1 = s[0], s[1], s[2], s[3]
            line_points = analyzer.get_bresenham_line_points(s)
            
            for p in line_points:
                x, y = p
                dbg_img[y, x] = analyzer.RED
                    
            is_field_boundary = False
            is_field_marking = False
            is_not_a_field_feature = False

            grad_Bx = grad_x[0]
            grad_Gx = grad_x[1]
            grad_Rx = grad_x[2]
            grad_By = grad_y[0]
            grad_Gy = grad_y[1]
            grad_Ry = grad_y[2]

            segment_length = len(line_points)

            cv2.imshow('elsed segments', dbg_img)
            key = cv2.waitKey(0) & 0xFF
            
            if key==ord('q'):
                # quit
                quit()
            elif key==ord('s'):
                # skip and save current annotations
                df = pd.DataFrame(annotations, columns=columns)
                df.to_csv(annotations_path, mode='a', header=False, index=False)
                print(f"updated annotations at: {annotations_path}")
                annotations = []
                break
            elif key==ord('r'):
                # remove last annotation (in case it was wrong)
                annotations.pop()
            else:
                if key==ord('b'):
                    # annotate as field boundary
                    is_field_boundary = True
                if key==ord('m'):
                    # annotate as field marking
                    is_field_marking = True
                
                is_not_a_field_feature = (not is_field_boundary and not is_field_marking)
                annotation = [img_path, 
                              x0, 
                              y0, 
                              x1, 
                              y1,
                              grad_Bx,
                              grad_Gx,
                              grad_Rx,
                              grad_By,
                              grad_Gy,
                              grad_Ry,
                              segment_length,
                              is_field_boundary, 
                              is_field_marking, 
                              is_not_a_field_feature]
                annotations.append(annotation)
                
                
            