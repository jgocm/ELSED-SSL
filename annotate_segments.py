import numpy as np
import cv2
import os
import pandas as pd
from elsed_analyzer import SegmentsAnalyzer
import utils

if __name__ == "__main__":
    analyzer = SegmentsAnalyzer()
    paths = utils.load_paths_from_config_file("configs.json")
    dataset_path = paths["images"]
    annotations_path = paths["segments_annotations"]

    directory = os.path.dirname(annotations_path)

    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' has been created.")

    if not os.path.isfile(annotations_path):
        with open(annotations_path, "w") as file:
            file.write("img_path,x0,y0,x1,y1,grad_Bx,grad_Gx,grad_Rx,grad_By,grad_Gy,grad_Ry,segment_length,is_field_boundary,is_field_marking,is_not_a_field_feature\n")
        print(f"File '{annotations_path}' created with the header.")
    else:
        print(f"File '{annotations_path}' already exists.")


    image_files = [f for f in os.listdir(dataset_path) if f.endswith(('png', 'jpg', 'jpeg'))]
    image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

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
    for img_file in image_files:
        img_path = os.path.join(dataset_path, img_file)
        original_img = cv2.imread(img_path)
        if original_img is None:
            continue
        print(f"Img: {img_path}")
        
        segments, scores, labels, grads_x, grads_y = analyzer.detect(original_img, gradientThreshold=30, minLineLen=40)
        
        for s, score, label, grad_x, grad_y in zip(segments.astype(np.int32), scores, labels, grads_x, grads_y):
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

            segment_length = score[0]

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
            elif key==ord('d'):
                # discard the current line segment
                continue
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
                
                
    df = pd.DataFrame(annotations, columns=columns)
    df.to_csv(annotations_path, mode='a', header=False, index=False)
    print(f"updated annotations at: {annotations_path}")