import pyelsed
import numpy as np
import cv2
from elsed_analyzer import SegmentsAnalyzer
import pandas as pd
import utils
import os

if __name__ == "__main__":
    # load json with paths
    config = utils.load_config_file('configs.json')
    dataset_label = config['dataset_label']
    images_path = config['paths']['eval_images']
    boundary_thresholds_path = config['paths']['boundary_thresholds']
    marking_thresholds_path = config['paths']['marking_thresholds']
    results_path = f'evaluation/{dataset_label}/results'
    os.makedirs(results_path, exist_ok=True)
    
    # load images and thresholds
    image_files = [f for f in os.listdir(images_path) if f.endswith(('png', 'jpg', 'jpeg'))]
    boundary_thresholds = np.load(boundary_thresholds_path)
    marking_thresholds = np.load(marking_thresholds_path)

    # load ELSED
    analyzer = SegmentsAnalyzer(pyelsed,
                                boundary_thresholds,
                                marking_thresholds)
    
    # configure csv with results
    columns = ['img_file',
               'x0', 
               'y0', 
               'x1', 
               'y1',
               'is_field_boundary',
               'is_field_marking',
               'is_field_boundary_gt',
               'is_field_marking_gt']
    
    for img_file in image_files:
        img_path = os.path.join(images_path, img_file)
        original_img = cv2.imread(img_path)
        result_img = original_img.copy()

        update = True
        
        segments, scores, labels, grads_x, grads_y = analyzer.detect(original_img, gradientThreshold=30, minLineLen=40)
        
        img_result = []

        file_exists = os.path.exists(f'{results_path}/processed_{img_file}')
        if file_exists:
            print('file already exists')
        else:
            print('file does not exist')

        for s, score, label in zip(segments.astype(np.int32), scores, labels):
            dbg_img = original_img.copy()

            is_field_boundary_gt = False
            is_field_marking_gt = False
            
            is_field_boundary = (label==1)[0]
            is_field_marking = (label==2)[0]

            cv2.line(dbg_img, s[:2], s[2:], analyzer.RED.tolist(), 2)
            
            concatenated_image = np.hstack((dbg_img, result_img))
            cv2.imshow('result', concatenated_image)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                quit()
            elif key == ord('s'):
                update = False
                break
            elif key == ord('m'):
                is_field_marking_gt = True
            elif key == ord('b'):
                is_field_boundary_gt = True

            # Check field boundary
            is_true_positive = (is_field_boundary and is_field_boundary_gt) or (is_field_marking and is_field_marking_gt)
            is_false_positive = (is_field_boundary and not is_field_boundary_gt) or (is_field_marking and not is_field_marking_gt)
            is_false_negative = (not is_field_boundary and is_field_boundary_gt) or (not is_field_marking and is_field_marking_gt)

            if is_true_positive:
                cv2.line(result_img, s[:2], s[2:], analyzer.GREEN.tolist(), 2)
            elif is_false_positive or is_false_negative:
                cv2.line(result_img, s[:2], s[2:], analyzer.RED.tolist(), 2)

            concatenated_image = np.hstack((dbg_img, result_img))
            cv2.imshow('result', concatenated_image)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                quit()
            
            x0, y0, x1, y1 = s[0], s[1], s[2], s[3]
            result = [img_file, x0, y0, x1, y1, is_field_boundary, is_field_marking, is_field_boundary_gt, is_field_marking_gt]
            img_result.append(result)

        if update:
            cv2.imwrite(f'{results_path}/processed_{img_file}', result_img)

            df = pd.DataFrame(img_result, columns=columns)
            df.to_csv(f'{results_path}/results_{img_file[:-4]}.csv', mode='w', header=True, index=False)
