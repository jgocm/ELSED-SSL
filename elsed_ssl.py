import pyelsed
import numpy as np
import cv2
import time
from elsed_analyzer import SegmentsAnalyzer

if __name__ == "__main__":
    dataset_path = '/home/joao-dt/ssl-navigation-dataset'
    scenarios = ['igs', 'sqr', 'rnd']
    rounds = 3
    max_img_nr = 2000

    boundary_thresholds_path = 'annotations/selected_images/boundary_thresholds_50.npy'
    marking_thresholds_path = 'annotations/selected_images/marking_thresholds_50.npy'
    
    boundary_thresholds = np.load(boundary_thresholds_path)
    marking_thresholds = np.load(marking_thresholds_path)
    print(f'Boundary Thresholds: {boundary_thresholds}')
    print(f'Markings Thresholds: {marking_thresholds}')

    analyzer = SegmentsAnalyzer(pyelsed, boundary_thresholds, marking_thresholds)

    scenario = 'sqr'
    round = 1
    img_nr = 10
    step = 5

    while True:
        #original_img, img_path = get_img_from_dataset(dataset_path, 'rnd', 2, 1566) # this frame is problematic -> test with it
        
        original_img, img_path, img_details = analyzer.get_random_img_from_dataset(dataset_path, scenarios, rounds, max_img_nr)
        #original_img, img_path = analyzer.get_img_from_dataset(dataset_path, scenario, round, img_nr)
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
                elif is_field_boundary:
                    dbg_img[y, x] = analyzer.GREEN                    
                else:
                    dbg_img[y, x] = analyzer.BLACK
        
        merged_img = np.concatenate((gs_img, dbg_img), axis=1)
        cv2.imshow('ELSED-SSL', merged_img)        
        key = cv2.waitKey(0) & 0xFF

        if key==ord('q'):
            break
        elif key==ord('s'):
            #scenario, round, img_nr = img_details
            cv2.imwrite(f'data/selected_images/original/{scenario}_0{round}_{img_nr}_original.png', original_img)
            cv2.imwrite(f'data/selected_images/inference/{scenario}_0{round}_{img_nr}_inference.png', gs_img)
        
        img_nr += step

