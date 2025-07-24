import cv2
import numpy as np
import os
import pandas as pd

def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=10, gap_length=5):
    line_length = int(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
    dx = (pt2[0] - pt1[0]) / line_length
    dy = (pt2[1] - pt1[1]) / line_length

    for i in range(0, line_length, dash_length + gap_length):
        start_x = int(pt1[0] + dx * i)
        start_y = int(pt1[1] + dy * i)
        end_x = int(pt1[0] + dx * min(i + dash_length, line_length))
        end_y = int(pt1[1] + dy * min(i + dash_length, line_length))
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)

def draw_dotted_line(img, pt1, pt2, color, thickness=2, dot_spacing=10):
    line_length = int(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
    dx = (pt2[0] - pt1[0]) / line_length
    dy = (pt2[1] - pt1[1]) / line_length

    for i in range(0, line_length, dot_spacing):
        cx = int(pt1[0] + dx * i)
        cy = int(pt1[1] + dy * i)
        cv2.circle(img, (cx, cy), 1, color, thickness)

if __name__ == "__main__":
    datasets_labels = ['lines_15pm_lights_off_window_open',
                       'lines_15pm_lights_on_window_open',
                       'lines_15pm_lights_on_window_closed',
                       'humanoid-kid',
                       'humanoid-nao']
    for dataset_label in datasets_labels:
        images_path = f'evaluation/{dataset_label}/images'
        results_path = f'evaluation/{dataset_label}/results'
        csv_files = [f for f in os.listdir(results_path) if f.endswith(('csv'))]

        for csv_file in csv_files:
            csv_path = os.path.join(results_path, csv_file)
            df = pd.read_csv(csv_path)
            if len(df)==0:
                continue
            img_file = df['img_file'][0]
            img_path = os.path.join(images_path, img_file)

            img = cv2.imread(img_path)
            for index, row in df.iterrows():
                img_file,x0,y0,x1,y1,is_field_boundary,is_field_marking,is_field_boundary_gt,is_field_marking_gt = row
                pt1, pt2 = (x0, y0), (x1, y1)
                mid_point = ((x0 + x1) // 2, (y0 + y1) // 2)

                is_true_positive = (is_field_boundary and is_field_boundary_gt) or (is_field_marking and is_field_marking_gt)
                is_false_positive = (is_field_boundary and not is_field_boundary_gt) or (is_field_marking and not is_field_marking_gt)
                is_false_negative = (not is_field_boundary and is_field_boundary_gt) or (not is_field_marking and is_field_marking_gt)

                if is_true_positive:
                    color = (0, 255, 0)  # green
                    label = "TP"
                    cv2.line(img, pt1, pt2, color, 2)  # solid line
                    cv2.putText(img, label, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
                if is_false_positive:
                    color = (0, 0, 255)  # red
                    label = "FP"
                    draw_dashed_line(img, pt1, pt2, color, thickness=2)
                    cv2.putText(img, label, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
                if is_false_negative:
                    color = (255, 0, 0)  # blue
                    label = "FN"
                    draw_dotted_line(img, pt1, pt2, color, thickness=2)
                    cv2.putText(img, label, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)

            cv2.imshow('frame', img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                quit()
            
            result_path = os.path.join(results_path, f'processed_{img_file}')
            cv2.imwrite(result_path, img)
        
