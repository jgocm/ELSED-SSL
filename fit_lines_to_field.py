import pyelsed
import numpy as np
import cv2
import pandas as pd
from elsed_analyzer import SegmentsAnalyzer

class LineSegment2D:
    def __init__(self,
                 x1: float,
                 y1: float,
                 x2: float,
                 y2: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self._coef = self.get_coef()
        
    def get_coef(self):
        return np.arctan2(self.y2-self.y1, self.x2-self.x1)

field_markings = {"TopTouchLine": LineSegment2D(x1=0, y1=2700, x2=3900, y2=2700),
                  "BottomTouchLine": LineSegment2D(x1=0, y1=-2700, x2=3900, y2=-2700),
                  "RightGoalLine": LineSegment2D(x1=3900, y1=2700, x2=3900, y2=-2700),
                  "HalfwayLine": LineSegment2D(x1=0, y1=-2700, x2=0, y2=2700),
                  "CenterLine": LineSegment2D(x1=0, y1=0, x2=3900, y2=0),
                  "RightPenaltyStretch": LineSegment2D(x1=3000, y1=-900, x2=3000, y2=900),
                  "RightFieldRightPenaltyStretch": LineSegment2D(x1=3900, y1=-900, x2=3000, y2=-900),
                  "RightFieldLeftPenaltyStretch": LineSegment2D(x1=3900, y1=900, x2=3000, y2=900)}

class FieldBoundaries:
    TopBoundaryLine = LineSegment2D(x1=-300, y1=3000, x2=4200, y2=3000)
    BottomBoundaryLine = LineSegment2D(x1=-300, y1=-3000, x2=4200, y2=-3000)
    RightBoundaryLine = LineSegment2D(x1=4200, y1=3000, x2=4200, y2=-3000)
    LeftBoundaryLine = LineSegment2D(x1=-300, y1=3000, x2=-300, y2=-3000)

def project_line_from_robot_pose(x, y, theta):
    coef = np.tan(np.deg2rad(theta))
    intercept = y - coef*x
    return coef, intercept

def intercept_vertical_line(a, b, line_x):
    x = line_x
    y = a*x + b
    return x, y

def intercept_horizontal_line(a, b, line_y):
    y = line_y
    if a==0: x=100000
    else: x = (y-b)/a
    return x, y    

def get_log_file_from_img_path(img_path):
    data_dir = img_path[:46]
    scenario = img_path[46:52]
    frame_nr = int(img_path[57:-4])
    log_path = data_dir + scenario + '/logs/processed.csv'
    df = pd.read_csv(log_path)
    for index, row in df.iterrows():
        if row['FRAME_NR'] == frame_nr: 
            frame_data = row
            break
            
    return frame_data

def get_robot_gt_pose_from_dataset(img_path):
    frame_data = get_log_file_from_img_path(img_path)
    x, y, theta = frame_data['POSITION X'], frame_data['POSITION Y'], frame_data['POSITION THETA']
    return np.array([x, y, theta])

if __name__ == "__main__":
    analyzer = SegmentsAnalyzer(pyelsed)
    annotations_path = '/home/rc-blackout/Documents/PhD/cadeiras/visao-computacional/ELSED-SSL/segments_annotations.csv'
    df = pd.read_csv(annotations_path)
    
    mAP, TP_count, FP_count = 0, 0, 0
    
    for index, row in df.iterrows():
        img_path = row['img_path']
        original_img = cv2.imread(img_path)
        dbg_img = original_img.copy()
        
        robot_pose = get_robot_gt_pose_from_dataset(img_path)
        
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
        key = cv2.waitKey(0) & 0xFF
        
        if key==ord('q'):
            print(f'mAP: {mAP}')
            quit()
    
    print(f'mAP: {mAP}')                
