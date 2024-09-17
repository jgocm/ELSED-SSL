import pyelsed
import numpy as np
import cv2
import pandas as pd
from elsed_analyzer import SegmentsAnalyzer
from camera import Camera
from render import *

M_TO_MM = 1000

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
    idx = 42
    data_dir = img_path[:idx]
    scenario = img_path[idx:idx+6]
    frame_nr = int(img_path[idx+11:-4])
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
    img_height = 850
    annotations_path = 'annotations/segments_annotations.csv'
    camera_intrinsics_path = '/home/joao-dt/ssl-navigation-dataset/configs/intrinsic_parameters.txt'
    camera_matrix = np.loadtxt(camera_intrinsics_path, dtype="float64")

    camera = Camera(camera_matrix)
    camera.setPoseFrom3DModel(height=170, angle=106.7)

    render = Render(img_height, 4500, 6000, 300, (300, 3000))

    df = pd.read_csv(annotations_path)
    analyzer = SegmentsAnalyzer(pyelsed)

    mAP, TP_count, FP_count = 0, 0, 0
    
    for index, row in df.iterrows():
        img_path = row['img_path']
        original_img = cv2.imread(img_path)
        
        robot_pose = get_robot_gt_pose_from_dataset(img_path)
        render.draw_robot(robot=Robot(x_mm=M_TO_MM*robot_pose[0],
                                      y_mm=M_TO_MM*robot_pose[1],
                                      theta_rad=robot_pose[2]))

        s = np.array([row['x0'], row['y0'], row['x1'], row['y1']], dtype=np.int32)

        x1, y1, _ = camera.pixelToRobotCoordinates(s[0], s[1], 0)
        x2, y2, _ = camera.pixelToRobotCoordinates(s[2], s[3], 0)

        global_x1, global_y1 = camera.robotToFieldCoordinates(x1, y1, robot_pose[0], robot_pose[1], robot_pose[2])
        global_x2, global_y2 = camera.robotToFieldCoordinates(x2, y2, robot_pose[0], robot_pose[1], robot_pose[2])
        projected_segment = LineSegment2D(global_x1, global_y1, global_x2, global_y2)
        projected_segment.mul(M_TO_MM)
        render.draw_line_segment(color=COLORS['RED'], segment=projected_segment)

        line_points = analyzer.get_bresenham_line_points(s)   
        
        for p in line_points:
            x, y = p
            original_img[y, x] = analyzer.RED

        new_width = int((original_img.shape[1]/original_img.shape[0]) * img_height)
        resized_img = cv2.resize(original_img, (new_width, img_height))
        full_img = np.hstack((resized_img, render.img))
        cv2.imshow('elsed segments', full_img)
        key = cv2.waitKey(0) & 0xFF
        
        if key==ord('q'):
            break

        render.reset()

    
    print(f'quit')                
