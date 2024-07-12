import pyelsed
import numpy as np
import cv2
from scipy.signal import convolve2d
import random
import os
import pandas as pd

class SegmentsAnalyzer():
    def __init__(self,
                 segments_detector = pyelsed):
        
        self.segments_detector = segments_detector
            
        self.WHITE = np.array([255, 255, 255])
        self.BLUE =  np.array([255, 0, 0])
        self.GREEN = np.array([0, 255, 0])
        self.RED =   np.array([0, 0, 255])
        self.BLACK = np.array([0, 0, 0])

    def get_line_parameters_from_endpoints(self, p1, p2):
        length = np.linalg.norm(p2-p1)
        r = (p2-p1)/length
        p = p1
        return r, p, length

    def get_line_points(self, s):
        p1 = np.array([s[0], s[1]])
        p2 = np.array([s[2], s[3]])
        r, p, segment_length = self.get_line_parameters_from_endpoints(p1, p2)
        
        lambdas = np.arange(round(segment_length))
        lambdas_column = lambdas[:, np.newaxis]
        line_points = (p + r*lambdas_column).astype(np.int32)
        return line_points

    def get_bresenham_line_points(self, s):
        """Bresenham's line algorithm."""
        x0, y0, x1, y1 = s[0], s[1], s[2], s[3]
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return np.array(points)

    def get_gradients_from_line_points(self, src, line_points):
        operator_x = 1/4*np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
        operator_y = 1/4*np.array([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]])
        
        g_x = []
        g_y = []
        for pixel in line_points[2:-2]:
            x, y = pixel
            window = src[y-1:y+2, x-1:x+2]
            B, G, R = window[:,:,0], window[:,:,1], window[:,:,2]
            convolved_Bx = convolve2d(B, operator_x, mode='valid')[0,0]
            convolved_Gx = convolve2d(G, operator_x, mode='valid')[0,0]
            convolved_Rx = convolve2d(R, operator_x, mode='valid')[0,0]
            convolved_By = convolve2d(B, operator_y, mode='valid')[0,0]
            convolved_Gy = convolve2d(G, operator_y, mode='valid')[0,0]
            convolved_Ry = convolve2d(R, operator_y, mode='valid')[0,0]
            g_BGRx = np.array([convolved_Bx, convolved_Gx, convolved_Rx])     
            g_BGRy = np.array([convolved_By, convolved_Gy, convolved_Ry])     
            g_x.append(g_BGRx)
            g_y.append(g_BGRy)
        
        return np.mean(g_x, axis=0), np.mean(g_y, axis=0)

    def get_img_from_dataset(self, dataset_path='/home/rc-blackout/ssl-navigation-dataset', scenario='rnd', round=2, img_nr=100, gray_scale=False):
        img_path = f"{dataset_path}/data/{scenario}_0{round}/cam/{img_nr}.png"
        if os.path.isfile(img_path):
            if gray_scale:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(img_path)
        else:
            print(f'Img {img_path} not available')
            img = None
            
        return img, img_path
        
    def get_random_img_from_dataset(self, dataset_path, scenarios, rounds, max_img_nr):
        img = None
        while img is None:
            scenario = random.choice(scenarios)
            round = random.randint(1, rounds)
            img_nr = random.randint(0, max_img_nr)
            img, img_path = self.get_img_from_dataset(dataset_path, scenario, round, img_nr, False)
        
        return img, img_path, [scenario, round, img_nr]

    def check_boundary_classification(self, g, l):
        gradient_threshold = 8000
        angle_threshold = np.deg2rad(50)
        min_segment_length = 200
        projection = np.dot(g, self.GREEN)
        proj_angle = np.arccos(projection/(np.linalg.norm(g)*np.linalg.norm(self.GREEN)))
        is_field_boundary = (projection>gradient_threshold and \
                            np.abs(proj_angle)<angle_threshold and \
                            l>min_segment_length)
        return is_field_boundary

    def check_marking_classification(self, g, l):
        gradient_threshold = 8000
        angle_threshold = np.deg2rad(30)
        min_segment_length = 50
        projection = np.dot(g, self.GREEN-self.WHITE)
        proj_angle = np.arccos(projection/(np.linalg.norm(g)*np.linalg.norm(self.GREEN-self.WHITE)))
        if proj_angle>np.pi/2: proj_angle = np.pi-proj_angle
        is_field_marking = (np.abs(projection)>gradient_threshold and \
                            np.abs(proj_angle)<angle_threshold and 
                            l>min_segment_length)
        return is_field_marking
