import pyelsed
import numpy as np
import cv2
from scipy.signal import convolve2d
import random
import os
import time
from elsed_analyzer import *


WHITE = np.array([255, 255, 255])
BLUE =  np.array([255, 0, 0])
GREEN = np.array([0, 255, 0])
RED =   np.array([0, 0, 255])
BLACK = np.array([0, 0, 0])

def get_line_parameters_from_endpoints(p1, p2):
    length = np.linalg.norm(p2-p1)
    r = (p2-p1)/length
    p = p1
    return r, p, length

def get_line_points(s):
    p1 = np.array([s[0], s[1]])
    p2 = np.array([s[2], s[3]])
    r, p, segment_length = get_line_parameters_from_endpoints(p1, p2)
    
    lambdas = np.arange(round(segment_length))
    lambdas_column = lambdas[:, np.newaxis]
    line_points = (p + r*lambdas_column).astype(np.int32)
    return line_points

def get_bresenham_line_points(s):
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

def get_gradients_from_line_points(src, line_points):
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

def get_img_from_dataset(dataset_path='/home/rc-blackout/ssl-navigation-dataset', scenario='rnd', round=2, img_nr=100, gray_scale=False):
    img_path = f'{dataset_path}/data/{scenario}_0{round}/cam/{img_nr}.png'
    if os.path.isfile(img_path):
        if gray_scale:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(img_path)
    else:
        print(f'Img {img_path} not available')
        img = None
        
    return img, img_path
    
def get_random_img_from_dataset(dataset_path, scenarios, rounds, max_img_nr):
    img = None
    while img is None:
        scenario = random.choice(scenarios)
        round = random.randint(1, rounds)
        img_nr = random.randint(0, max_img_nr)
        img, img_path = get_img_from_dataset(dataset_path, scenario, round, img_nr, False)
    
    return img, img_path, [scenario, round, img_nr]

def check_boundary_classification(g, l):
    gradient_threshold = 6288.33
    angle_threshold = np.deg2rad(52)
    min_segment_length = 84
    projection = np.dot(g, GREEN)
    proj_angle = np.arccos(projection/(np.linalg.norm(g)*np.linalg.norm(GREEN)))
    is_field_boundary = (projection>gradient_threshold and \
                         np.abs(proj_angle)<angle_threshold and \
                         l>min_segment_length)
    return is_field_boundary

def check_marking_classification(g, l):
    gradient_threshold = 7575.37
    angle_threshold = np.deg2rad(32)
    min_segment_length = 57
    projection = np.dot(g, GREEN-WHITE)
    proj_angle = np.arccos(projection/(np.linalg.norm(g)*np.linalg.norm(GREEN-WHITE)))
    if proj_angle>np.pi/2: proj_angle = np.pi-proj_angle
    is_field_marking = (np.abs(projection)>gradient_threshold and \
                        np.abs(proj_angle)<angle_threshold and 
                        l>min_segment_length)
    return is_field_marking

if __name__ == "__main__":
    # TODO: refactor this script to use analyzer class
    dataset_path = '/home/joao-dt/ssl-navigation-dataset'
    scenarios = ['rnd', 'sqr', 'igs']
    rounds = 3
    max_img_nr = 2000
    while True:
        original_img, img_path, img_details = get_random_img_from_dataset(dataset_path, scenarios, rounds, max_img_nr)
        #original_img, img_path = get_img_from_dataset(dataset_path, 'rnd', 2, 1566)
        print(f"Img: {img_path}")
        gs_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        #dbg_img = cv2.cvtColor(gs_img, cv2.COLOR_GRAY2RGB)
        dbg_img = original_img.copy()
        
        t0 = time.time()
        segments, scores, labels, grads_x, grads_y = pyelsed.detect(original_img, 1, 30, 150)
        print(f'elapsed_time: {time.time() - t0}')
        gs_img = cv2.cvtColor(gs_img, cv2.COLOR_GRAY2BGR)
        
        for s, label, grad_x, grad_y in zip(segments.astype(np.int32), labels, grads_x, grads_y):
            line_points = get_bresenham_line_points(s)
            
            #gx, gy = get_gradients_from_line_points(original_img, line_points)
            
            #if np.linalg.norm(gy)>np.linalg.norm(gx): g = gy
            #else: g = gx

            #is_field_boundary = check_boundary_classification(g, len(line_points))
            #is_field_marking = check_marking_classification(g, len(line_points))
            
            #print(f'grad_x: {grad_x}, grad_y: {-grad_y}')

            is_field_boundary = (label==1)
            is_field_marking = (label==2)

            for p in line_points:
                x, y = p
                gs_img[y, x] = RED
                
                if is_field_marking:
                    dbg_img[y, x] = RED
                elif is_field_boundary:
                    dbg_img[y, x] = GREEN                    
                else:
                    dbg_img[y, x] = BLACK
        
        cv2.imshow('elsed', gs_img)        
        cv2.imshow('elsed-ssl', dbg_img)
        #cv2.imshow('original', original_img)
        key = cv2.waitKey(0) & 0xFF

        if key==ord('q'):
            break
        elif key==ord('s'):
            scenario, round, img_nr = img_details
            cv2.imwrite(f'experiments/original/{scenario}_0{round}_{img_nr}_original.png', original_img)
            cv2.imwrite(f'experiments/inference/{scenario}_0{round}_{img_nr}_inference.png', dbg_img)

