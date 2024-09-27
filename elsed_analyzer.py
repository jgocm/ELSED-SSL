import pyelsed
import numpy as np
import cv2
from scipy.signal import convolve2d
import random
import os

class SegmentsAnalyzer():
    boundary_thresholds = [27.78332309, 51.21788891, 95.33978007]
    marking_thresholds  = [32.17114637, 29.69457975, 104.8096455]

    def __init__(self,
                 segments_detector = pyelsed,
                 boundary_thresholds = boundary_thresholds,
                 marking_thresholds = marking_thresholds):
        
        self.segments_detector = segments_detector
            
        self.WHITE = np.array([255, 255, 255])
        self.BLUE =  np.array([255, 0, 0])
        self.GREEN = np.array([0, 255, 0])
        self.RED =   np.array([0, 0, 255])
        self.BLACK = np.array([0, 0, 0])

        self.boundary_thresholds = boundary_thresholds
        self.marking_thresholds = marking_thresholds
        #print(self.boundary_thresholds, self.marking_thresholds)

    def detect(self, img, sigma = 1, gradientThreshold = 30, minLineLen = 15):
        boundary_grad_th, boundary_angle_threshold_deg, boundary_min_seg_len = self.boundary_thresholds
        markings_grad_th, markings_angle_threshold_deg, markings_min_seg_len = self.marking_thresholds        
        return self.segments_detector.detect(img,
                                             sigma = sigma,
                                             gradientThreshold = gradientThreshold,
                                             minLineLen = minLineLen,
                                             boundaryGradTh = boundary_grad_th, 
                                             boundaryAngleTh = boundary_angle_threshold_deg, 
                                             boundaryMinLength = boundary_min_seg_len, 
                                             markingGradTh = markings_grad_th, 
                                             markingAngleTh = markings_angle_threshold_deg, 
                                             markingMinLength = markings_min_seg_len)

    def classify(self, grad_x, grad_y, segment_length):
        boundary_grad_th, boundary_angle_threshold_deg, boundary_min_seg_len = self.boundary_thresholds
        markings_grad_th, markings_angle_threshold_deg, markings_min_seg_len = self.marking_thresholds
        return self.segments_detector.classify(grad_x,
                                               grad_y,
                                               segment_length,
                                               boundaryGradTh = boundary_grad_th, 
                                               boundaryAngleTh = boundary_angle_threshold_deg, 
                                               boundaryMinLength = boundary_min_seg_len, 
                                               markingGradTh = markings_grad_th, 
                                               markingAngleTh = markings_angle_threshold_deg, 
                                               markingMinLength = markings_min_seg_len)


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

    def is_pixel_window_valid(self, pixel_x, pixel_y, img_height, img_width, window_size=3):
        x_is_valid = (pixel_x-(window_size-1)/2 >= 0) and (pixel_x+(window_size+1)/2 <= img_width)
        y_is_valid = (pixel_y-(window_size-1)/2 >= 0) and (pixel_y+(window_size+1)/2 <= img_height)
        return x_is_valid and y_is_valid

    def get_gradients_from_line_points(self, src, line_points):
        operator_x = 1/4 * np.array([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]])
        operator_y = 1/4 * np.array([[1, 2, 1],
                                     [0, 0, 0],
                                     [-1, -2, -1]])
        
        # Initialize lists to store pixel values for the 3x3 windows
        B_pixels = []
        G_pixels = []
        R_pixels = []

        # Iterate over the desired pixels and collect the 3x3 window pixel values
        for pixel in line_points[2:-2]:
            x, y = pixel
            window = src[y-1:y+2, x-1:x+2]
            # sometimes the position on the image might not be able to compute this window
            # it is easier to check the result of the window than use the is_pixel_window_valid()
            if window.shape[0]==0:
                continue
            B, G, R = window[:,:,0], window[:,:,1], window[:,:,2]
            B_pixels.append(B)
            G_pixels.append(G)
            R_pixels.append(R)

        # Compute the mean of the collected pixel values
        mean_B = np.mean(B_pixels, axis=0)
        mean_G = np.mean(G_pixels, axis=0)
        mean_R = np.mean(R_pixels, axis=0)
        
        # Apply the convolution to the mean windows
        convolved_Bx = convolve2d(mean_B, operator_x, mode='valid')[0,0]
        convolved_Gx = convolve2d(mean_G, operator_x, mode='valid')[0,0]
        convolved_Rx = convolve2d(mean_R, operator_x, mode='valid')[0,0]
        convolved_By = convolve2d(mean_B, operator_y, mode='valid')[0,0]
        convolved_Gy = convolve2d(mean_G, operator_y, mode='valid')[0,0]
        convolved_Ry = convolve2d(mean_R, operator_y, mode='valid')[0,0]
        
        # Create gradient arrays
        g_BGRx = np.array([convolved_Bx, convolved_Gx, convolved_Rx])     
        g_BGRy = np.array([convolved_By, convolved_Gy, convolved_Ry])     
        
        return g_BGRx, g_BGRy

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

    def get_grad_similarity(self, grad, color1, color2):
        color_transition = color2-color1
        projection = np.dot(grad, color_transition)
        proj_angle_rad = np.arccos(projection/(np.linalg.norm(grad)*np.linalg.norm(projection)))
        return projection, proj_angle_rad

    def check_one_side_line_classification(self, grad, seg_length, grad_threshold, angle_threshold_deg, min_seg_length, color1, color2):
        projection, proj_angle_rad = self.get_grad_similarity(grad, color1, color2)
        angle_threshold_rad = np.deg2rad(angle_threshold_deg)
        is_field_boundary = (projection>grad_threshold and \
                            np.abs(proj_angle_rad)<angle_threshold_rad and \
                            seg_length>min_seg_length)
        return is_field_boundary

    def check_two_sides_line_classification(self, grad, seg_length, grad_threshold, angle_threshold_deg, min_seg_length, color1, color2):
        projection, proj_angle_rad = self.get_grad_similarity(grad, color1, color2)
        angle_threshold_rad = np.deg2rad(angle_threshold_deg)
        if proj_angle>np.pi/2: proj_angle = np.pi-proj_angle
        is_field_marking = (np.abs(projection)>grad_threshold and \
                            np.abs(proj_angle)<angle_threshold_rad and 
                            seg_length>min_seg_length)
        return is_field_marking

    def check_boundary_classification(self, g, l, gradient_threshold=8000, angle_threshold_deg=50, min_segment_length=200):
        angle_threshold = np.deg2rad(angle_threshold_deg)
        projection = np.dot(g, self.GREEN)/np.linalg.norm(self.GREEN)
        proj_angle = np.arccos(projection/np.linalg.norm(g))
        is_field_boundary = (projection>gradient_threshold and \
                            np.abs(proj_angle)<angle_threshold and \
                            l>min_segment_length)
        return is_field_boundary

    def check_marking_classification(self, g, l, gradient_threshold=8000, angle_threshold_deg=30, min_segment_length=50):
        angle_threshold = np.deg2rad(angle_threshold_deg)
        projection = np.dot(g, self.GREEN-self.WHITE)/np.linalg.norm(self.GREEN-self.WHITE)
        proj_angle = np.arccos(projection/np.linalg.norm(g))
        if proj_angle>np.pi/2: proj_angle = np.pi-proj_angle
        is_field_marking = (np.abs(projection)>gradient_threshold and \
                            np.abs(proj_angle)<angle_threshold and 
                            l>min_segment_length)
        return is_field_marking

def test_on_random_image_from_dataset():
    dataset_path = '/home/joao-dt/ssl-navigation-dataset'
    scenarios = ['rnd', 'sqr', 'igs']
    rounds = 3
    max_img_nr = 2000

    boundary_thresholds_path = 'annotations/optimal_boundary_thresholds.npy'
    marking_thresholds_path = 'annotations/optimal_marking_thresholds.npy'
    
    boundary_thresholds = np.load(boundary_thresholds_path)
    #print(boundary_thresholds)
    marking_thresholds = np.load(marking_thresholds_path)
    #print(marking_thresholds)

    boundary_grad_th, boundary_angle_threshold_deg, boundary_min_seg_len = boundary_thresholds
    marking_grad_th,  marking_angle_threshold_deg,  marking_min_seg_len  = marking_thresholds

    analyzer = SegmentsAnalyzer(pyelsed)

    while True:
        original_img, img_path, img_details = analyzer.get_random_img_from_dataset(dataset_path, scenarios, rounds, max_img_nr)
        gs_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        gs_img = cv2.cvtColor(gs_img, cv2.COLOR_GRAY2BGR)
        dbg_img = original_img.copy()
        print(f"Img: {img_path}")
    
        segments, scores, labels, grads_x, grads_y = analyzer.detect(original_img)

        for s, score, label, grad_x, grad_y in zip(segments.astype(np.int32), scores, labels, grads_x, grads_y):
            line_points = analyzer.get_bresenham_line_points(s)

            label_test = analyzer.classify(grad_x, -grad_y, score)
            
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

        cv2.imshow('elsed', gs_img)
        cv2.imshow('elsed-ssl', dbg_img)
        
        key = cv2.waitKey(0) & 0xFF

        if key==ord('q'):
            quit()

def test_with_annotations():
    import pandas as pd

    annotations_path = 'annotations/segments_annotations.csv'
    boundary_thresholds_path = 'annotations/optimal_boundary_thresholds.npy'
    marking_thresholds_path = 'annotations/optimal_marking_thresholds.npy'
    df = pd.read_csv(annotations_path)
    
    boundary_thresholds = np.load(boundary_thresholds_path)
    marking_thresholds = np.load(marking_thresholds_path)
    print(boundary_thresholds)
    print(marking_thresholds)

    boundary_grad_th, boundary_angle_threshold_deg, boundary_min_seg_len = boundary_thresholds
    marking_grad_th,  marking_angle_threshold_deg,  marking_min_seg_len  = marking_thresholds

    analyzer = SegmentsAnalyzer(pyelsed, boundary_thresholds, marking_thresholds)

    for index, row in df.iterrows():
        img_path = row['img_path']
        original_img = cv2.imread(img_path)
        gs_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        gs_img = cv2.cvtColor(gs_img, cv2.COLOR_GRAY2BGR)
        dbg_img = original_img.copy()
        print(f"Img: {img_path}")
    

        segment = np.array([row['x0'], row['y0'], row['x1'], row['y1']], dtype=np.int32)
        is_field_boundary_gt = row['is_field_boundary']
        is_field_marking_gt = row['is_field_marking']

        grad_x = np.array([row['grad_Bx'],row['grad_Gx'],row['grad_Rx']], dtype=np.float32)
        grad_y = np.array([row['grad_By'],row['grad_Gy'],row['grad_Ry']], dtype=np.float32)
        segment_length = row['segment_length']

        line_points = analyzer.get_bresenham_line_points(segment)
        
        # this approach is not working, probably there is some issue with the annotations
        label = analyzer.classify(grad_x, -grad_y, segment_length)

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

        cv2.imshow('elsed', gs_img)
        cv2.imshow('elsed-ssl', dbg_img)
        
        key = cv2.waitKey(0) & 0xFF

        if key==ord('q'):
            quit()

if __name__ ==  "__main__":
    #test_on_random_image_from_dataset()

    test_with_annotations()