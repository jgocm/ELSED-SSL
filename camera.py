import numpy as np
import cv2
import pandas as pd

class Camera():
    def __init__(self,
                 camera_matrix=np.identity(3),
                 camera_distortion=np.zeros((4,1)),
                 camera_initial_position=np.zeros(3),
                 vision_offset=np.zeros(3)):
        
        self.intrinsic_parameters = camera_matrix
        self.distortion_parameters = camera_distortion

        self.rotation_vector = np.zeros((3,1))
        self.rotation_matrix = np.zeros((3,3))
        self.translation_vector = np.zeros((3,1)).T

        self.position = np.zeros((3,1))
        self.rotation = np.zeros((3,1)) # EULER ROTATION ANGLES
        self.height = 0

        self.initial_position = camera_initial_position
        # apply XYW offset if calibration ground truth position is known
        self.offset = vision_offset

    def setIntrinsicParameters(self, mtx):
        if np.shape(mtx)==(3,3): 
            self.intrinsic_parameters = mtx
            print(f"Camera Matrix is:")
            print(mtx)
            return self
        else:
            print(f"Camera Matrix must be of shape (3,3) and the inserted matrix has shape {np.shape(mtx)}")
            return self

    def setDistortionParameters(self, dist):
        if np.shape(dist)==(4,1):
            self.distortion_parameters = dist
            print("Camera distortion parameters are:")
            print(dist)
            return True
        else:
            print(f"Camera distortion parameters must be of shape (4,1) and the inserted array has shape {np.shape(dist)}")
            return False
    
    def setOffset(self, offset):
        if np.shape(offset)==(3,):
            self.offset = offset
            print(f"Position offset is:")
            print(offset)
        else:
            print(f"Position offset must be of shape (3,) and the inserted matrix has shape {np.shape(offset)}")
    
    def fixPoints3d(self, points3d):
        return points3d-self.initial_position

    def computePoseFromPoints(self, points3d, points2d):
        """
        Compute camera pose to object from 2D-3D points correspondences

        Solves PnP problem using OpenCV solvePnP() method assigning
        camera pose from the corresponding 2D-3D matched points.

        Parameters
        ------------
        points3d: 3D coordinates of points

        points2d: pixel positions on image
        """
        #points3d = self.fixPoints3d(points3d=points3d)
        _,rvec,tvec=cv2.solvePnP(points3d,
                                 points2d,
                                 self.intrinsic_parameters,
                                 self.distortion_parameters)

        rmtx, jacobian=cv2.Rodrigues(rvec)
        
        pose = cv2.hconcat((rmtx,tvec))

        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose)

        camera_position = -np.linalg.inv(rmtx)@tvec
        height = camera_position[2,0]
        self.offset = (camera_position.T-self.initial_position).T

        self.rotation_vector = rvec
        self.rotation_matrix = rmtx
        self.translation_vector = tvec

        self.position = camera_position
        self.rotation = euler_angles
        self.height = height

        return camera_position, euler_angles
    
    def computeRotationMatrixFromAngles(self, euler_angles):
        theta_x = euler_angles[0][0]
        theta_x = np.deg2rad(theta_x)
        cx = np.cos(theta_x)
        sx = np.sin(theta_x)
        rX_cam = np.array([
            [1,0,0],
            [0,cx,-sx],
            [0,sx,cx]
        ])

        theta_y = euler_angles[1][0]
        theta_y = np.deg2rad(theta_y)
        cy = np.cos(theta_y)
        sy = np.sin(theta_y)
        rY_cam = np.array([
            [cy,0,sy],
            [0,1,0],
            [-sy,0,cy]
        ])

        theta_z = euler_angles[2][0]
        theta_z = np.deg2rad(theta_z)
        cz = np.cos(theta_z)
        sz = np.sin(theta_z)
        rZ_cam = np.array([
            [cz,-sz,0],
            [sz,cz,0],
            [0,0,1]
        ])

        rmtx = rZ_cam @ rY_cam @ rX_cam
        return rmtx

    def setPoseFromFile(self, camera_position, euler_angles):
        self.position = camera_position
        self.rotation = euler_angles
        self.height = camera_position[2,0]

        rmtx = self.computeRotationMatrixFromAngles(euler_angles)
        self.rotation_matrix = rmtx

        tvec = -np.matmul(np.linalg.inv(rmtx),np.matrix(camera_position))

        return tvec, rmtx
    
    def setPoseFrom3DModel(self, height, angle, center_offset=0):
        camera_position = np.array([[0], [center_offset], [height]])
        euler_angles = np.array([[angle], [0], [0]])
        self.setPoseFromFile(camera_position, euler_angles)

    def pixelToCameraCoordinates(self, x, y, z_world=0):
        uvPoint = np.array([(x,y,1)])
        mtx = self.intrinsic_parameters
        rmtx = self.rotation_matrix
        height = self.height
        
        leftsideMat = np.linalg.inv(rmtx)@(np.linalg.inv(mtx)@np.transpose(uvPoint))
        s = -(height-z_world)/leftsideMat[2]
        
        p = s*leftsideMat

        return p

    def ballAsPoint(self, left, top, right, bottom, weight_x = 0.5, weight_y=0.15):
        x = weight_x*left+(1-weight_x)*right
        y = weight_y*top+(1-weight_y)*bottom
        return x, y
    
    def ballAsPointLinearRegression(self, left, top, right, bottom, weight_x, weight_y):
        x = [left, right, top, bottom, 1]@weight_x
        y = [left, right, top, bottom, 1]@weight_y
        return x, y
    
    def robotAsPoint(self, left, top, right, bottom, weight_x = 0.5, weight_y=0.1):
        x = weight_x*left+(1-weight_x)*right
        y = weight_y*top+(1-weight_y)*bottom
        return x, y
    
    def goalAsPoint(self, left, top, right, bottom, weight_x = 0.5, weight_y=0.1):
        x = weight_x*left+(1-weight_x)*right
        y = weight_y*top+(1-weight_y)*bottom
        return x, y

    def cameraToPixelCoordinates(self, x, y, z_world=0):
        M = self.intrinsic_parameters
        R = self.rotation_matrix
        t = self.translation_vector
        height = self.height
        cameraPoint = np.array([(x,y,z_world-height)])

        rightSideMat = M@(R@(cameraPoint).T)

        s = rightSideMat[2]

        uvPoint = rightSideMat/s

        return uvPoint
    
    def cameraToRobotCoordinates(self, x, y, camera_offset=90):
        """
        Converts x, y ground position from camera axis to robot axis
        
        Parameters:
        x: x position from camera coordinates in millimeters
        y: y position from camera coordinates in millimeters
        camera_offset: camera to robot center distance in millimeters
        -----------
        Returns:
        robot_x: x position from robot coordinates in meters
        robot_y: y position from robot coordinates in meters
        robot_w: direction from x, y coordinates in radians
        """
        robot_x = (y + camera_offset)/1000
        robot_y = -x/1000
        robot_w = np.atan2(robot_y, robot_x)

        return robot_x, robot_y, robot_w

    def pixelToRobotCoordinates(self, pixel_x, pixel_y, z_world):
        # BACK PROJECT OBJECT POSITION TO CAMERA 3D COORDINATES
        object_position = self.pixelToCameraCoordinates(x=pixel_x, y=pixel_y, z_world=0)
        x, y = object_position[0], object_position[1]

        # CONVERT COORDINATES FROM CAMERA TO ROBOT AXIS
        x, y, w = self.cameraToRobotCoordinates(x[0], y[0], camera_offset=80)
        return x, y, w

    def robotToFieldCoordinates(self, point_x, point_y, robot_x, robot_y, theta_rad):
        # Rotation matrix for converting local to global coordinates
        global_x = (point_x * np.cos(theta_rad)) - (point_y * np.sin(theta_rad)) + robot_x
        global_y = (point_x * np.sin(theta_rad)) + (point_y * np.cos(theta_rad)) + robot_y

        return global_x, global_y

if __name__ == "__main__":
    from render import Render

    annotations_path = 'annotations/segments_annotations.csv'
    camera_intrinsics_path = '/home/joao-dt/ssl-navigation-dataset/configs/intrinsic_parameters.txt'
    camera_matrix = np.loadtxt(camera_intrinsics_path, dtype="float64")

    camera = Camera(camera_matrix)
    camera.setPoseFrom3DModel(height=170, angle=106.8)

    df = pd.read_csv(annotations_path)

    for index, row in df.iterrows():
        img_path = row['img_path']
        img = cv2.imread(img_path)
        s = np.array([row['x0'], row['y0'], row['x1'], row['y1']], dtype=np.int32)

        cv2.drawMarker(img, (s[0], s[1]), (0, 0, 255), 1, 10, 2)
        cv2.drawMarker(img, (s[2], s[3]), (0, 0, 255), 1, 10, 2)

        x1, y1, _ = camera.pixelToRobotCoordinates(s[0], s[1], 0)
        x2, y2, _ = camera.pixelToRobotCoordinates(s[2], s[3], 0)

        print(x1, y1)
        print(x2, y2)

        cv2.imshow('elsed segments', img)
        key = cv2.waitKey(0) & 0xFF

        if key==ord('q'):
            print(f'quiting...')
            quit()