import numpy as np
import time
import cv2

img = cv2.imread('base.jpg')
half_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
cv2.imshow('base', half_img)
cv2.waitKey(0)

# === 1. Define known square dimensions in world coordinates ===
# Let's assume a square of size 1.0 units (e.g., 1 meter x 1 meter)
square_size = 1.0
object_points = np.array([
    [0, 0, 0],                          # Top-left
    [square_size, 0, 0],                # Top-right
    [square_size, square_size, 0],      # Bottom-right
    [0, square_size, 0]                 # Bottom-left
], dtype=np.float32)

# === 2. Define corresponding image coordinates (manually or from detection) ===
# Example values in pixels (replace with your actual coordinates)
image_points = 2*np.array([
    [226, 201],  # top-left
    [489, 35],  # top-right
    [666, 300],  # bottom-right
    [384, 477]   # bottom-left
], dtype=np.float32)
print(f'Image Points (u, v):\n{image_points}\n')

# === 3. Define the camera intrinsic matrix K ===
# Example intrinsics — replace with your calibrated values
fx = 627.27  # focal length in x
fy = 626.51  # focal length in y
cx = 971.74  # optical center x
cy = 544.96  # optical center y

K = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
], dtype=np.float32)

# === 4. Compute homography H from world to image plane (Z=0 plane) ===
t0 = time.time()
H, _ = cv2.findHomography(object_points[:, :2], image_points)

# === 5. Decompose H into rotation and translation using K ===
H_normalized = np.linalg.inv(K) @ H

r1 = H_normalized[:, 0]
r2 = H_normalized[:, 1]
t = H_normalized[:, 2]

# Normalize rotation vectors
norm = np.linalg.norm(r1)
r1 = r1 / norm
r2 = r2 / norm
t = t / norm
r3 = np.cross(r1, r2)

# Construct rotation matrix
R = np.stack([r1, r2, r3], axis=1)

# Fix orthonormality using SVD
U, _, Vt = np.linalg.svd(R)
R = U @ Vt

# === 6. Build projection matrix P = K * [R | t] ===
Rt = np.hstack((R, t.reshape(3, 1)))
P = K @ Rt

# === 7. Decompose projection matrix to get Euler angles ===
_, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(P)
elapsed_time = time.time() - t0

# === 8. Output ===
print("Rotation Matrix (R):")
print(R)

print("\nTranslation Vector (t):")
print(t)

print("\nEuler Angles from cv2.decomposeProjectionMatrix (degrees):")
print(f"Roll (X):  {euler_angles[0][0]:.2f}°")
print(f"Pitch (Y): {euler_angles[1][0]:.2f}°")
print(f"Yaw (Z):   {euler_angles[2][0]:.2f}°")

print(f'elapsed_time: {1000*elapsed_time:.5f} ms')