import cv2
import os

video_label = "lines_15pm_lights_on_window_closed"

# Path to the video
video_path = f"data/videos/{video_label}.mp4"

# Directory to save frames
save_dir = f"data/{video_label}/images"
os.makedirs(save_dir, exist_ok=True)

# Open the video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Unable to open video {video_path}")
    exit()

# Get total frame count
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_count = 0

while True:
    # Set the video to the current frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    ret, frame = cap.read()

    if not ret:
        print("End of video or error reading the video.")
        break

    # Display the frame
    cv2.imshow("Video Frame", frame)
    print(f"Current frame: {frame_count}")

    key = cv2.waitKey(0)  # Wait indefinitely for a key press

    if key == ord('s'):
        # Save the current frame
        frame_filename = os.path.join(save_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        print(f"Frame saved: {frame_filename}")
    elif key == 32:  # Space bar key
        print("Skipping to the next frame.")
        frame_count = min(frame_count + 1, total_frames - 1)
    elif key == ord('a'):  # 'a' key to move back to the previous frame
        frame_count = max(frame_count - 1, 0)
        print("Moving back to the previous frame.")
    elif key == 27 or key == ord('q'):  # ESC key
        print("Exiting.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
