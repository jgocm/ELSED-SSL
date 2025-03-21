import pyelsed
import numpy as np
import cv2
from elsed_analyzer import SegmentsAnalyzer
import utils

if __name__ == "__main__":
    config = utils.load_config_file("configs.json")
    dataset_label = config["dataset_label"]

    paths = utils.load_paths_from_config_file("configs.json")
    boundary_thresholds_path = paths['boundary_thresholds']
    marking_thresholds_path = paths['marking_thresholds']


    boundary_thresholds = np.load(boundary_thresholds_path)
    marking_thresholds = np.load(marking_thresholds_path)
    print(boundary_thresholds)
    print(marking_thresholds)

    analyzer = SegmentsAnalyzer(pyelsed,
                                boundary_thresholds,
                                marking_thresholds,
                                True)

    # Load video and open capture
    input_video_path = f"data/videos/{dataset_label}.mp4"
    output_video_path = f"data/videos/{dataset_label}_processed.mp4"
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video

    # Initialize video writer
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no frames are left

        # Run detection on the frame
        result = analyzer.detect(frame)

        # Write the processed frame to the output video
        out.write(frame)

        cv2.imshow('frame', frame)
        cv2.waitKey(int(1000/fps))

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Processing complete. Video saved at:", output_video_path)

