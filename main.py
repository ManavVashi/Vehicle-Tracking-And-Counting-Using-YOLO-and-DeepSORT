import os
import cv2
from ultralytics import YOLO
from tracker import Tracker
import numpy as np

# Path to the video file
video_path = os.path.join('.', 'demo_1.mp4')

# Load the pre-trained YOLO model for object detection
model = YOLO('.', 'best.pt')

# Initialize video capture object
vidObj = cv2.VideoCapture(video_path)

# Set up the output video file
video_out_path = os.path.join('.', 'output_demo_1_1.mp4')
fps = int(vidObj.get(cv2.CAP_PROP_FPS))  # Frames per second
width = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
height = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

# Initialize the tracker for tracking detected objects
tracker = Tracker()

# Detection threshold to filter out low-confidence detections
detection_threshold = 0.5

# Dictionary to store the previous positions of tracked vehicles
previous_positions = {}

# Define the class names for the detected vehicles (COCO dataset)
vehicle_classes = ['car', 'bus', 'bicycle', 'ambassador', 'van', 'two_wheeler', 
                   'rickshaw_oldGen', 'wan', 'truck', 'rickshaw_newGen', 
                   'toto', 'tempo']

# Define specific colors for bounding boxes based on vehicle class
class_colors = {
    'car': (255, 0, 0),  # Red
    'bus': (0, 255, 0),  # Green
    'bicycle': (0, 0, 255),  # Blue
    'ambassador': (255, 255, 0),  # Cyan
    'van': (255, 0, 255),  # Magenta
    'two_wheeler': (255, 20, 147),  # DeepPink
    'rickshaw_oldGen': (210, 105, 30),  # Chocolate
    'wan': (102, 51, 153),  # RebeccaPurple
    'truck': (255, 69, 0),  # OrangeRed
    'rickshaw_newGen': (165, 42, 42),  # Brown
    'toto': (47, 79, 79),  # DarkSlateGray
    'tempo': (139, 0, 139),  # DarkMagenta
}

# Initialize counters for each class, tracking vehicles moving towards or away from the camera
class_counts_towards = {class_name: 0 for class_name in vehicle_classes}
class_counts_away = {class_name: 0 for class_name in vehicle_classes}

# Define the line coordinates and thickness for detecting vehicle direction
start_point = (-10, 400)
end_point = (1500, 400)
line_y = start_point[1]  # y-coordinate of the line
thickness = 3

# Read the first frame of the video
success, frame = vidObj.read()

# Process video frames until the end
while success:
    # Pass the frame to the YOLO model to get detections
    results = model(frame)

    # Get the center x-coordinate of the frame to determine vehicle direction
    frame_center_x = width // 2

    # Iterate over detection results
    for result in results:
        detections = []
        
        # Extract bounding box, score, and class ID for each detection
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if score > detection_threshold:  # Only consider detections above the threshold
                detections.append([int(x1), int(y1), int(x2), int(y2), score])

        # Update the tracker with the new detections
        tracker.update(frame, detections)

        # Process each tracked object
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            center_y = int((y1 + y2) / 2)
            center_x = int((x1 + x2) / 2)

            # Retrieve the class name and detection confidence
            class_name = vehicle_classes[int(class_id)]
            label = f'{class_name}: {score:.2f}'

            # Determine the color for the bounding box based on the class
            color = class_colors.get(class_name, (255, 255, 255))  # Default to white if class not found

            # Draw the bounding box with the determined color
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

            # Display the class label and confidence on the bounding box
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Check if the vehicle has crossed the line in either direction
            if track_id in previous_positions:
                prev_center_y = previous_positions[track_id]
                if prev_center_y < line_y <= center_y:  # Vehicle moved from top to bottom
                    if center_x < frame_center_x:
                        class_counts_towards[class_name] += 1  # Count vehicle towards the camera
                        print(f"{class_name} towards count: {class_counts_towards[class_name]}")
                    else:
                        class_counts_away[class_name] += 1  # Count vehicle away from the camera
                        print(f"{class_name} away count: {class_counts_away[class_name]}")

            # Update the previous position of the tracked vehicle
            previous_positions[track_id] = center_y

    # Draw the detection line on the frame
    frame = cv2.line(frame, start_point, end_point, (0, 0, 0), thickness)

    # Display the count of each class for vehicles moving towards the camera
    y_offset_towards = 30
    for class_name, count in class_counts_towards.items():
        cv2.putText(frame, f'{class_name} towards: {count}', (10, y_offset_towards), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_offset_towards += 30
    
    # Display the count of each class for vehicles moving away from the camera
    y_offset_away = 30
    for class_name, count in class_counts_away.items():
        cv2.putText(frame, f'{class_name} away: {count}', (width - 300, y_offset_away), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_offset_away += 30

    # Write the processed frame to the output video
    out.write(frame)

    # Read the next frame from the video
    success, frame = vidObj.read()

# Release the video capture and output resources
vidObj.release()
out.release()
cv2.destroyAllWindows()
