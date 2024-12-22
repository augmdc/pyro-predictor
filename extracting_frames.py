import cv2
import os

# Parameters
video_path = 'Dataset/Data2of3/Data2.mp4'
output_folder = 'extracted_frames2'
os.makedirs(output_folder, exist_ok=True)

resize_width, resize_height = 1920, 1080  # Target resolution (e.g., 1080p)
frames_per_second = 3  # Number of frames to extract per second

# Open the video file
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # Original frame rate of the video
frame_interval = int(fps / frames_per_second)  # Interval to capture frames

frame_count = 0
save_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save frame at the specified interval
    if frame_count % frame_interval == 0:
        resized_frame = cv2.resize(frame, (resize_width, resize_height))
        save_path = os.path.join(output_folder, f'dataset2_frame_{save_count:04d}.jpg')
        cv2.imwrite(save_path, resized_frame)
        save_count += 1
    
    frame_count += 1

cap.release()
print(f"Extracted {save_count} resized frames to {output_folder}")
