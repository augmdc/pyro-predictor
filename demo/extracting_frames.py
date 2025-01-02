import cv2
import os

# Parameters
BLAST_NAME = "C1_352_121"

# Input video file
video_path = 'Demo1/demo1_video.mp4'
output_folder = f'dev/demo/{BLAST_NAME}_output/{BLAST_NAME}_frames'
os.makedirs(output_folder, exist_ok=True)

resize_width, resize_height = 1920, 1080  # Target resolution (e.g., 1080p)
frames_per_second = 10  # Number of frames to extract per second

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
        save_path = os.path.join(output_folder, f'{BLAST_NAME}_frame_{save_count:04d}.jpg')
        cv2.imwrite(save_path, resized_frame)
        save_count += 1
    
    frame_count += 1

cap.release()
print(f"Extracted {save_count} resized frames to {output_folder}")
