import cv2
import os
from tqdm import tqdm


folder_names = [
    "C1_328_109",
    "C1_352_121",
    "C1_352_131",
    "C1_352_132",
    "C1_352_137",
    "C1_352_312",
    "C1_364_105",
]

data_folder = ["Data1of3", "Data2of3", "Data3of3"]
data_folder_names = data_folder[2]
BLAST_NAME = folder_names[5]

video_name = BLAST_NAME.split("_", 1)[1]


# Input video file
video_path = f"{data_folder_names}/{BLAST_NAME}/{video_name}.mp4"
output_folder = f"demo/{BLAST_NAME}_output/{BLAST_NAME}_frames"
os.makedirs(output_folder, exist_ok=True)

resize_width, resize_height = 1920, 1080  # Target resolution (e.g., 1080p)
frames_per_second = 10  # Number of frames to extract per second

# Open the video file
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # Original frame rate of the video
frame_interval = int(fps / frames_per_second)  # Interval to capture frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_count = 0
save_count = 0


with tqdm(total=total_frames, desc="Processing Frames") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame at the specified interval
        if frame_count % frame_interval == 0:
            resized_frame = cv2.resize(frame, (resize_width, resize_height))
            save_path = os.path.join(
                output_folder, f"{BLAST_NAME}_frame_{save_count:04d}.jpg"
            )
            cv2.imwrite(save_path, resized_frame)
            save_count += 1

        frame_count += 1
        pbar.update(1)  # Update progress bar

cap.release()
print(f"Extracted {save_count} resized frames to {output_folder}")
