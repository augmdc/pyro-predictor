import cv2
import os
from tqdm import tqdm
import pandas as pd
import streamlit as st

# Directory to store uploaded files
UPLOAD_FOLDER = "demo/App/Uploads"


def process_video(blast_name):
    """
    Processes the video associated with the given blast name to extract frames at a specific interval and resolution.
    """
    # Define video path and output folder based on blast name
    uploaded_video_path = os.path.join(UPLOAD_FOLDER, f"{blast_name}.mp4")
    output_folder = f"{UPLOAD_FOLDER}/{blast_name}_output/{blast_name}_frames"
    os.makedirs(output_folder, exist_ok=True)

    # Parameters for frame extraction
    resize_width, resize_height = 1920, 1080  # Target resolution
    frames_per_second = 10  # Frames per second to extract

    # Open the video file
    cap = cv2.VideoCapture(uploaded_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {uploaded_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # Original frame rate
    frame_interval = int(fps / frames_per_second)  # Interval for capturing frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    save_count = 0

    print(f"Processing video: {uploaded_video_path}")
    print(f"Output frames will be saved to: {output_folder}")

    # Initialize Streamlit progress bar
    progress_bar = st.progress(0)
    progress_text = st.text("Processing frames...")

    with tqdm(total=total_frames, desc="Processing Frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame at the specified interval
            if frame_count % frame_interval == 0:
                resized_frame = cv2.resize(frame, (resize_width, resize_height))
                save_path = os.path.join(
                    output_folder, f"{blast_name}_frame_{save_count:04d}.jpg"
                )
                cv2.imwrite(save_path, resized_frame)
                save_count += 1

            frame_count += 1
            pbar.update(1)
            progress_bar.progress(frame_count / total_frames)
            progress_text.text(
                f"Processing frame {frame_count} of {total_frames} ({frame_count / total_frames * 100:.2f}%)"
            )
    # Finalizing progress
    progress_bar.progress(1.0)
    
    cap.release()
    print(f"Extracted {save_count} resized frames to {output_folder}")
