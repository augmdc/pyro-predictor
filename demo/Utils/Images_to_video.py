import os
import cv2

folder_names = [
    "C1_328_109",
    "C1_352_121",
    "C1_352_131",
    "C1_352_132",
    "C1_352_137",
    "C1_352_312",
    "C1_364_105",
]

folder_index = 5
BLAST_NAME = folder_names[folder_index]


def images_to_video(folder_path, output_video_path, frame_rate=30):
    """
    Combines images from a folder into an MP4 video.

    Parameters:
    - folder_path: Path to the folder containing frame images.
    - output_video_path: Path to save the output MP4 video.
    - frame_rate: Frame rate for the video.
    """
    # Get a sorted list of all image files in the folder
    images = sorted(
        [
            img
            for img in os.listdir(folder_path)
            if img.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    if not images:
        print("No images found in the specified folder.")
        return

    # Read the first image to get video dimensions
    first_image_path = os.path.join(folder_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 'mp4v' for MP4 format
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Write each image to the video
    for image_name in images:
        image_path = os.path.join(folder_path, image_name)
        frame = cv2.imread(image_path)
        out.write(frame)

    # Release the video writer
    out.release()
    print(f"Video successfully created at {output_video_path}")


if __name__ == "__main__":

    output_folder_names = ["frames_combined", "frames_wave"]
    output_video_names = ["video_cv_output.mp4", "video_cv_output_wave.mp4"]

    for i in range(len(output_folder_names)):
        folder_path = f"demo/{BLAST_NAME}_output/{output_folder_names[i]}"
        output_video_path = f"demo/{BLAST_NAME}_output/{output_video_names[i]}"
        frame_rate = 10

        images_to_video(folder_path, output_video_path, frame_rate)
