import os
import cv2
import pandas as pd
import math


# Global variable to maintain the current angle offset
current_angle_offset = 0  # Initialize to 0 degrees


def find_non_overlapping_position(
    drill_x,
    drill_y,
    existing_labels,
    text_width,
    text_height,
    initial_radius=50,
    angle_step=15,
    max_attempts=100,
):
    """
    Find a non-overlapping position for the label text.
    Expands outward in a spiral pattern with a unique starting angle.
    """
    global current_angle_offset
    radius = initial_radius
    angle = current_angle_offset  # Start with the global angle offset

    for attempt in range(max_attempts):
        # Calculate offsets using polar coordinates
        offset_x = int(radius * math.cos(math.radians(angle)))
        offset_y = int(radius * math.sin(math.radians(angle)))
        label_x, label_y = drill_x + offset_x, drill_y + offset_y

        # Check for overlap
        collision = False
        for ex_label_x, ex_label_y, ex_width, ex_height in existing_labels:
            if (
                label_x < ex_label_x + ex_width + 5  # Add padding
                and label_x + text_width + 5 > ex_label_x
                and label_y < ex_label_y + ex_height + 5
                and label_y + text_height + 5 > ex_label_y
            ):
                collision = True
                break

        if not collision:
            # Update the global angle offset for the next label
            current_angle_offset = (current_angle_offset + angle_step) % 360
            return label_x, label_y

        # Update angle and radius for the next attempt
        angle += angle_step
        if angle >= 360:  # Completed a full rotation
            angle -= 360
            radius += 10  # Increment radius after full rotation

    # If no non-overlapping position is found, return the original position offset
    return drill_x + initial_radius, drill_y + initial_radius


# Configuration
folder_names = [
    "C1_328_109",
    "C1_352_121",
    "C1_352_131",
    "C1_352_132",
    "C1_352_137",
    "C1_352_312",
    "C1_364_105",
]
BLAST_NAME = folder_names[6]

FINAL_CSV_PATH = f"dev/demo/{BLAST_NAME}_output/{BLAST_NAME}_Final_Result.csv"
FRAMES_FOLDER = f"dev/demo/{BLAST_NAME}_output/{BLAST_NAME}_frames"
OUTPUT_LABELING_FOLDER = f"dev/demo/{BLAST_NAME}_output/for_labeling"

# Ensure the output folder exists
os.makedirs(OUTPUT_LABELING_FOLDER, exist_ok=True)

# Load the final CSV
final_df = pd.read_csv(FINAL_CSV_PATH)
existing_labels = []  # Maintain a global list of all label positions and sizes

for idx, row in final_df.iterrows():
    if pd.isna(row["start_time"]) or pd.isna(row["end_time"]):
        continue

    end_frame_name = row["end_frame_name"]
    if pd.isna(end_frame_name):
        continue

    frame_path = os.path.join(FRAMES_FOLDER, end_frame_name)
    output_path = os.path.join(OUTPUT_LABELING_FOLDER, end_frame_name)

    # Load the original or already labeled image
    if os.path.exists(output_path):
        image = cv2.imread(output_path)
    else:
        if not os.path.exists(frame_path):
            print(f"Frame not found: {frame_path}")
            continue
        image = cv2.imread(frame_path)
        if image is None:
            print(f"Failed to load image: {frame_path}")
            continue

    # Draw the point with red color and Hole id
    drill_x, drill_y = int(row["Transformed_Pixel_X"]), int(row["Transformed_Pixel_Y"])
    hole_id = row["Hole.id"]

    # Calculate text size and bounding box
    text = str(hole_id)
    text_size = cv2.getTextSize(
        text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=2
    )[0]
    text_width, text_height = text_size[0] + 10, text_size[1] + 10

    # Find a non-overlapping position for the label
    label_x, label_y = find_non_overlapping_position(
        drill_x, drill_y, existing_labels, text_width, text_height
    )

    # Add the new label's bounding box to the list
    existing_labels.append((label_x, label_y - text_height, text_width, text_height))

    # Draw a red circle at the drill hole position
    cv2.circle(image, (drill_x, drill_y), radius=5, color=(0, 0, 255), thickness=-1)

    # Draw a line connecting the label text to the drill hole
    cv2.line(
        image, (drill_x, drill_y), (label_x, label_y), color=(0, 0, 255), thickness=1
    )

    # Add a semi-transparent rectangle background for the label
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (label_x, label_y - text_size[1] - 5),
        (label_x + text_size[0] + 10, label_y + 5),
        color=(255, 255, 255),
        thickness=-1,
    )
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Draw the text label
    cv2.putText(
        image,
        text,
        (label_x + 5, label_y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(0, 0, 255),
        thickness=2,
    )

    # Save the updated image
    cv2.imwrite(output_path, image)

print(f"Processing complete. Labeled images saved in '{OUTPUT_LABELING_FOLDER}'.")
