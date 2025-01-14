import os
import cv2
import pandas as pd
import numpy as np

import os
import cv2
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import KDTree


def generate_rating(blast_name):
    BLAST_NAME = blast_name

    # Paths
    FINAL_CSV_PATH = f"demo/{BLAST_NAME}_output/{BLAST_NAME}_Final_Result.csv"
    DRONE_IMAGE_PATH = (
        f"demo/{BLAST_NAME}_output/{BLAST_NAME}_frames/{BLAST_NAME}_frame_0000.jpg"
    )
    OUTPUT_CANVAS_PATH = (
        f"demo/{BLAST_NAME}_output/{BLAST_NAME}_rating_design_map.png"
    )
    OUTPUT_DRONE_IMAGE_PATH = (
        f"demo/{BLAST_NAME}_output/{BLAST_NAME}_rating_drone_footage.png"
    )

    # Load data
    df = pd.read_csv(FINAL_CSV_PATH)

    # Ensure required columns exist for both functionalities
    required_columns_canvas = ["Drillhole.X", "Drillhole.Y", "rating"]
    required_columns_drone = ["Transformed_Pixel_X", "Transformed_Pixel_Y", "rating"]

    if not all(col in df.columns for col in required_columns_canvas):
        raise ValueError("CSV file is missing one or more required columns for canvas.")
    if not all(col in df.columns for col in required_columns_drone):
        raise ValueError(
            "CSV file is missing one or more required columns for drone image."
        )

    # Define color mapping for ratings
    def get_rating_color(rating):
        """
        Map rating categories to colors.
        A: Green, B: Yellow, C: Orange, D: Red
        """
        color_map = {
            "A": (0, 255, 0),  # Green
            "B": (0, 255, 255),  # Yellow
            "C": (0, 165, 255),  # Orange
            "D": (0, 0, 255),  # Red
        }
        return color_map.get(rating, (169, 169, 169))  # Gray for invalid ratings

    # Create rating design map (blank canvas)
    canvas_width, canvas_height = 1000, 1000
    background_color = (240, 240, 240)
    canvas = np.full((canvas_height, canvas_width, 3), background_color, dtype=np.uint8)

    # Draw drill holes on canvas
    for _, row in df.iterrows():
        x, y = int(row["Pixel_X"]), int(row["Pixel_Y"])

        # Determine color based on rating
        color = get_rating_color(row["rating"])
        cv2.circle(canvas, (x, y), 4, color, -1)
        if pd.notna(row["rating"]):
            # Add rating annotation
            cv2.putText(
                canvas,
                str(row["rating"]),
                (x + 10, y),  # Slight offset for text
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

    # Save the canvas
    cv2.imwrite(OUTPUT_CANVAS_PATH, canvas)
    print(f"Rating design map saved to: {OUTPUT_CANVAS_PATH}")

    # Overlay rating data on drone image
    drone_image = cv2.imread(DRONE_IMAGE_PATH)

    if drone_image is None:
        raise FileNotFoundError(f"Drone image not found at {DRONE_IMAGE_PATH}")

    for _, row in df.iterrows():
        drill_x, drill_y = int(row["Transformed_Pixel_X"]), int(
            row["Transformed_Pixel_Y"]
        )

        # Check if the coordinates are within the image bounds
        if 0 <= drill_x < drone_image.shape[1] and 0 <= drill_y < drone_image.shape[0]:
            # Determine color based on rating
            color = get_rating_color(row["rating"])
            cv2.circle(
                drone_image, (drill_x, drill_y), radius=4, color=color, thickness=-1
            )
            if pd.notna(row["rating"]):
                # Add rating annotation
                cv2.putText(
                    drone_image,
                    str(row["rating"]),
                    (drill_x + 10, drill_y),  # Offset for text
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

    # Save the drone image
    cv2.imwrite(OUTPUT_DRONE_IMAGE_PATH, drone_image)
    print(f"Rating drone footage overlay saved to: {OUTPUT_DRONE_IMAGE_PATH}")


if __name__ == "__main__":

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
    generate_rating(folder_names[2])
