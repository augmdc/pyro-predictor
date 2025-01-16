import cv2
import pandas as pd
import numpy as np


def generate_time_sequence(blast_name):
    BLAST_NAME = blast_name

    # Paths
    FINAL_CSV_PATH = f"demo/App/Uploads/{BLAST_NAME}_output/{BLAST_NAME}_Final_Result.csv"
    DRONE_IMAGE_PATH = (
        f"demo/App/Uploads/{BLAST_NAME}_output/{BLAST_NAME}_frames/{BLAST_NAME}_frame_0000.jpg"
    )
    OUTPUT_CANVAS_PATH = (
        f"demo/App/Uploads/{BLAST_NAME}_output/{BLAST_NAME}_time_sequence_design_map.png"
    )
    OUTPUT_DRONE_IMAGE_PATH = (
        f"demo/App/Uploads/{BLAST_NAME}_output/{BLAST_NAME}_time_sequence_drone_footage.png"
    )

    # Load data
    df = pd.read_csv(FINAL_CSV_PATH)

    # Ensure required columns exist for both functionalities
    required_columns_canvas = ["Drillhole.X", "Drillhole.Y", "start_time"]
    required_columns_drone = [
        "Transformed_Pixel_X",
        "Transformed_Pixel_Y",
        "start_time",
    ]

    if not all(col in df.columns for col in required_columns_canvas):
        raise ValueError("CSV file is missing one or more required columns for canvas.")
    if not all(col in df.columns for col in required_columns_drone):
        raise ValueError(
            "CSV file is missing one or more required columns for drone image."
        )

    # Normalize start_time for color mapping
    if df["start_time"].notna().any():
        start_min = df["start_time"].min()
        start_max = df["start_time"].max()
    else:
        start_min = 0
        start_max = 1

    def get_color(value):
        """
        Map start_time to a color gradient (blue to red).
        """
        normalized_value = (value - start_min) / (start_max - start_min)
        red = int(255 * (1 - normalized_value))
        blue = int(255 * normalized_value)
        return (blue, 0, red)

    # Create time-sequence design map (blank canvas)
    canvas_width, canvas_height = 1000, 1000
    background_color = (240, 240, 240)
    canvas = np.full((canvas_height, canvas_width, 3), background_color, dtype=np.uint8)

    x_min, x_max = df["Drillhole.X"].min(), df["Drillhole.X"].max()
    y_min, y_max = df["Drillhole.Y"].min(), df["Drillhole.Y"].max()

    def normalize(value, min_val, max_val, new_min, new_max):
        return int(
            (value - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
        )

    # Draw drill holes on canvas
    for _, row in df.iterrows():
        # Normalize coordinates
        x = normalize(row["Drillhole.X"], x_min, x_max, 50, canvas_width - 50)
        y = normalize(row["Drillhole.Y"], y_min, y_max, 50, canvas_height - 50)

        # Determine color based on start_time
        if pd.notna(row["start_time"]):
            color = get_color(row["start_time"])
            cv2.circle(canvas, (x, y), 4, color, -1)
            # Add start_time annotation
            cv2.putText(
                canvas,
                str(row["start_time"]),
                (x + 10, y),  # Slight offset for text
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
        else:
            color = (169, 169, 169)  # Gray for holes without start_time
            cv2.circle(canvas, (x, y), 4, color, -1)

    # Save the canvas
    cv2.imwrite(OUTPUT_CANVAS_PATH, canvas)
    print(f"Time-sequence design map saved to: {OUTPUT_CANVAS_PATH}")

    # Overlay time-sequence data on drone image
    drone_image = cv2.imread(DRONE_IMAGE_PATH)

    if drone_image is None:
        raise FileNotFoundError(f"Drone image not found at {DRONE_IMAGE_PATH}")

    for _, row in df.iterrows():
        drill_x, drill_y = int(row["Transformed_Pixel_X"]), int(
            row["Transformed_Pixel_Y"]
        )

        # Check if the coordinates are within the image bounds
        if 0 <= drill_x < drone_image.shape[1] and 0 <= drill_y < drone_image.shape[0]:
            # Determine color based on start_time
            if pd.notna(row["start_time"]):
                color = get_color(row["start_time"])
                cv2.circle(
                    drone_image, (drill_x, drill_y), radius=4, color=color, thickness=-1
                )
                # Add start_time annotation
                cv2.putText(
                    drone_image,
                    str(row["start_time"]),
                    (drill_x + 10, drill_y),  # Offset for text
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
            else:
                # Use gray for holes without start_time
                cv2.circle(
                    drone_image,
                    (drill_x, drill_y),
                    radius=4,
                    color=(169, 169, 169),
                    thickness=-1,
                )

    # Save the drone image
    cv2.imwrite(OUTPUT_DRONE_IMAGE_PATH, drone_image)
    print(f"Time-sequence drone footage overlay saved to: {OUTPUT_DRONE_IMAGE_PATH}")


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
    generate_time_sequence(folder_names[2])
