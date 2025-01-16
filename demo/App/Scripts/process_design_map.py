import pandas as pd
import cv2
import numpy as np
import os


def generate_design_map(blast_name):

    # --- Define Paths ---
    base_path = "demo/App/Uploads"
    base_output_path = f"{base_path}/{blast_name}_output"
    map_align_output_path = os.path.join(base_output_path, "map_align_output")
    output_dir_csv = os.path.join(map_align_output_path, "csv")
    output_dir_images = os.path.join(map_align_output_path, "images")
    os.makedirs(output_dir_csv, exist_ok=True)
    os.makedirs(output_dir_images, exist_ok=True)

    # --- Input file path ---
    file_path = f"{base_path}/{blast_name}.csv"

    # --- Step 1: Preprocessing and Scaling Drillhole Coordinates ---
    df = pd.read_csv(file_path)

    df["Hole.id"] = df["Hole.id"].astype(str)
    df = df[~df["Hole.id"].str.contains("BMM", na=False)]

    x = df["Drillhole.X"].values
    y = df["Drillhole.Y"].values

    canvas_size = (1000, 1000)  # Canvas dimensions
    margin = 50  # Margin around points
    min_x, max_x = x.min(), x.max()
    min_y, max_y = y.min(), y.max()

    scaled_x = ((x - min_x) / (max_x - min_x)) * (canvas_size[0] - 2 * margin) + margin
    scaled_y = ((y - min_y) / (max_y - min_y)) * (canvas_size[1] - 2 * margin) + margin

    canvas = np.zeros((canvas_size[1], canvas_size[0], 4), dtype=np.uint8)
    for sx, sy in zip(scaled_x, scaled_y):
        cv2.circle(
            canvas, (int(sx), int(sy)), radius=6, color=(0, 255, 0, 255), thickness=-1
        )

    output_image_path = os.path.join(
        output_dir_images, f"{blast_name}_drill_holes_map.png"
    )
    cv2.imwrite(output_image_path, canvas)

    df["Pixel_X"] = scaled_x.astype(int)
    df["Pixel_Y"] = scaled_y.astype(int)
    output_csv_path = os.path.join(output_dir_csv, f"{blast_name}_coordinates.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"Map saved to {output_image_path}")
    print(f"Coordinates CSV saved to {output_csv_path}")
