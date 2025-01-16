import os
import cv2
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import KDTree


def generate_contour_map(blast_name):
    BLAST_NAME = blast_name

    # Paths
    FINAL_CSV_PATH = f"demo/App/Uploads/{BLAST_NAME}_output/{BLAST_NAME}_Final_Result.csv"
    OUTPUT_CANVAS_PATH = (
        f"demo/App/Uploads/{BLAST_NAME}_output/{BLAST_NAME}_rating_contour_design_map.png"
    )

    # Load data
    df = pd.read_csv(FINAL_CSV_PATH)

    # Ensure required columns exist
    required_columns = ["Pixel_X", "Pixel_Y", "rating"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV file is missing one or more required columns.")

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

    # Filter rows with valid ratings only
    valid_rows = df[df["rating"].notna()]

    # Convert ratings to numerical values
    z = valid_rows["rating"].apply(lambda r: ord(str(r)[0]) - ord("A")).values

    # Extract drill hole coordinates
    points_array = df[["Pixel_X", "Pixel_Y"]].to_numpy().astype(np.float32)

    # Calculate the convex hull
    hull = cv2.convexHull(points_array)

    # Determine bounding box of the convex hull
    min_x, max_x = np.min(hull[:, 0, 0]), np.max(hull[:, 0, 0])
    min_y, max_y = np.min(hull[:, 0, 1]), np.max(hull[:, 0, 1])

    # Fixed canvas size
    canvas_width, canvas_height = 1000, 1000

    # Create a fine grid over the fixed canvas size based on original coordinates
    grid_x, grid_y = np.mgrid[
        min_x : max_x : 1j * 500,  # Adjust resolution as needed
        min_y : max_y : 1j * 500,
    ]

    # Flatten the grid for filtering
    grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T

    # Mask points inside the convex hull
    mask = np.array(
        [cv2.pointPolygonTest(hull, tuple(p), False) >= 0 for p in grid_points]
    )
    grid_points_within_hull = grid_points[mask]
    valid_x, valid_y = grid_points_within_hull[:, 0], grid_points_within_hull[:, 1]

    # Build KDTree for distance calculations
    kdtree = KDTree(valid_rows[["Pixel_X", "Pixel_Y"]].to_numpy().astype(np.float32))

    # Define a maximum distance threshold
    max_distance = 60  # Adjust threshold as needed

    # Check distances for grid points
    distances, _ = kdtree.query(grid_points_within_hull)
    distance_mask = distances <= max_distance

    # Filter points based on distance
    valid_x = valid_x[distance_mask]
    valid_y = valid_y[distance_mask]

    # Perform interpolation only for valid points
    grid_z = griddata(
        (valid_rows["Pixel_X"], valid_rows["Pixel_Y"]),
        z,
        (valid_x, valid_y),
        method="nearest",
        fill_value=0,
    )

    # Normalize interpolated values
    grid_z_normalized = cv2.normalize(
        grid_z, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Create heatmap
    heatmap = cv2.applyColorMap(grid_z_normalized, cv2.COLORMAP_JET)

    # Initialize blank canvas
    background_color = (240, 240, 240)
    canvas = np.full((canvas_height, canvas_width, 3), background_color, dtype=np.uint8)

    # Overlay heatmap onto canvas directly using its coordinates
    for x, y, z_value, distance in zip(
        valid_x, valid_y, grid_z, distances[distance_mask]
    ):
        if distance > max_distance:
            color = (169, 169, 169)  # Gray for points outside max distance
        else:
            color = get_rating_color(chr(int(z_value) + ord("A")))
        cv2.circle(canvas, (int(x), int(y)), 2, color, -1)

    # Draw drill holes on the heatmap
    for _, row in df.iterrows():
        x, y = int(row["Pixel_X"]), int(row["Pixel_Y"])

        # Ensure coordinates are within bounds
        if 0 <= x < canvas_width and 0 <= y < canvas_height:
            # Determine color based on rating
            color = get_rating_color(row["rating"])

            # Draw the drill hole point
            cv2.circle(canvas, (x, y), 4, color, -1)

            # Add rating annotation
            if pd.notna(row["rating"]):
                cv2.putText(
                    canvas,
                    str(row["rating"]),
                    (x + 10, y),  # Slight offset for text
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),  # Black text for contrast
                    1,
                    cv2.LINE_AA,
                )

    # Save the canvas
    cv2.imwrite(OUTPUT_CANVAS_PATH, canvas)
    print(f"Polygon-constrained heatmap saved to: {OUTPUT_CANVAS_PATH}")

    # Paths
    DRONE_IMAGE_PATH = (
        f"demo/App/Uploads/{BLAST_NAME}_output/{BLAST_NAME}_frames/{BLAST_NAME}_frame_0000.jpg"
    )
    OUTPUT_DRONE_IMAGE_PATH = (
        f"demo/App/Uploads/{BLAST_NAME}_output/{BLAST_NAME}_rating_contour_drone_footage.png"
    )
    # Load drone image
    drone_image = cv2.imread(DRONE_IMAGE_PATH)
    if drone_image is None:
        raise FileNotFoundError(f"Drone image not found at {DRONE_IMAGE_PATH}")

    # Ensure drone image dimensions
    drone_image_height, drone_image_width = drone_image.shape[:2]

    H = np.load(
        f"demo/App/Uploads/{BLAST_NAME}_output/map_align_output/csv/{BLAST_NAME}_homography.npy"
    )
    # Convert the heatmap to include an alpha channel
    heatmap_with_alpha = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)  # Add alpha channel

    # Define transparency for the background (gray areas)
    transparent_color = (240, 240, 240)  # Background color used in the canvas
    for y in range(heatmap_with_alpha.shape[0]):
        for x in range(heatmap_with_alpha.shape[1]):
            # If pixel is background color, make it transparent
            if (heatmap_with_alpha[y, x, :3] == transparent_color).all():
                heatmap_with_alpha[y, x, 3] = 0  # Set alpha to 0 for transparency
            else:
                heatmap_with_alpha[y, x, 3] = 255  # Set alpha to 255 for non-background

    # Warp the heatmap to align with the drone image using the homography matrix
    aligned_heatmap_with_alpha = cv2.warpPerspective(
        heatmap_with_alpha,  # The heatmap canvas
        H,  # Homography matrix
        (drone_image_width, drone_image_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),  # Black for areas outside
    )

    # Extract the RGB and Alpha channels
    aligned_rgb = aligned_heatmap_with_alpha[:, :, :3]
    aligned_alpha = aligned_heatmap_with_alpha[:, :, 3] / 255.0

    # Blend aligned heatmap with the drone image using the alpha channel
    for y in range(drone_image.shape[0]):
        for x in range(drone_image.shape[1]):
            # Blend only where alpha is greater than 0
            alpha = aligned_alpha[y, x]
            if alpha > 0:
                drone_image[y, x] = (1 - alpha) * drone_image[
                    y, x
                ] + alpha * aligned_rgb[y, x]

    # Draw drill holes and ratings on the blended image
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
                    color=(0, 0, 0),  # Black text for contrast
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

    # Save the blended drone image with heatmap overlay
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
    generate_contour_map(folder_names[2])
