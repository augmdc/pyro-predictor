import pandas as pd
import cv2
import numpy as np
import os

# --- User Inputs ---

data_folder = ["Data1of3", "Data2of3", "Data3of3"]

folder_names = [
    "C1_328_109",
    "C1_352_121",
    "C1_352_131",
    "C1_352_132",
    "C1_352_137",
    "C1_352_312",
    "C1_364_105",
]
data_folder_names = data_folder[0]
BLAST_NAME = folder_names[0]

# Input file path
file_path = f"{data_folder_names}/{BLAST_NAME}/{BLAST_NAME}.csv"

# get first frame of drone footage
original_drone_image = cv2.imread(
    f"demo/{BLAST_NAME}_output/{BLAST_NAME}_frames/{BLAST_NAME}_frame_0000.jpg"
)

# --- Output directories ---

# Folder for CSV files
output_dir_csv = os.path.join(f"demo/{BLAST_NAME}_output/map_align_output", "csv")
# Folder for images
output_dir_images = os.path.join(f"demo/{BLAST_NAME}_output/map_align_output", "images")
os.makedirs(output_dir_csv, exist_ok=True)
os.makedirs(output_dir_images, exist_ok=True)

# ---- Main Script ----

# --- Step 1: Preprocessing and Scaling Drillhole Coordinates ---
df = pd.read_csv(file_path)

df["Hole.id"] = df["Hole.id"].astype(str)
# Filter out rows where "Hole.id" contains "BMM"
df = df[~df["Hole.id"].str.contains("BMM", na=False)]

# Extract and normalize coordinates
x = df["Drillhole.X"].values
y = df["Drillhole.Y"].values

canvas_size = (1000, 1000)  # Canvas dimensions
margin = 50  # Margin around points
min_x, max_x = x.min(), x.max()
min_y, max_y = y.min(), y.max()

scaled_x = ((x - min_x) / (max_x - min_x)) * (canvas_size[0] - 2 * margin) + margin
scaled_y = ((y - min_y) / (max_y - min_y)) * (canvas_size[1] - 2 * margin) + margin

# Create a blank canvas and draw points
canvas = np.zeros((canvas_size[1], canvas_size[0], 4), dtype=np.uint8)
for sx, sy in zip(scaled_x, scaled_y):
    cv2.circle(
        canvas, (int(sx), int(sy)), radius=6, color=(0, 255, 0, 255), thickness=-1
    )

# Save outputs
output_image_path = os.path.join(output_dir_images, f"{BLAST_NAME}_drill_holes_map.png")
cv2.imwrite(output_image_path, canvas)

df["Pixel_X"] = scaled_x.astype(int)
df["Pixel_Y"] = scaled_y.astype(int)
output_csv_path = os.path.join(output_dir_csv, f"{BLAST_NAME}_coordinates.csv")
df.to_csv(output_csv_path, index=False)
print(f"Map saved to {output_image_path}")
print(f"Coordinates CSV saved to {output_csv_path}")


# --- Step 2: Homography Point Selection ---
def select_points(image, window_name):
    points = []
    img_copy = image.copy()  # Create a copy of the image to update dynamically

    def mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Point selected: ({x}, {y})")

            # Draw a circle with a number at the clicked point
            cv2.circle(img_copy, (x, y), radius=7, color=(0, 0, 255), thickness=-1)
            cv2.putText(
                img_copy,
                str(len(points)),  # Number of the point
                (x + 10, y + 10),  # Offset the text slightly from the point
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5,
                color=(0, 0, 255),
                thickness=2,
            )

            # Display the updated image
            cv2.imshow(window_name, img_copy)

    cv2.namedWindow(window_name)
    cv2.imshow(window_name, img_copy)
    cv2.setMouseCallback(window_name, mouse_click)
    print(f"Select 4 points in the {window_name} window...")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or len(points) == 4:  # Press ESC or select 4 points to exit
            break

    cv2.destroyAllWindows()
    return points


# Load images
design_map = cv2.imread(output_image_path)
drone_image = original_drone_image

# Select points and calculate homography
design_points = np.array(select_points(design_map, "Design Map"), dtype=np.float32)
drone_points = np.array(select_points(drone_image, "Drone Footage"), dtype=np.float32)

H, status = cv2.findHomography(design_points, drone_points)
homography_matrix_path = os.path.join(output_dir_csv, f"{BLAST_NAME}_homography.npy")
np.save(homography_matrix_path, H)
print(f"Homography matrix saved to {homography_matrix_path}")

# --- Step 3: Overlay Design Map onto Drone Image ---
design_map = cv2.imread(output_image_path, cv2.IMREAD_UNCHANGED)
drone_image = original_drone_image
drone_image = cv2.cvtColor(drone_image, cv2.COLOR_BGR2BGRA)

aligned_map = cv2.warpPerspective(
    design_map,
    H,
    (drone_image.shape[1], drone_image.shape[0]),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0, 0),
)

alpha_map = aligned_map[:, :, 3] / 255.0
for c in range(3):
    drone_image[:, :, c] = (
        alpha_map * aligned_map[:, :, c] + (1 - alpha_map) * drone_image[:, :, c]
    ).astype(np.uint8)
drone_image[:, :, 3] = np.maximum(drone_image[:, :, 3], aligned_map[:, :, 3])

overlay_result_path = os.path.join(output_dir_images, f"{BLAST_NAME}_overlay.png")
cv2.imwrite(overlay_result_path, drone_image)
print(f"Overlay result saved to {overlay_result_path}")

# --- Step 4: Transform and Save Drillhole Coordinates ---
design_map_coords = df[["Pixel_X", "Pixel_Y"]].values
design_map_coords_homogeneous = np.hstack(
    (design_map_coords, np.ones((design_map_coords.shape[0], 1)))
)
transformed_coords_homogeneous = (H @ design_map_coords_homogeneous.T).T
transformed_coords = (
    transformed_coords_homogeneous[:, :2]
    / transformed_coords_homogeneous[:, 2, np.newaxis]
)

df["Transformed_Pixel_X"] = transformed_coords[:, 0].astype(int)
df["Transformed_Pixel_Y"] = transformed_coords[:, 1].astype(int)

transformed_csv_path = os.path.join(
    output_dir_csv, f"{BLAST_NAME}_transformed_coordinates.csv"
)
df.to_csv(transformed_csv_path, index=False)
print(f"Transformed coordinates saved to {transformed_csv_path}")
