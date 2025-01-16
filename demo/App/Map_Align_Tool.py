import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os
import cv2
import numpy as np

# Set the Streamlit page configuration to wide mode
st.set_page_config(layout="wide")

# Get blast_name from URL query parameters
# Check if 'blast_name' exists in query parameters before accessing
if "blast_name" in st.query_params:
    blast_name = st.query_params["blast_name"]
else:
    st.error(
        "Missing 'blast_name' in query parameters. Please provide a valid blast name in the URL."
    )
    st.stop()

if not blast_name:
    st.warning(
        "No blast name provided. Please pass a valid blast name in the URL query parameter."
    )
    st.stop()

# Paths for primary and secondary images
base_path = "demo/App/Uploads"
base_output_path = f"{base_path}/{blast_name}_output"
frame_image_path = os.path.join(
    base_path, f"{blast_name}_output/{blast_name}_frames/{blast_name}_frame_0000.jpg"
)
map_image_path = os.path.join(
    base_path,
    f"{blast_name}_output/map_align_output/images/{blast_name}_drill_holes_map.png",
)

if not os.path.exists(frame_image_path):
    st.error(f"Image for blast name '{blast_name}' not found at {frame_image_path}.")
    st.stop()

if not os.path.exists(map_image_path):
    st.error(f"Map image for blast name '{blast_name}' not found at {map_image_path}.")
    st.stop()

# Load the background images
frame_image = Image.open(frame_image_path)
map_image = Image.open(map_image_path)


# Adjust canvas dimensions for design map with correct aspect ratio
def get_canvas_dimensions(image, canvas_width):
    img_width, img_height = image.size
    aspect_ratio = img_height / img_width
    canvas_height = int(canvas_width * aspect_ratio)
    return canvas_width, canvas_height


canvas_width = 1000
frame_canvas_width, frame_canvas_height = get_canvas_dimensions(
    frame_image, canvas_width
)
map_canvas_width, map_canvas_height = get_canvas_dimensions(map_image, canvas_width)

# Sidebar for drawing settings
drawing_mode = st.sidebar.selectbox("Drawing tool:", ("point"))

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)

if drawing_mode == "point":
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 6)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

realtime_update = st.sidebar.checkbox("Update in realtime", True)


# Function to handle canvas drawing and point selection
def handle_canvas(image, canvas_key, canvas_width, canvas_height):
    return st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=image.resize((canvas_width, canvas_height)),
        update_streamlit=realtime_update,
        height=canvas_height,
        width=canvas_width,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == "point" else 0,
        key=canvas_key,
    )


# Display canvases for frame and map images
st.subheader("Select Points on Frame Image")
frame_canvas = handle_canvas(
    frame_image, "frame_canvas", frame_canvas_width, frame_canvas_height
)

st.subheader("Select Points on Map Image")
map_canvas = handle_canvas(map_image, "map_canvas", map_canvas_width, map_canvas_height)

# Collect points from canvases
frame_points = []
map_points = []

if frame_canvas.json_data is not None:
    objects = pd.json_normalize(frame_canvas.json_data["objects"])
    if not objects.empty:
        frame_points = objects[["left", "top"]].values.tolist()

if map_canvas.json_data is not None:
    objects = pd.json_normalize(map_canvas.json_data["objects"])
    if not objects.empty:
        map_points = objects[["left", "top"]].values.tolist()


# Scale points back to original image dimensions
def scale_points(points, original_width, original_height, canvas_width, canvas_height):
    x_scale = original_width / canvas_width
    y_scale = original_height / canvas_height
    return [[x * x_scale, y * y_scale] for x, y in points]


frame_points = scale_points(
    frame_points,
    frame_image.width,
    frame_image.height,
    frame_canvas_width,
    frame_canvas_height,
)
map_points = scale_points(
    map_points, map_image.width, map_image.height, map_canvas_width, map_canvas_height
)

# Organize points into groups of four
frame_point_groups = [frame_points[i : i + 4] for i in range(0, len(frame_points), 4)]
map_point_groups = [map_points[i : i + 4] for i in range(0, len(map_points), 4)]

# st.write("Frame Points (Grouped):", frame_point_groups)
# st.write("Map Points (Grouped):", map_point_groups)

# Button to trigger Homography computation
if st.button("Apply Align Map"):
    if frame_point_groups and map_point_groups:
        for frame_group, map_group in zip(frame_point_groups, map_point_groups):
            if len(frame_group) == 4 and len(map_group) == 4:
                design_points = np.array(map_group, dtype=np.float32)
                drone_points = np.array(frame_group, dtype=np.float32)

                # Compute homography
                H, status = cv2.findHomography(design_points, drone_points)

                # Save homography matrix
                output_dir_csv = os.path.join(
                    base_path, f"{blast_name}_output/map_align_output/csv"
                )
                os.makedirs(output_dir_csv, exist_ok=True)
                homography_matrix_path = os.path.join(
                    output_dir_csv, f"{blast_name}_homography.npy"
                )
                np.save(homography_matrix_path, H)
                st.write(f"Homography matrix saved to {homography_matrix_path}")

                # Overlay design map onto drone image
                design_map = cv2.imread(map_image_path, cv2.IMREAD_UNCHANGED)
                drone_image = cv2.imread(frame_image_path, cv2.IMREAD_UNCHANGED)
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
                        alpha_map * aligned_map[:, :, c]
                        + (1 - alpha_map) * drone_image[:, :, c]
                    ).astype(np.uint8)
                drone_image[:, :, 3] = np.maximum(
                    drone_image[:, :, 3], aligned_map[:, :, 3]
                )

                overlay_result_path = os.path.join(
                    base_output_path,
                    f"map_align_output/images/{blast_name}_overlay.png",
                )
                cv2.imwrite(overlay_result_path, drone_image)
                st.write(f"Overlay result saved to {overlay_result_path}")

                # --- Step 4: Transform and Save Drillhole Coordinates ---
                file_path = f"{output_dir_csv}/{blast_name}_coordinates.csv"
                df = pd.read_csv(file_path)
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
                    output_dir_csv, f"{blast_name}_transformed_coordinates.csv"
                )
                df.to_csv(transformed_csv_path, index=False)
                print(f"Transformed coordinates saved to {transformed_csv_path}")

    else:
        st.warning(
            "Please select at least 4 points on both images to compute homography."
        )
