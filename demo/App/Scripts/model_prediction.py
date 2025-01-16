import os
import cv2
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
import pickle
import warnings
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# # Dynamically resolve the directory path relative to the current script
# base_dir = os.path.dirname(
#     os.path.abspath(__file__)
# )  # Get the directory of the current script
# project_root = os.path.abspath(os.path.join(base_dir, "../.."))
# detectron2_path = os.path.join(project_root, "detectron2")  # Build the relative path
# sys.path.insert(0, detectron2_path)  # Add it to sys.path


from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from scipy.spatial.distance import cdist


CV_MODEL_PATH = "Models/cv_model.pth"
RF_MODEL_PATH = "Models/random_forest_model.pkl"


def setup_inference():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cpu"
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.WEIGHTS = CV_MODEL_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    return DefaultPredictor(cfg)


def process_single_image(blast_name, image_path, predictor, metadata, H, output_folder):
    im = cv2.imread(image_path)
    if im is None:
        raise ValueError(f"Failed to load image at {image_path}")

    outputs = predictor(im)

    # Extract the prediction masks and bounding boxes
    pred_boxes = outputs["instances"].pred_boxes
    pred_masks = (
        outputs["instances"].pred_masks
        if outputs["instances"].has("pred_masks")
        else None
    )

    # Remove probabilities before passing to Visualizer
    instances = outputs["instances"].to("cpu")

    if instances.has("scores"):
        # Remove scores to avoid showing probabilities
        instances.remove("scores")

    if instances.has("pred_classes"):
        instances.remove("pred_classes")

    v = Visualizer(
        im[:, :, ::-1], metadata=metadata, scale=1, instance_mode=ColorMode.IMAGE_BW
    )
    out = v.draw_instance_predictions(instances)
    output_image = np.array(out.get_image()[:, :, ::-1], dtype=np.uint8)

    design_map = cv2.imread(
        f"demo/App/Uploads/{blast_name}_output/map_align_output/images/{blast_name}_drill_holes_map.png",
        cv2.IMREAD_UNCHANGED,
    )
    if design_map is None:
        raise ValueError("Failed to load drill_hole_map.png")

    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2BGRA)
    aligned_map = cv2.warpPerspective(
        design_map,
        H,
        (output_image.shape[1], output_image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    alpha_map = aligned_map[:, :, 3] / 255.0
    for c in range(3):
        output_image[:, :, c] = (
            alpha_map * aligned_map[:, :, c] + (1 - alpha_map) * output_image[:, :, c]
        ).astype(np.uint8)

    output_image[:, :, 3] = np.maximum(output_image[:, :, 3], aligned_map[:, :, 3])
    output_filename = os.path.join(
        output_folder, "cv_output_" + os.path.basename(image_path)
    )
    cv2.imwrite(output_filename, output_image)

    return pred_boxes, pred_masks


def find_related_drill_holes(transformed_df, detected_points):
    drill_coords = transformed_df[["Transformed_Pixel_X", "Transformed_Pixel_Y"]].values
    detected_points_np = np.array(detected_points)

    # Adjust the threshold as needed
    distances_within_df = cdist(drill_coords, drill_coords)
    np.fill_diagonal(distances_within_df, np.inf)
    # max_threshold = np.min(distances_within_df)
    neighbor_count = 4
    # Calculate the threshold as the maximum distance among the k-nearest neighbors
    nearest_distances = np.sort(distances_within_df, axis=1)[
        :, :neighbor_count
    ]  # Take k-nearest distances
    max_threshold = np.max(
        nearest_distances
    )  # Set threshold as the largest of these distances

    distances = cdist(detected_points_np, drill_coords)
    results = []
    for idx, point in enumerate(detected_points):
        nearest_idx = np.argmin(distances[idx])
        nearest_distance = distances[idx][nearest_idx]
        # print(f"Nearest distance: {nearest_distance}, threshold: {max_threshold}")
        # Check if the nearest distance is within the threshold
        if nearest_distance <= max_threshold:
            results.append(
                {
                    "Detected Point": point,
                    "Drill_X": transformed_df.iloc[nearest_idx]["Transformed_Pixel_X"],
                    "Drill_Y": transformed_df.iloc[nearest_idx]["Transformed_Pixel_Y"],
                    "Pattern.Name": transformed_df.iloc[nearest_idx]["Pattern.Name"],
                    "Hole.id": transformed_df.iloc[nearest_idx]["Hole.id"],
                }
            )

    return results


def map_rating_to_category(rating):
    if rating == 0:
        return "A"
    elif rating == 1:
        return "B"
    elif rating == 2:
        return "C"
    elif rating == 3:
        return "D"
    else:
        return "Unknown"  # Fallback for unexpected ratings


def process_folder(
    blast_name,
    input_folder,
    output_frame_folder,
    drill_hole_folder,
    combined_output_folder,
    predictor,
    metadata,
    H,
    transformed_df,
):
    if not os.path.exists(output_frame_folder):
        os.makedirs(output_frame_folder)
    if not os.path.exists(drill_hole_folder):
        os.makedirs(drill_hole_folder)
    if not os.path.exists(combined_output_folder):
        os.makedirs(combined_output_folder)

    frame_number = 0
    frame_time = 1 / 10  # 10 FPS
    all_blasting_points = []

    # Add new columns to the transformed_df
    transformed_df["start_time"] = np.nan
    transformed_df["end_time"] = np.nan
    transformed_df["duration"] = np.nan
    transformed_df["duration_missing"] = np.nan
    transformed_df["start_frame_name"] = None
    transformed_df["end_frame_name"] = None
    transformed_df["smoke_height"] = np.nan
    transformed_df["smoke_size"] = np.nan
    transformed_df["smoke_up_speed"] = np.nan
    transformed_df["speed_missing"] = np.nan
    transformed_df["smoke_color_r"] = np.nan
    transformed_df["smoke_color_g"] = np.nan
    transformed_df["smoke_color_b"] = np.nan
    transformed_df["rf_output"] = np.nan
    transformed_df["rating"] = None

    total_files = len(os.listdir(input_folder))

    # Initialize Streamlit progress bar
    progress_bar = st.progress(0)
    progress_text = st.text("Processing frames...")

    with tqdm(total=total_files, desc="Processing Frames") as pbar:
        for file_name in sorted(os.listdir(input_folder)):
            pbar.update(1)  # Update progress bar
            image_path = os.path.join(input_folder, file_name)
            if not image_path.endswith(".jpg"):
                continue

            frame_name = image_path.split("/")[-1]

            pred_boxes, smoke_masks = process_single_image(
                blast_name, image_path, predictor, metadata, H, output_frame_folder
            )

            pred_boxes = pred_boxes.tensor.to("cpu").numpy()
            smoke_masks = smoke_masks.to("cpu").numpy()

            blasting_points = []
            for box in pred_boxes:
                x_min, y_min, x_max, y_max = box
                center_x = (x_min + x_max) / 2
                center_y = y_max
                blasting_points.append((center_x, center_y))

            all_blasting_points.extend(blasting_points)

            # Draw drill holes on the drone image
            drone_image = cv2.imread(image_path)
            for _, row in transformed_df.iterrows():
                x, y = int(row["Transformed_Pixel_X"]), int(row["Transformed_Pixel_Y"])
                cv2.circle(
                    drone_image, (x, y), radius=5, color=(0, 255, 0), thickness=-1
                )

            if len(blasting_points) == 0:
                # print(f"No blasting points detected in image {image_path}.")
                continue
            match_info = find_related_drill_holes(transformed_df, blasting_points)

            # Open cv_output image
            cv_output_image_path = os.path.join(
                output_frame_folder, "cv_output_" + file_name
            )
            cv_output_image = cv2.imread(cv_output_image_path)

            for idx, match in enumerate(match_info):
                x, y = int(match["Drill_X"]), int(match["Drill_Y"])
                cv2.circle(
                    drone_image, (x, y), radius=5, color=(0, 0, 255), thickness=-1
                )
                cv2.circle(
                    cv_output_image, (x, y), radius=5, color=(0, 0, 255), thickness=-1
                )

                hole_id = match["Hole.id"]
                # Find corresponding row in transformed_df by Hole.id
                row_idx = transformed_df[transformed_df["Hole.id"] == hole_id].index[0]

                # Update start_time if it is NaN (first detection)
                if pd.isna(transformed_df.loc[row_idx, "start_time"]):
                    transformed_df.loc[row_idx, "start_frame_name"] = frame_name
                    transformed_df.loc[row_idx, "start_time"] = round(
                        frame_number * frame_time, 3
                    )
                # Update end_time to the current frame time
                transformed_df.loc[row_idx, "end_time"] = round(
                    frame_number * frame_time, 3
                )

                transformed_df.loc[row_idx, "end_frame_name"] = frame_name

                box = pred_boxes[idx]
                x_min, y_min, x_max, y_max = box
                transformed_df.loc[row_idx, "smoke_height"] = round(y_max - y_min)

                # Process mask for the current match
                if smoke_masks is not None and idx < len(smoke_masks):
                    mask = smoke_masks[idx]  # Get the corresponding mask
                    smoke_size = np.sum(mask > 0)  # Number of pixels in the mask
                    smoke_color = cv2.mean(drone_image, mask=mask.astype(np.uint8))[
                        :3
                    ]  # RGB average

                    transformed_df.loc[row_idx, "smoke_size"] = smoke_size
                    r, g, b = map(int, smoke_color)
                    transformed_df.loc[row_idx, "smoke_color_r"] = r
                    transformed_df.loc[row_idx, "smoke_color_g"] = g
                    transformed_df.loc[row_idx, "smoke_color_b"] = b

            drill_hole_image_path = os.path.join(
                drill_hole_folder, f"blasting_points_{file_name}"
            )
            combined_output_image_path = os.path.join(
                combined_output_folder, f"combined_{file_name}"
            )
            cv2.imwrite(drill_hole_image_path, drone_image)
            cv2.imwrite(combined_output_image_path, cv_output_image)

            frame_number += 1

            # Update Streamlit progress
            progress_bar.progress(frame_number / total_files)
            progress_text.text(
                f"Processing frame {frame_number} of {total_files} ({frame_number / total_files * 100:.2f}%)"
            )

    # transformed_df caculte smoke_up_speed
    for idx, row in transformed_df.iterrows():
        if pd.isna(row["start_time"]) or pd.isna(row["end_time"]):
            continue
        if row["end_time"] == row["start_time"]:
            continue
        transformed_df.loc[idx, "duration"] = round(row["end_time"] - row["start_time"])
        transformed_df.loc[idx, "smoke_up_speed"] = round(
            row["smoke_height"] / (row["end_time"] - row["start_time"])
        )

    # Apply transformations only to rows where start_time is not null
    condition = ~transformed_df["start_time"].isnull()

    transformed_df.loc[condition, "duration_missing"] = (
        transformed_df.loc[condition, "duration"].isnull().astype(float)
    )
    transformed_df.loc[condition, "duration"] = transformed_df.loc[
        condition, "duration"
    ].fillna(0)
    transformed_df.loc[condition, "speed_missing"] = (
        transformed_df.loc[condition, "smoke_up_speed"].isnull().astype(float)
    )
    transformed_df.loc[condition, "smoke_up_speed"] = transformed_df.loc[
        condition, "smoke_up_speed"
    ].fillna(0)

    # Load the trained model for predicting ratings
    with open(RF_MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    # Select relevant features for rating prediction
    feature_columns = [
        "smoke_height",
        "smoke_size",
        "smoke_up_speed",
        "smoke_color_r",
        "smoke_color_g",
        "smoke_color_b",
        "duration",
        "duration_missing",
        "speed_missing",
    ]
    features = transformed_df.loc[condition, feature_columns]

    # Predict ratings
    predicted_ratings = model.predict(features)
    transformed_df.loc[condition, "rf_output"] = predicted_ratings
    mapped_ratings = [map_rating_to_category(rating) for rating in predicted_ratings]

    # Assign mapped ratings to the transformed_df
    transformed_df.loc[condition, "rating"] = mapped_ratings

    output_folder = f"demo/App/Uploads/{blast_name}_output"
    # Save the updated transformed_df to a new CSV file
    updated_csv_path = os.path.join(output_folder, f"{blast_name}_Final_Result.csv")
    transformed_df.to_csv(updated_csv_path, index=False)
    print(
        f"Processing complete. Results saved in '{output_frame_folder}' and '{drill_hole_folder}'."
    )
    # Finalizing progress
    progress_bar.progress(1.0)


def run_blast_processing(blast_name):
    # extract from extracting_frames.py
    input_folder = f"demo/App/Uploads/{blast_name}_output/{blast_name}_frames"

    # generated by map_align.ipynb
    H = np.load(
        f"demo/App/Uploads/{blast_name}_output/map_align_output/csv/{blast_name}_homography.npy"
    )
    transformed_coords_df = pd.read_csv(
        f"demo/App/Uploads/{blast_name}_output/map_align_output/csv/{blast_name}_transformed_coordinates.csv"
    )

    # CV model related
    metadata = MetadataCatalog.get("my_dataset_val")
    predictor = setup_inference()

    # output folders
    output_frame_folder = f"demo/App/Uploads/{blast_name}_output/frames_cv_output"
    drill_hole_folder = (
        f"demo/App/Uploads/{blast_name}_output/frames_with_blasting_points"
    )
    combined_output_folder = f"demo/App/Uploads/{blast_name}_output/frames_combined"

    process_folder(
        blast_name,
        input_folder,
        output_frame_folder,
        drill_hole_folder,
        combined_output_folder,
        predictor,
        metadata,
        H,
        transformed_coords_df,
    )
