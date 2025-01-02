import os
import cv2
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath("/Users/shuai/Desktop/Mining/dev/detectron2"))
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from scipy.spatial.distance import cdist


CV_MODEL_PATH = "/Users/shuai/Desktop/Mining/dev/output/model_final.pth"
BLAST_NAME = "C1_352_121"


def setup_inference():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Adjust to your number of classes
    cfg.MODEL.DEVICE = "cpu"
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, CV_MODEL_PATH)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    return DefaultPredictor(cfg)


def process_single_image(image_path, predictor, metadata, H, output_folder):
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

    v = Visualizer(
        im[:, :, ::-1], metadata=metadata, scale=1, instance_mode=ColorMode.IMAGE_BW
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    output_image = np.array(out.get_image()[:, :, ::-1], dtype=np.uint8)

    design_map = cv2.imread(
        f"dev/demo/{BLAST_NAME}_output/map_align_output/images/{BLAST_NAME}_drill_holes_map.png",
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
        output_folder, "output_" + os.path.basename(image_path)
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
        print(f"Nearest distance: {nearest_distance}, threshold: {max_threshold}")
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


def process_folder(
    input_folder,
    output_frame_folder,
    drill_hole_folder,
    predictor,
    metadata,
    H,
    transformed_df,
):
    if not os.path.exists(output_frame_folder):
        os.makedirs(output_frame_folder)
    if not os.path.exists(drill_hole_folder):
        os.makedirs(drill_hole_folder)

    frame_number = 0
    frame_time = 1 / 10  # 10 FPS
    all_blasting_points = []
    # Add new columns to the transformed_df
    transformed_df["Start_time"] = np.nan
    transformed_df["End_time"] = np.nan
    transformed_df["smoke_height"] = np.nan
    transformed_df["smoke_size"] = np.nan
    transformed_df["smoke_up_speed"] = np.nan
    transformed_df["smoke_color_r"] = np.nan
    transformed_df["smoke_color_g"] = np.nan
    transformed_df["smoke_color_b"] = np.nan

    for file_name in sorted(os.listdir(input_folder)):
        image_path = os.path.join(input_folder, file_name)
        if not image_path.endswith(".jpg"):
            continue

        pred_boxes, smoke_masks = process_single_image(
            image_path, predictor, metadata, H, output_frame_folder
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
            cv2.circle(drone_image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

        if len(blasting_points) == 0:
            print(f"No blasting points detected in image {image_path}.")
            continue
        match_info = find_related_drill_holes(transformed_df, blasting_points)

        for idx, match in enumerate(match_info):
            x, y = int(match["Drill_X"]), int(match["Drill_Y"])
            cv2.circle(drone_image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

            hole_id = match["Hole.id"]
            # Find corresponding row in transformed_df by Hole.id
            row_idx = transformed_df[transformed_df["Hole.id"] == hole_id].index[0]

            # Update Start_time if it is NaN (first detection)
            if pd.isna(transformed_df.loc[row_idx, "Start_time"]):
                transformed_df.loc[row_idx, "Start_time"] = round(
                    frame_number * frame_time, 3
                )
            # Update End_time to the current frame time
            transformed_df.loc[row_idx, "End_time"] = round(
                frame_number * frame_time, 3
            )

            box = pred_boxes[idx]
            x_min, y_min, x_max, y_max = box
            transformed_df.loc[row_idx, "smoke_height"] = y_max - y_min

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

        drill_hole_image_path = os.path.join(drill_hole_folder, f"drill_{file_name}")
        cv2.imwrite(drill_hole_image_path, drone_image)

        frame_number += 1

    # transformed_df caculte smoke_up_speed
    for idx, row in transformed_df.iterrows():
        if pd.isna(row["Start_time"]) or pd.isna(row["End_time"]):
            continue
        if row["End_time"] == row["Start_time"]:
            continue
        transformed_df.loc[idx, "smoke_up_speed"] = round(
            row["smoke_height"] / (row["End_time"] - row["Start_time"])
        )

    output_folder = f"dev/demo/{BLAST_NAME}_output"
    # Save the updated transformed_df to a new CSV file
    updated_csv_path = os.path.join(output_folder, f"{BLAST_NAME}_Final_Result.csv")
    transformed_df.to_csv(updated_csv_path, index=False)
    print(
        f"Processing complete. Results saved in '{output_frame_folder}' and '{drill_hole_folder}'."
    )


# Script entry point
if __name__ == "__main__":

    # extract from extracting_frames.py
    input_folder = f"dev/demo/{BLAST_NAME}_output/{BLAST_NAME}_frames"

    # generated by map_align.ipynb
    H = np.load(f"dev/demo/{BLAST_NAME}_output/map_align_output/csv/{BLAST_NAME}_homography.npy")
    transformed_coords_df = pd.read_csv(
        f"dev/demo/{BLAST_NAME}_output/map_align_output/csv/{BLAST_NAME}_transformed_coordinates.csv"
    )

    # CV model related
    metadata = MetadataCatalog.get("my_dataset_val")
    predictor = setup_inference()

    # output folders
    output_frame_folder = f"dev/demo/{BLAST_NAME}_output/frames_cv_output"
    drill_hole_folder = f"dev/demo/{BLAST_NAME}_output/frames_with_blasting_points"

    process_folder(
        input_folder,
        output_frame_folder,
        drill_hole_folder,
        predictor,
        metadata,
        H,
        transformed_coords_df,
    )
