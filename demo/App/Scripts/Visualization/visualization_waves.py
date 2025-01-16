import os
import cv2
import pandas as pd
import numpy as np


def generate_wave(blast_name):
    BLAST_NAME = blast_name

    FINAL_CSV_PATH = (
        f"demo/App/Uploads/{BLAST_NAME}_output/{BLAST_NAME}_Final_Result.csv"
    )
    FRAMES_FOLDER = f"demo/App/Uploads/{BLAST_NAME}_output/{BLAST_NAME}_frames"
    OUTPUT_FOLDER = f"demo/App/Uploads/{BLAST_NAME}_output/frames_wave"

    # Ensure the output folder exists
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Load the final CSV
    final_df = pd.read_csv(FINAL_CSV_PATH)

    # Add a list to store changed edges and their start times
    changed_edges = []

    def draw_shape_with_changes(
        output_path, blasting_points, prev_hull_edges, start_time
    ):
        if blasting_points:
            points_array = np.array(blasting_points, dtype=np.int32)
            image = cv2.imread(output_path)

            # Calculate the convex hull
            hull = cv2.convexHull(points_array)
            hull_edges = []

            # Extract edges (pairs of points) from the hull
            for i in range(len(hull)):
                pt1 = tuple(hull[i][0])
                pt2 = tuple(hull[(i + 1) % len(hull)][0])  # Loop to the first point
                hull_edges.append((pt1, pt2))
                cv2.circle(image, pt1, radius=5, color=(255, 0, 255), thickness=-1)
                cv2.circle(image, pt2, radius=5, color=(255, 0, 255), thickness=-1)

            # Draw edges with simplified change detection
            for edge in hull_edges:
                pt1, pt2 = edge

                if prev_hull_edges:
                    matched = False
                    for prev_edge in prev_hull_edges:
                        prev_pt1, prev_pt2 = prev_edge

                        # Check if edges share a common point
                        common_points = set(edge) & set(prev_edge)
                        if common_points:
                            # Check if edges are in the same or opposite direction
                            if is_same_or_opposite_direction(edge, prev_edge):
                                # Find the non-overlapping segment
                                non_overlap = get_non_overlap(edge, prev_edge)
                                if non_overlap:
                                    # Add the non-overlapping segment to changed edges
                                    changed_edges.append((non_overlap, start_time))
                                    # Draw the non-overlapping segment in red
                                    cv2.line(
                                        image,
                                        non_overlap[0],
                                        non_overlap[1],
                                        (0, 0, 255),
                                        thickness=2,
                                    )
                                # Draw the overlapping part in green
                                overlap = list(common_points)
                                if len(overlap) == 2:  # If the overlap is a line
                                    cv2.line(
                                        image,
                                        overlap[0],
                                        overlap[1],
                                        (0, 255, 0),
                                        thickness=2,
                                    )
                                matched = True
                                break

                    if not matched:
                        # If no match, draw the whole edge in red and add to changed edges
                        cv2.line(image, pt1, pt2, (0, 0, 255), thickness=2)
                        changed_edges.append(((pt1, pt2), start_time))
                else:
                    # If no previous edges, draw the whole edge in red and add to changed edges
                    cv2.line(image, pt1, pt2, (0, 0, 255), thickness=2)
                    changed_edges.append(((pt1, pt2), start_time))

            # Save the updated image
            cv2.imwrite(output_path, image)

            return hull_edges  # Return the current edges for the next comparison
        return prev_hull_edges

    def is_same_or_opposite_direction(edge1, edge2):
        """Check if two edges are in the same or opposite direction."""

        def vector(pt1, pt2):
            return (pt2[0] - pt1[0], pt2[1] - pt1[1])

        vec1 = vector(edge1[0], edge1[1])
        vec2 = vector(edge2[0], edge2[1])

        # Compute cross product to check collinearity
        cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]

        if cross_product == 0:  # Vectors are collinear
            # Check if they are same or opposite direction
            dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
            return dot_product != 0  # Same or opposite if dot product is non-zero

        return False

    def get_non_overlap(edge1, edge2):
        """Find the non-overlapping segment of edge1 compared to edge2."""
        all_points = sorted(set(edge1 + edge2))  # Combine and sort all points
        if len(all_points) == 3:
            # Return the non-overlapping segment (the one not shared by both edges)
            return (all_points[0], all_points[2])
        elif len(all_points) == 4:
            # Fully distinct edges, no overlap
            return edge1
        return None

    last_output_path = ""
    last_image = None
    prev_hull_edges = None
    blasting_points = []

    # Filter out rows where 'start_time' is NaN
    final_df = final_df[final_df["start_time"].notna()]

    # Sort final_df by 'start_time' in ascending order
    final_df = final_df.sort_values(by="start_time", ascending=True).reset_index(
        drop=True
    )

    # Iterate through rows with valid start_time and end_time
    for idx, row in final_df.iterrows():
        if pd.isna(row["start_time"]) or pd.isna(row["end_time"]):
            continue

        start_frame_name = row["start_frame_name"]
        if pd.isna(start_frame_name):
            continue

        frame_path = os.path.join(FRAMES_FOLDER, start_frame_name)
        output_path = os.path.join(OUTPUT_FOLDER, start_frame_name)

        # Load the original or already labeled image
        if os.path.exists(output_path):
            image = cv2.imread(output_path)
        else:
            prev_hull_edges = draw_shape_with_changes(
                last_output_path, blasting_points, prev_hull_edges, row["start_time"]
            )
            if not os.path.exists(frame_path):
                print(f"Frame not found: {frame_path}")
                continue
            image = cv2.imread(frame_path)
            if image is None:
                print(f"Failed to load image: {frame_path}")
                continue
            last_output_path = output_path

        drill_x, drill_y = int(row["Transformed_Pixel_X"]), int(
            row["Transformed_Pixel_Y"]
        )
        blasting_points.append([drill_x, drill_y])

        # Draw a red circle at the drill hole position
        cv2.circle(image, (drill_x, drill_y), radius=5, color=(0, 0, 255), thickness=-1)

        # Add the start_time text nearby the circle
        start_time = str(row["start_time"])
        text_position = (drill_x + 10, drill_y)  # Slightly offset from the circle
        cv2.putText(
            image,
            start_time,
            text_position,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        # Save the updated image
        cv2.imwrite(output_path, image)

    prev_hull_edges = draw_shape_with_changes(
        last_output_path,
        blasting_points,
        prev_hull_edges,
        final_df["start_time"].iloc[-1],
    )

    print(
        f"Processing complete. Labeled images with shapes and changing edges saved in '{OUTPUT_FOLDER}'."
    )

    def get_color(value, start_min, start_max):
        """
        Map start_time to a color gradient (blue to red).
        """
        normalized_value = (value - start_min) / (start_max - start_min)
        red = int(255 * (1 - normalized_value))
        blue = int(255 * normalized_value)
        return (blue, 0, red)

    # Calculate the min and max start_time for normalization
    start_times = [time for _, time in changed_edges]
    start_min = min(start_times)
    start_max = max(start_times)

    # Load frame_0000 as the base image
    frame_0000_path = os.path.join(FRAMES_FOLDER, f"{BLAST_NAME}_frame_0000.jpg")
    final_image_path = (
        f"demo/App/Uploads/{BLAST_NAME}_output/{BLAST_NAME}_time_sequence_waves.jpg"
    )
    final_image = cv2.imread(frame_0000_path)

    if final_image is None:
        raise FileNotFoundError(
            f"Base image 'frame_0000' not found at {frame_0000_path}"
        )

    # Draw all changed edges with gradient colors
    for edge, time in changed_edges:
        pt1, pt2 = edge
        # Get the color based on the start_time
        color = get_color(time, start_min, start_max)
        # Draw the edge with the calculated color
        cv2.line(final_image, pt1, pt2, color, thickness=2)
        # Annotate with the start time
        mid_point = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.putText(
            final_image,
            str(time),
            mid_point,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    # Save the final image
    cv2.imwrite(final_image_path, final_image)

    print(f"Final image with gradient-colored edges saved at '{final_image_path}'.")

    # Define output image path for the design map
    OUTPUT_IMAGE_PATH = f"demo/App/Uploads/{BLAST_NAME}_output/{BLAST_NAME}_time_sequence_waves_design_map.png"

    # Load CSV file
    df = pd.read_csv(FINAL_CSV_PATH)

    # Ensure required columns exist
    required_columns = [
        "Drillhole.X",
        "Drillhole.Y",
        "Transformed_Pixel_X",
        "Transformed_Pixel_Y",
        "start_time",
    ]
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV file is missing one or more required columns.")

    # Canvas dimensions
    canvas_width = 1000
    canvas_height = 1000
    background_color = (240, 240, 240)
    canvas = np.full((canvas_height, canvas_width, 3), background_color, dtype=np.uint8)

    # Normalize data for visualization (scale coordinates to canvas size)
    x_min, x_max = df["Drillhole.X"].min(), df["Drillhole.X"].max()
    y_min, y_max = df["Drillhole.Y"].min(), df["Drillhole.Y"].max()

    def normalize(value, min_val, max_val, new_min, new_max):
        return int(
            (value - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
        )

    # Normalize start_time for color mapping
    if df["start_time"].notna().any():
        start_min = df["start_time"].min()
        start_max = df["start_time"].max()
    else:
        start_min = 0
        start_max = 1

    def get_color(value, start_min, start_max):
        """
        Map start_time to a color gradient (blue to red).
        """
        normalized_value = (value - start_min) / (start_max - start_min)
        red = int(255 * (1 - normalized_value))
        blue = int(255 * normalized_value)
        return (blue, 0, red)

    # Map Transformed_Pixel_X and Transformed_Pixel_Y to Drillhole.X and Drillhole.Y
    pixel_to_drillhole = {}
    for _, row in df.iterrows():
        pixel_coords = (
            int(row["Transformed_Pixel_X"]),
            int(row["Transformed_Pixel_Y"]),
        )
        drill_coords = (row["Drillhole.X"], row["Drillhole.Y"])
        pixel_to_drillhole[pixel_coords] = drill_coords

    # Draw all drill holes in gray
    for _, row in df.iterrows():
        if pd.notna(row["Drillhole.X"]) and pd.notna(row["Drillhole.Y"]):
            # Normalize drillhole coordinates to canvas dimensions
            x = normalize(row["Drillhole.X"], x_min, x_max, 50, canvas_width - 50)
            y = normalize(row["Drillhole.Y"], y_min, y_max, 50, canvas_height - 50)
            # Draw the drill hole as a gray circle
            cv2.circle(canvas, (x, y), radius=5, color=(169, 169, 169), thickness=-1)

    # Draw edges with gradient color based on `start_time`
    for edge, time in changed_edges:
        # Map edge points from transformed pixel coordinates to drillhole coordinates
        pixel_pt1, pixel_pt2 = edge
        drill_pt1 = pixel_to_drillhole.get(pixel_pt1)
        drill_pt2 = pixel_to_drillhole.get(pixel_pt2)
        if drill_pt1 and drill_pt2:
            # Normalize drillhole coordinates to canvas dimensions
            canvas_pt1 = (
                normalize(drill_pt1[0], x_min, x_max, 50, canvas_width - 50),
                normalize(drill_pt1[1], y_min, y_max, 50, canvas_height - 50),
            )
            canvas_pt2 = (
                normalize(drill_pt2[0], x_min, x_max, 50, canvas_width - 50),
                normalize(drill_pt2[1], y_min, y_max, 50, canvas_height - 50),
            )
            # Get the color based on the `start_time`
            color = get_color(time, start_min, start_max)
            # Draw the edge
            cv2.line(canvas, canvas_pt1, canvas_pt2, color, thickness=2)
            # Annotate with the start_time at the midpoint
            mid_point = (
                (canvas_pt1[0] + canvas_pt2[0]) // 2,
                (canvas_pt1[1] + canvas_pt2[1]) // 2,
            )
            cv2.putText(
                canvas,
                str(time),
                mid_point,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    # Save the final design map with edges
    cv2.imwrite(OUTPUT_IMAGE_PATH, canvas)
    print(
        f"Edges with gradient color based on start_time saved to: {OUTPUT_IMAGE_PATH}"
    )


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
    generate_wave(folder_names[2])
