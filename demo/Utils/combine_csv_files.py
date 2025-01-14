import os
import numpy as np
import pandas as pd

# Define folder names and base paths
folder_names = [
    "C1_328_109",
    "C1_352_121",
    "C1_352_131",
    "C1_352_132",
    "C1_352_137",
    "C1_352_312",
    "C1_364_105",
]
feature_folder = "demo"
label_folder = "RatingDataset"
output_file = "training_dataset.csv"

# Initialize an empty list to store combined data
combined_data = []

# Process each folder
for folder_name in folder_names:
    # Construct file paths
    feature_file = os.path.join(
        feature_folder, f"{folder_name}_output", f"{folder_name}_Final_Result.csv"
    )
    label_file = os.path.join(label_folder, f"{folder_name}_label.csv")

    # Check if both files exist
    if os.path.exists(feature_file) and os.path.exists(label_file):
        # Load the feature and label data
        feature_df = pd.read_csv(feature_file)
        label_df = pd.read_csv(label_file)

        feature_df = feature_df.drop(columns=["rating"])

        # Calculate duration and handle zero values
        if "start_time" in feature_df.columns and "end_time" in feature_df.columns:
            feature_df["duration"] = feature_df["end_time"] - feature_df["start_time"]
            feature_df["duration"] = feature_df["duration"].replace(0, np.nan)
            # Round duration to 1 decimal place
            feature_df["duration"] = feature_df["duration"].round(1)
            feature_df["smoke_height"] = feature_df["smoke_height"].round(1)
        else:
            feature_df["duration"] = np.nan

        # Merge the data on 'Hole.id', keeping only rows with matching IDs
        merged_df = pd.merge(
            feature_df,
            label_df[["Hole.id", "Pattern.Name", "rating"]],
            on=["Hole.id", "Pattern.Name"],
            how="inner",
        )

        # Keep only the required columns
        merged_df = merged_df[
            [
                "Pattern.Name",
                "Hole.id",
                "smoke_height",
                "smoke_size",
                "smoke_up_speed",
                "smoke_color_r",
                "smoke_color_g",
                "smoke_color_b",
                "duration",
                "rating",
                "duration_missing",
                "speed_missing",
            ]
        ]

        # Append to the combined data
        combined_data.append(merged_df)
    else:
        print(f"Warning: Missing files for {folder_name}")

# Concatenate all combined data into a single DataFrame
if combined_data:
    final_dataset = pd.concat(combined_data, ignore_index=True)
    # Save the final dataset to a CSV file
    final_dataset.to_csv(output_file, index=False)
    print(f"Training dataset saved to {output_file}")
else:
    print("No data to save. Check the input files.")
