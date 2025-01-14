import pandas as pd


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
BLAST_NAME = folder_names[6]

input_file = f"dev/demo/{BLAST_NAME}_output/{BLAST_NAME}_Final_Result.csv"
output_file = f"dev/demo/{BLAST_NAME}_output/{BLAST_NAME}_label.csv"

# Read the CSV file
df = pd.read_csv(input_file)

# Filter rows where 'start_time' and 'end_time' are not NaN
filtered_df = df.dropna(subset=["start_time", "end_time"])

# Sort by 'end_time' from small to large
sorted_df = filtered_df.sort_values(by="end_time")

# Select only the 'Pattern.Name', 'Hole.id', and 'rating' columns
selected_columns_df = sorted_df[["Pattern.Name", "Hole.id", "rating"]]

# Save the resulting dataframe to a new CSV file
selected_columns_df.to_csv(output_file, index=False)

print(f"Filtered and sorted data has been saved to {output_file}")
