import pandas as pd


def get_average_rating(blast_name):
    BLAST_NAME = blast_name
    # Paths
    FINAL_CSV_PATH = (
        f"demo/App/Uploads/{BLAST_NAME}_output/{BLAST_NAME}_Final_Result.csv"
    )

    # Define rating conversion dictionary
    rating_scale = {"A": 4, "B": 3, "C": 2, "D": 1}
    numeric_to_rating = {4: "A", 3: "B", 2: "C", 1: "D"}  # Exact match
    range_to_rating = {  # Define ranges for average conversion
        (3.5, 4.0): "A",
        (2.5, 3.49): "B",
        (1.5, 2.49): "C",
        (1.0, 1.49): "D",
    }

    try:
        # Load data
        df = pd.read_csv(FINAL_CSV_PATH)

        # Check if a column named 'Rating' exists
        if "rating" in df.columns:
            # Convert categorical ratings to numerical values
            df["Numeric_Rating"] = df["rating"].map(rating_scale)

            # Drop any rows where conversion failed (e.g., missing or incorrect values)
            df = df.dropna(subset=["Numeric_Rating"])

            if df.empty:
                return "Error: No valid ratings found."
            # Compute average numeric rating
            avg_numeric_rating = df["Numeric_Rating"].mean()

            # Convert numeric average back to ABCD rating
            avg_rating_abcd = next(
                (
                    letter
                    for (low, high), letter in range_to_rating.items()
                    if low <= avg_numeric_rating <= high
                ),
                "Unknown",
            )

            return avg_rating_abcd
        else:
            return "Error: 'Rating' column not found in CSV."
    except FileNotFoundError:
        return f"Error: File '{FINAL_CSV_PATH}' not found."
    except pd.errors.EmptyDataError:
        return "Error: The CSV file is empty."
    except Exception as e:
        return f"An error occurred: {e}"
