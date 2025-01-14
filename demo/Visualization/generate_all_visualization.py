from visualization_contour import generate_contour_map
from visualization_rating import generate_rating
from visualization_time_sequence import generate_time_sequence
from visualization_waves import generate_wave

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
    folder_index = 5
    generate_rating(folder_names[folder_index])
    generate_contour_map(folder_names[folder_index])
    generate_time_sequence(folder_names[folder_index])
    generate_wave(folder_names[folder_index])
