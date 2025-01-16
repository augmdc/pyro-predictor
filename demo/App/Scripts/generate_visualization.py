from Scripts.Visualization.visualization_contour import generate_contour_map
from Scripts.Visualization.visualization_rating import generate_rating
from Scripts.Visualization.visualization_time_sequence import generate_time_sequence
from Scripts.Visualization.visualization_waves import generate_wave


def generate_all_visualization(blast_name):
    generate_rating(blast_name)
    generate_contour_map(blast_name)
    generate_time_sequence(blast_name)
    generate_wave(blast_name)
