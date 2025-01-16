import streamlit as st
import subprocess
from PIL import Image
import os
import json
import sys

# Dynamically resolve the directory path relative to the current script
base_dir = os.path.dirname(
    os.path.abspath(__file__)
)  # Get the directory of the current script
project_root = os.path.abspath(os.path.join(base_dir, "../.."))
detectron2_path = os.path.join(project_root, "detectron2")  # Build the relative path
sys.path.insert(0, detectron2_path)  # Add it to sys.path

from Scripts.extracting_frames import process_video
from Scripts.process_design_map import generate_design_map
from Scripts.model_prediction import run_blast_processing
from Scripts.generate_visualization import generate_all_visualization
from Scripts.get_rating import get_average_rating


# Ensure the Uploads folder exists
UPLOAD_FOLDER = "demo/App/Uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# File to store blast names
BLAST_NAMES_FILE = os.path.join(UPLOAD_FOLDER, "blast_names.json")


# Load or initialize blast names list
def load_blast_names():
    if os.path.exists(BLAST_NAMES_FILE):
        with open(BLAST_NAMES_FILE, "r") as f:
            return json.load(f)
    return {}


def save_blast_names(blast_names):
    """Save blast names dictionary in a human-readable JSON format."""
    with open(BLAST_NAMES_FILE, "w") as f:
        json.dump(blast_names, f, indent=4, sort_keys=True)


# Initialize blast names with stages tracking
blast_names = load_blast_names()


# Function to update the status of a stage
def update_stage_status(blast_name, stage, status):
    if blast_name in blast_names:
        blast_names[blast_name][stage] = status
        save_blast_names(blast_names)
    else:
        st.error(f"Blast name '{blast_name}' not found.")


# Helper to check stage completion
def is_stage_completed(blast_name, stage):
    return blast_names.get(blast_name, {}).get(stage, False)


# Helper function to display images from a directory in sorted order
def display_images(directory):
    images = sorted(
        [f for f in os.listdir(directory) if f.endswith(("png", "jpg", "jpeg"))]
    )
    for img in images:
        image_path = os.path.join(directory, img)
        st.image(image_path, caption=img, use_container_width=True)


# Initialize NEW_BLAST in session state
if "NEW_BLAST" not in st.session_state:
    st.session_state.NEW_BLAST = False


# Initialize session state to track selection
if "selected_blast" not in st.session_state:
    st.session_state.selected_blast = None


# App layout
st.title("Blasting Evaluation Demo")

# Sidebar for stage status and buttons
st.sidebar.header("Stage Selection")
st.sidebar.write("Select and execute a stage:")

# File upload section
st.sidebar.subheader("Upload Files")
uploaded_video = st.sidebar.file_uploader("Upload a Video File", type=["mp4"])
uploaded_csv = st.sidebar.file_uploader("Upload a CSV File", type=["csv"])

if uploaded_video and uploaded_csv:
    st.session_state.NEW_BLAST = True
    st.header("Evaluation on New Blast Data")
    # Extract blast name from CSV filename (without extension)
    blast_name = os.path.splitext(uploaded_csv.name)[0]

    video_path = os.path.join(UPLOAD_FOLDER, f"{blast_name}.mp4")
    csv_path = os.path.join(UPLOAD_FOLDER, uploaded_csv.name)

    # Save uploaded files to disk
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())
    with open(csv_path, "wb") as f:
        f.write(uploaded_csv.read())

    # Initialize entry for new blast name
    if blast_name not in blast_names:
        blast_names[blast_name] = {
            "stage_0_completed": False,
            "stage_1_completed": False,
            "stage_2&3_completed": False,
            "stage_4_completed": False,
        }
        save_blast_names(blast_names)

    st.session_state.selected_blast = blast_name
    st.success(f"Blast name '{blast_name}' added with stages initialized.")
    # st.write(f"st.session_state.selected_blast: {st.session_state.selected_blast}")

    # Provide information to the user
    st.write(f"Uploaded video saved to: `{video_path}`")
    st.write(f"Uploaded CSV saved to: `{csv_path}`")
    st.success(f"Blast name '{blast_name}' added to the list.")


else:
    st.sidebar.write("Please upload both a video and a CSV file.")


# Display blast names list in the sidebar
st.sidebar.subheader("Blast Name Selection")
if blast_names:
    # Sidebar selectbox with session state control

    blast_names_list = sorted(blast_names.keys())

    # Determine the correct index in the selectbox
    index = (
        blast_names_list.index(st.session_state.selected_blast)
        if st.session_state.selected_blast in blast_names_list
        else None
    )
    # st.write(f"st.session_state.selected_blast: {st.session_state.selected_blast}")
    # st.write(f"index: {index}")

    # Sidebar selectbox
    selected_blast_name = st.sidebar.selectbox(
        "Select a Blast Name", blast_names_list, index=index
    )

    st.sidebar.write("You selected:", selected_blast_name)
    # Detect if the selectbox was clicked
    # if selected_blast_name and selected_blast_name != st.session_state.selected_blast:
    #     st.session_state.selected_blast = selected_blast_name
    #     st.sidebar.write("You selected:", selected_blast_name)
    #     st.session_state.NEW_BLAST = False
else:
    st.sidebar.write("No blast names available. Upload a CSV file to add one.")


# Stage buttons
if st.sidebar.button("Run Stage 0: Extracting Frames"):
    st.session_state.last_button_clicked = "Stage0"

    st.header("Stage 0: Extracting Frames")
    if selected_blast_name:
        st.write(f"Processing files for blast name: `{selected_blast_name}`...")
        process_video(selected_blast_name)
        update_stage_status(selected_blast_name, "stage_0_completed", True)
        st.success("Processing completed!")

    else:
        st.warning("Please select a blast name before running this stage.")

# Initialize session state variables
if "open_map_clicked" not in st.session_state:
    st.session_state.open_map_clicked = False

if "completed_clicked" not in st.session_state:
    st.session_state.completed_clicked = False

if "last_button_clicked" not in st.session_state:
    st.session_state.last_button_clicked = None


# Open Map Button
if st.sidebar.button("Open Map Alignment Tool"):
    st.session_state.last_button_clicked = "MapAlignment"

    if selected_blast_name:
        generate_design_map(selected_blast_name)
        st.write("Click the link below to open the Map Alignment Tool in a new tab:")

        map_align_url = f"http://localhost:8502?blast_name={selected_blast_name}"
        st.markdown(
            f"[Open Map Alignment Tool]({map_align_url})", unsafe_allow_html=True
        )


if st.session_state.last_button_clicked == "MapAlignment":
    if st.button("I completed"):
        image_path = f"{UPLOAD_FOLDER}/{selected_blast_name}_output/map_align_output/images/{selected_blast_name}_overlay.png"

        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(
                image,
                caption=f"{selected_blast_name}_overlay Image",
                use_container_width=True,
            )
            update_stage_status(selected_blast_name, "stage_1_completed", True)
            st.success(
                f"Stage 1 for '{selected_blast_name}' has been marked as completed."
            )
        else:
            st.warning("Overlay image not found. Please complete the stage first.")

if st.sidebar.button("Run Stage 2&3: Model Prediction"):
    st.session_state.last_button_clicked = "Stage2&3"
    if selected_blast_name:
        st.write("Processing started...")
        run_blast_processing(selected_blast_name)
        update_stage_status(selected_blast_name, "stage_2&3_completed", True)
        st.success("Model prediction completed!")
    else:
        st.warning("Please select a blast name before running this stage.")

if st.sidebar.button("Run Stage 4: Visualization"):
    st.session_state.last_button_clicked = "Stage4"
    if selected_blast_name:
        generate_all_visualization(selected_blast_name)
        update_stage_status(selected_blast_name, "stage_4_completed", True)
        st.success("Visualization completed! Check the output directory for results.")
    else:
        st.warning("Please select a blast name before running this stage.")

# Main area for displaying output results
# st.header("Output Results")

st.write(
    """Select and execute stages from the left panel. Results will be displayed here."""
)

# Sidebar stage display
st.sidebar.subheader("Stage Completion Status")
if blast_names:
    for name, stages in blast_names.items():
        st.sidebar.write(f"**{name}:**")
        for stage, completed in stages.items():
            st.sidebar.write(f"- {stage}: {'✅' if completed else '❌'}")
else:
    st.sidebar.write("No blast names available.")


# st.write(f"st.session_state.NEW_BLAST: {st.session_state.NEW_BLAST}")
# st.write(f"selected_blast_name: {selected_blast_name}")
# Main area for displaying output results
if st.session_state.NEW_BLAST:
    if not is_stage_completed(selected_blast_name, "stage_0_completed"):
        st.header("Stage 0: Extracting Frames")
        st.write(f"Processing files for blast name: `{selected_blast_name}`...")
        process_video(selected_blast_name)
        st.success("Processing completed!")
        update_stage_status(selected_blast_name, "stage_0_completed", True)

        generate_design_map(selected_blast_name)

    if not is_stage_completed(selected_blast_name, "stage_1_completed"):
        st.header("Stage 1: Design Map Alignment")
        st.write("Click the link below to open the Map Alignment Tool in a new tab:")
        map_align_url = f"http://localhost:8502?blast_name={selected_blast_name}"
        st.markdown(
            f"[Open Design Map Alignment Tool]( {map_align_url})",
            unsafe_allow_html=True,
        )
        if st.button("I completed"):
            image_path = f"{UPLOAD_FOLDER}/{selected_blast_name}_output/map_align_output/images/{selected_blast_name}_overlay.png"
            if os.path.exists(image_path):
                image = Image.open(image_path)
                # Display the image
                st.image(
                    image,
                    caption=f"{selected_blast_name}_overlay Image",
                    use_container_width=True,
                )
                update_stage_status(selected_blast_name, "stage_1_completed", True)
                st.success(
                    f"Stage 1 for '{selected_blast_name}' has been marked as completed."
                )
            else:
                st.warning("Overlay image not found. Please complete the stage first.")
    if not is_stage_completed(selected_blast_name, "stage_2&3_completed"):
        if is_stage_completed(selected_blast_name, "stage_1_completed"):
            st.header("Stage 2&3: Model Prediction")
            if st.button("Continue to Model Prediction"):
                st.write("Processing started...")
                st.write(f"Processing files for blast name: `{selected_blast_name}`...")
                run_blast_processing(selected_blast_name)
                st.success("Model prediction completed!")
                update_stage_status(selected_blast_name, "stage_2&3_completed", True)
    if not is_stage_completed(selected_blast_name, "stage_4_completed"):
        if is_stage_completed(selected_blast_name, "stage_2&3_completed"):
            st.header("Stage 4: Visualization")
            st.write("Processing started...")
            generate_all_visualization(selected_blast_name)
            st.success("Visualization completed!")
            update_stage_status(selected_blast_name, "stage_4_completed", True)

            avg_letter = get_average_rating(selected_blast_name)

            # Define rating colors and corresponding emojis
            rating_styles = {
                "A": {"color": "green", "emoji": "🟢"},
                "B": {"color": "yellow", "emoji": "🟡"},
                "C": {"color": "orange", "emoji": "🟠"},
                "D": {"color": "red", "emoji": "🔴"},
            }

            # Get the corresponding color and emoji
            rating_color = rating_styles.get(avg_letter, {}).get(
                "color", "black"
            )  # Default black if unknown
            rating_emoji = rating_styles.get(avg_letter, {}).get(
                "emoji", "⚪"
            )  # Default white circle if unknown

            # Display formatted text
            st.markdown(
                f"""
            <h3 style="text-align:center; font-size:30px;">
                <b>Blast name:</b> <span style="color:gold; font-weight:bold;">{selected_blast_name}</span>
                <br>
                Average Rating: <span style="color:{rating_color};">{rating_emoji} {avg_letter}</span>
            </h3>
            """,
                unsafe_allow_html=True,
            )
            display_images(f"{UPLOAD_FOLDER}/{selected_blast_name}_output")
            st.session_state.NEW_BLAST = False
else:
    if blast_names and selected_blast_name:
        if is_stage_completed(selected_blast_name, "stage_2&3_completed"):

            avg_letter = get_average_rating(selected_blast_name)

            # Define rating colors and corresponding emojis
            rating_styles = {
                "A": {"color": "green", "emoji": "🟢"},
                "B": {"color": "yellow", "emoji": "🟡"},
                "C": {"color": "orange", "emoji": "🟠"},
                "D": {"color": "red", "emoji": "🔴"},
            }

            # Get the corresponding color and emoji
            rating_color = rating_styles.get(avg_letter, {}).get(
                "color", "black"
            )  # Default black if unknown
            rating_emoji = rating_styles.get(avg_letter, {}).get(
                "emoji", "⚪"
            )  # Default white circle if unknown

            # Display formatted text
            st.markdown(
                f"""
            <h3 style="text-align:center; font-size:30px;">
                <b>Blast name:</b> <span style="color:gold; font-weight:bold;">{selected_blast_name}</span>
                <br>
                Average Rating: <span style="color:{rating_color};">{rating_emoji} {avg_letter}</span>
            </h3>
            """,
                unsafe_allow_html=True,
            )

            display_images(f"{UPLOAD_FOLDER}/{selected_blast_name}_output")
