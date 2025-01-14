# Drone Footage Analysis for Open Pit Blast Rating

Setting Up Git LFS for Model Files

This repository uses Git Large File Storage (Git LFS) to manage large model files, such as cv_model.pth. Please ensure Git LFS is installed and configured on your system before cloning or pulling this repository.

### Multi-Stage Pipeline

#### get detectron2 in root_directory
` git clone 'https://github.com/facebookresearch/detectron2/'`

#### stage 0

`demo/Multi-stage/extracting_frames.py`

#### stage 1
`demo/Multi-stage/map_align.py`

#### stage 2, 3 model prediction
`demo/Multi-stage/process_frames.py`

### Visualization and Video Generation
#### generate visualizations
`demo/Visualization/generate_all_visualization.py`

#### generate videos
`demo/Utils/Images_to_video.py`


### CV Model Training
Dataset for Segmentation
1.	Labeling Data
Use Labelme to label the data.
https://www.labelme.io/docs/install-labelme
2.	Converting to COCO Format
Convert the labeled data into COCO format using the script `demo/Utils/labelme2coco.py`

Training Custom Model

After preparing the dataset:
* Train your custom segmentation model using `demo/Training/cv_model_training.ipynb`

### Decision Tree Training
`demo/Training/decision_tree_training.py`
