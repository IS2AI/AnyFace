# README

# AnyFace: A Data-Centric Approach For Input-Agnostic Face Detection 
<img src="https://github.com/IS2AI/AnyFace/blob/main/anyface_detections.png?raw=true">

## Installation requirements

Clone the repository and install the necessary packages:

```bash
git clone https://github.com/IS2AI/AnyFace.git
cd AnyFace
pip install -r requirements.txt
```

## Datasets

### Datasets used for training, validation, and testing the models

Download the datasets from the following links and rename the folders accordingly.

| Dataset Name | Link | Folder Name |
| --- | --- | --- |
| WIDER face | http://shuoyang1213.me/WIDERFACE/ | widerface |
| AnimalWeb | https://fdmaproject.wordpress.com/author/fdmaproject/ | animalweb |
| iCartoonFace | https://github.com/luxiangju-PersonAI/iCartoonFace#dataset | icartoon |
| TFW | https://github.com/IS2AI/TFW#downloading-the-dataset | TFW |

### External datasets used for testing

Download the datasets from the following links into the `original-external/` directory. Then, rename the folders, as shown in the table.

| Dataset Name | Link | Folder Name |
| --- | --- | --- |
| The Oxford-IIIT Pet Dataset | https://www.robots.ox.ac.uk/vgg/data/pets/ | pets |
| CUB-200-2011 | https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images | birds |
| Artistic-Faces Dataset | https://faculty.runi.ac.il/arik/site/foa/artistic-faces-dataset.asp | artisticfaces |
| MetFaces Dataset | https://github.com/NVlabs/metfaces-dataset | metfaces |
| Sea Turtle Face Detection | https://www.kaggle.com/datasets/smaranjitghose/sea-turtle-face-detection | turtles |
| Anime-Face-Detector | https://github.com/qhgz2013/anime-face-detector | animefaces |
| Tom and Jerry’s face detection dataset | https://www.kaggle.com/datasets/boltuzamaki/tom-and-jerrys-face-detection-dateset | tj |
| The Labeled Fishes in the Wild Dataset | https://swfscdata.nmfs.noaa.gov/labeled-fishes-in-the-wild/ | fishes |

## Pre-process the datasets

Use dataset pre-processing notebooks located in the main directory to pre-process corresponding datasets.

The preprocessed datasets are saved in `dataset/` directory. For each dataset, images are stored in `dataset/<dataset_name>/images/` and the corresponding labels are stored in `dataset/dataset_name/labels/` and in `dataset/<dataset_name>/labels_eval/`. Labels are saved in .txt files, where each .txt file has the same filename as corresponding image. Two labeling formats are used:

Annotations in `dataset/<dataset_name>/labels/` follow the format used for training YOLOv5Face models:

- each row contains labels of one face
- the format: `face x_center y_center width height x1 y1 x2 y2 x3 y3 x4 y4 x5 y5`
- x1, y1, … correspond to the left eye, right eye, nose top, left mouth corner, right mouth corner landmarks
- all coordinates are normalized to values [0-1]
- for faces without annotated landmarks, -1 is used instead of coordinates

Annotations in `dataset/dataset_name/labels_eval/` follow the format used for [Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics#create-the-ground-truth-files):

- each row contains labels of one face
- the format: `face x_left y_top width height`
- the coordinates are NOT normalized.

To make a random selection of 160 images from each external dataset, run the cells in [external_selection.ipynb](external_selection.ipynb). Selected images for each dataset are in `selected-external/<dataset_name>/` directory.

## Training

Run the following command to train the YOLOv5Face model on WIDER Face, AnimalWeb, iCartoonFace and TFW datasets. The paths to these datasets are specified in the [yolov5-face/data/agnostic.yaml](yolov5-face/data/agnostic.yaml) file. The model type is selected by `--cfg models/yolov5l6.yaml`. The weights are randomly initialized `--weights ''`. However, pre-trained weights can be also used by providing an appropriate path. We used default hyperparameters inside [yolov5-face/data/hyp.scratch.yaml](yolov5-face/data/hyp.scratch.yaml). They can be changed by providing a path to `--hyp` argument. The full list of arguments is in [yolov5-face/train.py](yolov5-face/train.py).

```bash
cd yolov5-face
CUDA_VISIBLE_DEVICES="0,1" python3 train.py --data data/agnostic.yaml --cfg models/yolov5l6.yaml --weights '' --workers 32 --name 'exp1' --batch-size 128 --epochs 350
```

The trained model is saved in `yolov5-face/runs/train/<exp_name>/weights/` directory.

For more information about training details, please refer to the [YOLOv5](https://github.com/ultralytics/yolov5) and [YOLOv5-face](https://github.com/deepcam-cn/yolov5-face) repository.

## Testing

To get the predictions on the validation and test sets of the main datasets, run the [yolov5-face/evaluate.ipynb](yolov5-face/evaluate.ipynb) notebook. For the external datasets, use [yolov5-face/evaluate-external.ipynb](yolov5-face/evaluate-external.ipynb). These notebooks create detection .txt files in Object-Detection-Metrics format: `face confidence_score x_left y_top width height`. (The coordinates are not normalized)

The detections are organized as follows:

```
yolov5-face
└─── Object-Detection-Metrics
│   └─── results
│   │   └─── val
│   │   └─── test
│   │   └─── external
│   │   │   └─── <experiment_name>
│   │   │   │   └─── <dataset_name>
│   │   │   │   │   └─── detections_{IoU}_{conf}
│   │   │   │   │   │   └─── <image_name>.txt
```

To compute average precision, precision, and recall for each dataset, use the [results.ipynb](yolov5-face/Object-Detection-Metrics/results.ipynb) and [results-external.ipynb](yolov5-face/Object-Detection-Metrics/results-external.ipynb) notebooks in the Object-Detection-Metrics directory.

For the WIDER Face dataset, use [yolov5-face/evaluate-widerface.ipynb](yolov5-face/evaluate-widerface.ipynb) that stores the predictions in WIDER Face submission format:

```
< image name >
< number of faces in this image >
< x_left y_top width height >
```

Please note that for the evaluation, the dataset images need to be copied to corresponding`yolov5-face/widerface/<set>/images` directory.

Then, use a matlab script from the website of WIDER Face to compute average precisions for the validation set. For more information refer to the WIDER Face dataset website. For more information refer to the [WIDER Face dataset](http://shuoyang1213.me/WIDERFACE/) website.

To evaluate the model on the external datasets, [yolov5-face/evaluate-selected-external.ipynb](yolov5-face/evaluate-selected-external.ipynb) notebook saves the selected images with drawn predicted bounding boxes and confidence scores. The predictions are saved in `yolov5-face/selected-external-results/`.

A similar notebook is provided for arbitrary test images. The predictions are saved in `yolov5-face/test-images-results`. We also provide [yolov5-face/evaluate-test-videos.ipynb](yolov5-face/evaluate-test-videos.ipynb) which extracts frames from the input video, makes predictions on the frames and constructs video back with bounding boxes. Testing images and testing videos should be put into `/test_images/` and `/test_videos/` in the main directory.

### Augmentation

Please note that we utilize augmentation procedure at the inference to get better results. Thus, you can find `augment` argument in evaluation code. By default, augmentation is not performed when the argument is not specified or `None`. `augment=1` sets `s = [1, 0.83, 0.67]` for scaling and `f = [None, 3, None]` for flipping, where 3 is horizontal flipping. `augment=2` sets `s = [1, 0.83, 0.67, 1, 0.83, 0.67]` for scaling and `f = [None, None, None, 3, 3, 3]` for flipping, meaning that for all three image scales we infer both original and flipped image. To change these parameters refer to [yolov5-face/models/yolo.py](yolov5-face/models/yolo.py).

### Feature Visualization

To visualize CNN layers of the model for a given input image, you can use `feature_vis` parameter in [yolov5-face/evaluate-selected-external.ipynb](yolov5-face/evaluate-selected-external.ipynb). By default feature maps of the C3 block are visualized. This can be changed in [yolov5-face/models/yolo.py](yolov5-face/models/yolo.py). Feature maps are saved to `yolov5-face/feature_visualization/`.

## Inference

Use can also use yolov5-face/detect_face.py to detect faces and landmarks on an image by providing image size, the path to weights and image.

```bash
python detect_face.py --weights path_to_weights --image path_to_image --img-size 800
```

## Models and Results

**Validation Set**

| Model | WF-easy | WF-medium | WF-hard | TFW Indoor | TFW Outdoor | AnimalWeb | iCartoonFace |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5n | 92.1 | 89.6 | 76.7 | 100 | 98.16 | 94.66 | 86.04 |
| YOLOv5n6 | 93.5 | 90.7 | 76.3 | 100 | 98.36 | 95.17 | 88.13 |
| YOLOv5s | 93.8 | 91.7 | 79.9 | 100 | 98.56 | 95.25 | 87.6 |
| YOLOv5s6 | 94.5 | 92.2 | 79.7 | 100 | 98.72 | 95.7 | 89.24 |
| YOLOv5m | 95.2 | 93.4 | 83.2 | 100 | 98.79 | 95.8 | 89.85 |
| YOLOv5m6 | 95.9 | 93.8 | 82.8 | 100 | 99.09 | 96.28 | 90.24 |
| YOLOv5l | 95.8 | 93.8 | 84.9 | 100 | 99.19 | 96.17 | 90.31 |
| YOLOv5l6 | 96.1 | 94.2 | 84.1 | 100 | 99.20 | 96.26 | 90.61 |

**Testing Set**

| Model | TFW Indoor | TFW Outdoor | AnimalWeb | iCartoonFace |
| --- | --- | --- | --- | --- |
| YOLOv5n | 76.26 | 83.86 | 57.31 | 60.56 |
| YOLOv5n6 | 100 | 99.3 | 91.94 | 89.44 |
| YOLOv5s | 100 | 99.36 | 91.58 | 89.5 |
| YOLOv5s6 | 100 | 99.53 | 92.83 | 90.25 |
| YOLOv5m | 100 | 99.52 | 93.12 | 90.86 |
| YOLOv5m6 | 100 | 99.42 | 93.72 | 91.3 |
| YOLOv5l | 100 | 99.5 | 93.59 | 91.24 |
| YOLOv5l6 | 100 | 99.47 | 93.59 | 91.65 |

The most accurate model, YOLOv5l6, can be downloaded from [here](https://www.dropbox.com/s/kvxjjpgiowvy40g/AnyFace.pt?dl=0).

## References

[https://github.com/deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)

[https://github.com/rafaelpadilla/Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)
