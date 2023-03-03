# README

# AnyFace: A Data-Centric Approach For Input-Agnostic Face Detection 
<img src="https://github.com/IS2AI/AnyFace/blob/main/anyface_detections.png?raw=true">

## Preprint on TechRxiv: [AnyFace: A Data-Centric Approach For Input-Agnostic Face Detection](https://www.techrxiv.org/articles/preprint/AnyFace_A_Data-Centric_Approach_For_Input-Agnostic_Face_Detection/21656993)

## Online Demo Page https://issai.nu.edu.kz/anyface/ 

## Installation requirements

Clone the repository and install all necessary packages:

```bash
git clone https://github.com/IS2AI/AnyFace.git
cd AnyFace
pip install -r requirements.txt
```

## Datasets

### The following datasets were used to train, validate, and test the models

Download them from the following links and rename their folders accordingly.

| Dataset | Link | Folder Name |
| --- | --- | --- |
| Wider Face | http://shuoyang1213.me/WIDERFACE/ | widerface |
| AnimalWeb | https://fdmaproject.wordpress.com/author/fdmaproject/ | animalweb |
| iCartoonFace | https://github.com/luxiangju-PersonAI/iCartoonFace#dataset | icartoon |
| TFW | https://github.com/IS2AI/TFW#downloading-the-dataset | tfw |

### Additional datasets

The following datasets were used to test the best model in addition to the test set of the main dataset. You should download them from the following links into the `original-external/` directory. Then, rename the folders as shown in the table.

| Dataset | Link | Folder Name |
| --- | --- | --- |
| Oxford-IIIT Pet Dataset | https://www.robots.ox.ac.uk/vgg/data/pets/ | pets |
| CUB-200-2011 | https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images | birds |
| Artistic-Faces| https://faculty.runi.ac.il/arik/site/foa/artistic-faces-dataset.asp | artisticfaces |
| MetFaces | https://github.com/NVlabs/metfaces-dataset | metfaces |
| Sea Turtle Faces | https://www.kaggle.com/datasets/smaranjitghose/sea-turtle-face-detection | turtles |
| Anime-Face-Detector | https://github.com/qhgz2013/anime-face-detector | animefaces |
| Tom & Jerry's Faces | https://www.kaggle.com/datasets/boltuzamaki/tom-and-jerrys-face-detection-dateset | tj |
| Labeled Fishes in the Wild Dataset | https://swfscdata.nmfs.noaa.gov/labeled-fishes-in-the-wild/ | fishes |

## Preprocessing Step

Use notebooks in the main directory to pre-process the corresponding datasets.

The preprocessed datasets are saved in `dataset/` directory. For each dataset, images are stored in `dataset/<dataset_name>/images/` and the corresponding labels are stored in `dataset/dataset_name/labels/` and in `dataset/<dataset_name>/labels_eval/`. Labels are saved in `.txt` files, where each `.txt` file has the same filename as corresponding image. Two labeling formats are used:

Annotations in `dataset/<dataset_name>/labels/` follow the format used for training YOLOv5Face models:
- `face x_center y_center width height x1 y1 x2 y2 x3 y3 x4 y4 x5 y5`
- `x1,y1,...,x5,y5` correspond to the coordinates of the left eye, right eye, nose top, left mouth corner, and right mouth corner
- all coordinates are normalized to values [0-1]
- for faces without annotated landmarks, -1 is used instead of coordinates

Annotations in `dataset/dataset_name/labels_eval/` follow the format used for [Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics#create-the-ground-truth-files):

- each row contains labels of one face
- the format: `face x_left y_top width height`
- the coordinates are NOT normalized.

To make a random selection of 160 images from each external dataset, run the cells in [external_selection.ipynb](external_selection.ipynb). Selected images for each dataset are in `selected-external/<dataset_name>/` directory.

## Training Step

Run the following command to train the YOLOv5Face model on WIDER Face, AnimalWeb, iCartoonFace and TFW datasets. The paths to these datasets are specified in the [yolov5-face/data/agnostic.yaml](yolov5-face/data/agnostic.yaml) file. The model type is selected by `--cfg models/yolov5l6.yaml`. The weights are randomly initialized `--weights ''`. However, pre-trained weights can be also used by providing an appropriate path. We used default hyperparameters inside [yolov5-face/data/hyp.scratch.yaml](yolov5-face/data/hyp.scratch.yaml). They can be changed by providing a path to `--hyp` argument. The full list of arguments is in [yolov5-face/train.py](yolov5-face/train.py).

```bash
cd yolov5-face
CUDA_VISIBLE_DEVICES="0,1" python train.py --data data/agnostic.yaml --cfg models/yolov5l6.yaml --weights '' --workers 32 --name 'exp1' --batch-size 128 --epochs 350
```

The trained model is saved in `yolov5-face/runs/train/<exp_name>/weights/` directory.

For more information about training details, please refer to the [YOLOv5](https://github.com/ultralytics/yolov5) and [YOLOv5-face](https://github.com/deepcam-cn/yolov5-face) repository.

## Testing Step

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

Download the most accurate model, YOLOv5l6, from [Google Drive](https://drive.google.com/file/d/1PuITYKZbpo6fFMX6mExN-sq_z54iWyDJ/view?usp=share_link) and save it in `yolov5-face/weights` directory. Then, run `yolov5-face/detect_face.py` to detect faces and facial landmarks on an image by providing image size, the path to weights and image.

```bash
python detect_face.py --weights weights/yolov5l6_best.pt  --source image_path --img-size 800
```

## In case of using our work in your research, please cite this paper
```
@article{Kuzdeuov2022, 
author = "Askat Kuzdeuov and Darina Koishigarina and Hüseyin Atakan Varol", 
title = "{AnyFace: A Data-Centric Approach For Input-Agnostic Face Detection}", 
year = "2022", 
month = "12", 
url = "https://www.techrxiv.org/articles/preprint/AnyFace_A_Data-Centric_Approach_For_Input-Agnostic_Face_Detection/21656993", 
doi = "10.36227/techrxiv.21656993.v1" } 
```

## References

[https://github.com/deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)

[https://github.com/rafaelpadilla/Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)
