# How to setup

## Train your own model (optional)

1. Go into `./ai/resources` folder
2. ```
   cat yolov8m_* > yolov8m.pt
   cat yolov8l_* > yolov8l.pt
   cat yolov8x_* > yolov8x.pt
   cat andrewmvd_dataset_* > andrewmvd_dataset.zip
   cat aslanahmedov_dataset_* > aslanahmedov_dataset.zip
   ```
3. Go one level up back into `./ai` folder
4. `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
5. `pip install -r requirements.txt`
6. `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` or `set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`
7. `python prepare.py`
8. Go into `train.py` and configure which pre-trained model you want to use.
9. `python train.py`

# Notes

- When adding stuff into resources, for anything over 25MB, use the following command: `split -b 25M --numeric-suffixes <name> <name>_` and add proper documentation on how to build it back together after clone.

# Acknowledgements

## ./ai/resources/yolov8\*

Yolo models were downloaded from the [ultralytics repository](https://github.com/ultralytics/ultralytics). I was unable to find any documentation on how to credit them, please, if you do, send a pull request.

## ./ai/resources/andrewmvd_dataset.zip

```
BibTeX
@misc{make ml},
title={Car License Plates Dataset},
url={https://makeml.app/datasets/cars-license-plates},
journal={Make ML}
```

I downloaded the following [kaggle dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection).

## ./ai/resources/aslanahmedov_dataset.zip

I downloaded the following [kaggle dataset](https://www.kaggle.com/datasets/aslanahmedov/number-plate-detection). The original author is [ASLAN AHMEDOV](https://www.kaggle.com/aslanahmedov), however the sources cited are "web scraping"

## General

- While learning, the following freeCodeCamp article was realy helpful to get started: [How to Detect Objects in Images Using the YOLOv8 Neural Network](https://www.freecodecamp.org/news/how-to-detect-objects-in-images-using-yolov8/) - [Andrey Germanov](https://www.freecodecamp.org/news/author/germanov_dev/)
