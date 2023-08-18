# How to setup

## Prerequisites

- docker
- python 3.11.2

1. Go into `./ai/resources` folder
2. ```
   cat yolov8m_* > yolov8m.pt
   cat yolov8l_* > yolov8l.pt
   cat yolov8x_* > yolov8x.pt
   cat andrewmvd_dataset_* > andrewmvd_dataset.zip
   cat aslanahmedov_dataset_* > aslanahmedov_dataset.zip
   cp *.pt ..
   ```
3. Go back into root folder
4. `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
5. `pip install -r requirements.txt`

### Start the web server

1. Go into `./ai/server`
2. Copy `.env.development` to `.env`
3. TODO (write guide to env)
4. Setup db server (optional, depends on your `.env`)
   1. Make sure you're following Microsoft's licensing requirements, since the DB I chose is MSSQL. I did this because it's already integrated at my workplace. Personally, I use the docker only for development / testing.
   2. Go back into root folder.
   3. ```
      # (on Windows replace `pwd` with `pwd -W`)
      docker run -e "ACCEPT_EULA=Y" -e "MSSQL_SA_PASSWORD=MyV€ryStr0ngP4ssW0rĐ" \
       -p 1433:1433 --name main_gate_aplr_db --hostname main_gate_aplr_db \
       -v $(pwd)/db/data:/var/opt/mssql/data \
       -v $(pwd)/db/log:/var/opt/mssql/log \
       -v $(pwd)/db/secrets:/var/opt/mssql/secrets \
       -d mcr.microsoft.com/mssql/server:2022-latest
      ```
   4. `docker exec -it main_gate_aplr_db "bash"`
   5. `export QUERY_TO_EXECUTE="{CONTENTS_OF_./db/init.sql}"`
   6. `/opt/mssql-tools/bin/sqlcmd -S localhost -U SA -P "MyV€ryStr0ngP4ssW0rĐ" -Q "$QUERY_TO_EXECUTE"`

## Train your own model (optional)

1. Go into `./ai` folder
2. `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` (on Linux) or `set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` (on Windows)
3. `python prepare.py`
4. Go into `train.py` and configure which pre-trained model you want to use.
5. `python train.py`

### Test your model visually

1. Install [tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html)
2. `python test.py {path_to_your_model} {path_to_image_to_test}`

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
