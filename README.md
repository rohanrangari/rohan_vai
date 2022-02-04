multi-class-img-clf
==============================

A complete Multi-class Image Classifier using Python

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── csv
    │   │   └── train-test-data
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │   └── make_dataset.py
    │   │   └── config.py
    │   │   └── train_models  <- train_models are saved in this directory
    │   │   └──training.ipynb <- Notebook used for training
    │   │
    │   └── plots    <- Plots/ Visualizations are saved here.
    │   └── reports  <- Model Performance Metrics Reports are saved here. eg: classification-reports.csv
