# mlbookcamp-capstone: Kitchenware Classification

This repository contains my capstone project for the 2022 edition of Alexey Grigorev's mlzoomcamp online course.

# Problem description

The Kitchenware Classification competition can be found on kaggle here: https://www.kaggle.com/competitions/kitchenware-classification/

The competition problem consists in correctly classifying images of kitchenware into one of 6 categories.

In order to address this problem, we will explore 2 different approaches

1) Build a neural network from scratch (similar to mlzoomcamp homework 9)

2) Use a prebuilt image classification framework (xception) and apply it to our data

Some model tuning will be applied in each case to determine which options performs better.

The resulting model will then be productionalized by setting up a docker container, and deployed to the cloud. This resource could then be used e.g. to suggest an appropriate category for selling kitchenware items on an online classifieds platform. From a more general point of view, the project structure and the two presented approaches can be reused and adapted to apply to any kind of image classification setting.

# Exploratory data analysis, model training, parameter tuning.

The training data is structured as follows (numbers of observations in brackets):
- cups (XXX)
- glasses (XXX)
- plates (XXX)
- spoons (XXX)
- forks (XXX)
- knives (XXX)

For all further details with respect to EDA, model training and parameter tuning, please refer to notebook.ipynb.

# Exporting notebook to script

The best performing model is recreated and exported using BentoMl by running the train.py script. Refer to this file for more details.

# Reproducibility

# Model deployment

# Dependency and environment management

# Containerization

# Cloud deployment

### asd

