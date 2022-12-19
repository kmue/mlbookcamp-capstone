# mlbookcamp-capstone: Kitchenware Classification

This repository contains my capstone project for the 2022 edition of Alexey Grigorev's mlzoomcamp online course.

# Problem description

The Kitchenware Classification competition can be found on kaggle here: https://www.kaggle.com/competitions/kitchenware-classification/

The competition problem consists in correctly classifying images of kitchenware into one of 6 categories.

In order to address this problem, we will explore 2 different approaches

1) Building a neural network from scratch (similar to mlzoomcamp homework 9)

2) Using a prebuilt image classification framework (xception) and applying it to our data

Some model tuning will be applied in each case to determine which options performs better.

The resulting model will then be productionalized by setting up a docker container, and deployed to the cloud. This resource could then be used e.g. to suggest an appropriate category for selling kitchenware items on an online classifieds platform. From a more general point of view, the project structure and the two presented approaches can be reused and adapted to apply to any kind of image classification setting.

# Exploratory data analysis, model training, parameter tuning.

The training data is structured as follows (numbers of observations in brackets):
- cups (923)
- glasses (583)
- plates (986)
- spoons (798)
- forks (436)
- knives (721)

Besides the observation that the training data contains some clearly ambiguous photographs (see notebook.ipynb for example), we stop EDA here and continue directly with modeling.

## 1) Building a neural network from scratch

Similar to mlzoomcamp homework 9, we build a basic neural network using keras. The network we are using here is structured as follows:

- 1 input layer for 150x150 images
- 1 pooling layer to extract and concentrate features from the provided images
- 1 flatten layer to compress the results of the pooling layer into a single dimension
- 1 single, 64-neuron-wide dense layer which then finally maps into...
- 1 dense output layer with 6 dimensions (1 output node for each class)

We use relu activation throughout, except for the output layer where we need probability values for each of the possible predicted classes.



## 2) Using a prebuilt image classification framework (xception) and applying it to our data

For all further details with respect to EDA, training of different and parameter tuning, please refer to notebook.ipynb.

# Exporting notebook to script

The best performing model is recreated and exported using BentoMl by running the train.py script. Refer to this file for more details.

# Reproducibility

# Model deployment

# Dependency and environment management

# Containerization

# Cloud deployment

### asd

