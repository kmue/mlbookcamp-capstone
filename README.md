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

Once parametrized, we can then proceed to feed this neural network with training data. The latter just needs to first pass through some pre-processing (ImageDataGenerator) in order to be presented to the model as an array with the correct format and value ranges that are valid input data.

After training for 10 epochs this simple model manages to close in on validation accuracy of 0.5 (half of the validation items are classified correctly). Since both training and test accuracy are clearly still steeply increasing, we train for another 10 epochs which yields validation accuracy close to 0.6. However, training accuracy continues to improve quite aggressively while validation accuracy clearly is reaching some limit, indicating a trend towards overfitting.

In order to tackle this, we start over and after 10 epochs of initial training feed the model with augmented data, i.e. the same training images but with some "mutations" applied to the images - such as rotating them, stretching them height-, width- or otherwise, zooming, as well as flipping and stretching the content. In the end, unfortunately, this approach still appears to top out at a validation accuracy of around 0.6.

In another attempt to improve the model, we try to add (here: 20%) dropout, i.e. the randomized exclusion of some training data as it passes through the neural network. Unfortunately, even with adapted learning rate and momentum, the introduction of dropout appears to quickly lead to overfitting and does not yield consistently better validation accuracy.

## 2) Using a prebuilt image classification framework (xception) and applying it to our data

As an alternative model, we base this section on the provided starter notebook and explore further to see if we can improve the xception model.

We use the pre-trained xception model, but differently from the starter notebook we choose an increased image size of 299x299 pixels.

Using this, we immediately (after training with default settings, e.g. a learning rate of 0.01) reach validation accuracy of 0.95, compared to 0.85 in the starter notebook baseline. Unfortunately, overfitting is evident as training accuracy exceeds 0.99 after just 10 epochs of training.

We therefore next check if changing the learning rate might yield improved validation accuracy. The comparison shows that a learning rate of 0.001 should improve the outcome, so we proceed from there.

In order to not waste time training an already overfitting model for epochs over epochs, we add early stopping as soon as validation accuracy exceeds a pre-defined threshold of 0.97.

Data augmentation was explored but outcomes not saved to the present notebook.ipynb. As for the from-scratch CNN, unfortunately it did not yield better validation accuracy and was thus discarded from the process.

NB: It is currently not clear (to me) why the current early stopping implemetation makes the training process stop after an epoch where validation accuracy is 0.9577 -- we may address this issue at a later point in time but will for now focus on preparing deployment.

For all further details with respect to EDA, training of different and parameter tuning, please refer to notebook.ipynb.

# Exporting notebook to script

The best performing model is saved, recreated and exported using BentoML by running the train.py script. Refer to this file for more details.

# Reproducibility

Note that in order to run train.py, the script needs to be run on a GPU-enabled machine where BentoML is installed. The kaggle training and validation data needs to be present on the machine (see commented out section of notebook.ipynb). Model training will not run if your machine does not have a GPU. 

# Model deployment

The final model was exported using BentoML (see: train.py).

In addition, the following files are required for deployment:

service.py configures the service interface: Expected/allowed inputs and data types, their transformation for prediction, and finally the prediction output (simple text indicating the predicited type of kitchenware).

bentofile.yaml adds metainformation about the project to the service interface as well as which non-standard packages are being used. In this case: numpy and tensorflow (which includes keras).

If your machine does not have a GPU, you can replicate the BentoML step by running build_bento_from_saved_cnn_2.1.py which skips the GPU-based training and builds a bento directly from the previously saved keras model (which is part of the repository).

# Dependency and environment management

A list of the required dependencies and environment (python version, required packages, requirements.txt, basic dockerfile etc.) was created automatically when exporting using BentoML and saved to the bento.

The following steps need to be done in order to prepare the model for deployment as a local service:

In the bash terminal, open the project folder /mlbookcamp-midterm

Build a BentoML "bento" from the above prepared file by entering "bentoml build". This will create a bento file which includes all necessary information for dependency and environment management.

The resulting bento file from step 2) is identified by a service name (as defined in service.py) + service tag (assigned during and displayed after the build process).

Test the bento locally by entering "bentoml serve". This command starts a so-called BentoServer on your current machine so the model service can easily be accessed in the browser (under http://0.0.0.0:3000 by default).

# Containerization

After making sure locally that the bento file results in a service that works as it should, you can easily containerize the service using BentoML once again.

In order to do this, you need to enter "bentoml containerize" along with the name and tag of the bento you want to containerize. An easy shortcut to containerize the most recently created bento (which is likely to be what you want to do) is to use the tag "latest". The full command to containerize the latest version of bento "insurance_response_classifier" would therefore be "bentoml containerize insurance_response_classifier:latest".

This will yield a docker image which can be deployed anywhere. The image we will continue working with is called

kitchenware_classifier_service:latest / with tag sngi2wt7dky2cd5x

For local testing (on an instance where docker is installed!), simply run

docker run -it --rm -p 3000:3000 kitchenware_classifier_service:sngi2wt7dky2cd5x serve --production

This will start a web-based swagger interface to which you can upload and classify images of kitchenware.

# Cloud deployment

### asd

