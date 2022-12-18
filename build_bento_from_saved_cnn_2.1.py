# load required packages
# import numpy as np
# import pandas as pd
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions

from keras.callbacks import EarlyStopping

import bentoml

model = keras.models.load_model('cnn_2.1')

# + saving final model to file using pickle or bentoml
saved_model = bentoml.keras.save_model("kitchenware_classifier",
                                       model)

print(f"Model saved: {saved_model}")



