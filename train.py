# load required packages
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions

from keras.callbacks import EarlyStopping

import bentoml

# load and prep data
df_train_full = pd.read_csv('data/train.csv', dtype = {'Id': str})
df_train_full['filename'] = 'data/images/' + df_train_full['Id'] + '.jpg'

val_cutoff = int(len(df_train_full) * 0.8) # define where to split data into train and validation
df_train = df_train_full[:val_cutoff] # define df_train using cutoff
df_val = df_train_full[val_cutoff:] # define df_val using cutoff

# train final model
train_xc_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_xc_generator = train_xc_datagen.flow_from_dataframe(
    df_train,
    x_col='filename',
    y_col='label',
    target_size=(299, 299),
    batch_size=32
)

val_xc_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_xc_generator = val_xc_datagen.flow_from_dataframe(
    df_val,
    x_col='filename',
    y_col='label',
    target_size=(299, 299),
    batch_size=32
)

X, y = next(train_xc_generator)

base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(299, 299, 3)
)

base_model.trainable = False

inputs = keras.Input(shape=(299, 299, 3))
base = base_model(inputs, training=False)
vectors = keras.layers.GlobalAveragePooling2D()(base)
outputs = keras.layers.Dense(6)(vectors)
model = keras.Model(inputs, outputs)

learning_rate = 0.001
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss = keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer = optimizer,
              loss = loss, 
              metrics=['accuracy'])

# simple early stopping as soon as we exceed 0.97 validation accuracy
es = EarlyStopping(monitor='val_accuracy', mode='min', baseline=0.970)

history = model.fit(train_xc_generator,
                    epochs = 10, 
                    validation_data = val_xc_generator,
                    callbacks=[es])


# + saving final model to file using pickle or bentoml
saved_model = bentoml.keras.save_model("kitchenware_classifier",
                                       model)

print(f"Model saved: {saved_model}")



