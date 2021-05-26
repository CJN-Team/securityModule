from sklearn.metrics import classification_report, confusion_matrix
import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, MaxPool2D, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


train_dir = "imagenes/traine"
test_dir = "imagenes/teste"
img_size = 48
epochs = 60
batch_size = 64

train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rescale=1./255,
    validation_split=0.2
)
validation_datagen = ImageDataGenerator(rescale=1./255,
                                        validation_split=0.2)

train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                    target_size=(
                                                        img_size, img_size),
                                                    batch_size=64,
                                                    color_mode="grayscale",
                                                    class_mode="categorical",
                                                    subset="training"
                                                    )
validation_generator = validation_datagen.flow_from_directory(directory=test_dir,
                                                              target_size=(
                                                                  img_size, img_size),
                                                              batch_size=64,
                                                              color_mode="grayscale",
                                                              class_mode="categorical",
                                                              subset="validation"
                                                              )


model = tf.keras.models.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same',
          activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
          kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
          kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(x=train_generator, epochs=epochs,
                    validation_data=validation_generator)

lossEvaluate, accEvaluate = model.evaluate(validation_generator,
                                           batch_size=batch_size)



y_pred = model.predict(train_generator)
y_pred = np.argmax(y_pred, axis=1)
class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}

cm_train = confusion_matrix(train_generator.classes, y_pred)
print('Confusion Matrix')
print(cm_train)
print('Classification Report')
target_names = list(class_labels.values())
print(classification_report(train_generator.classes,
      y_pred, target_names=target_names))

plt.figure(figsize=(8, 8))
plt.imshow(cm_train, interpolation='nearest')
plt.colorbar()
tick_mark = np.arange(len(target_names))
_ = plt.xticks(tick_mark, target_names, rotation=90)
_ = plt.yticks(tick_mark, target_names)
plt.savefig('confusionMatrix.jpg')


fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")
print("Saved model to disk")


data = {}
data['modelParams'] = []
data['modelParams'].append({
    'accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'],
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss'],
    'lossEvaluate': lossEvaluate,
    'accEvaluate': accEvaluate
})


with open('modelParams.txt', 'w') as outfile:
    json.dump(data, outfile)
