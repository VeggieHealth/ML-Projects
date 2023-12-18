import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from PIL import ImageEnhance
from skimage.io import imread
import matplotlib.pyplot as plt

import os, random, pathlib, warnings, itertools, math
warnings.filterwarnings("ignore")

import tensorflow as tf
import keras.backend as K
from sklearn.metrics import confusion_matrix

from keras import models
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dense, Dropout

K.clear_session()

train_dir = "..\\ML-Projects\\Vegetable Images\\train\\"
validation_dir = "..\\ML-Projects\\Vegetable Images\\validation\\"
test_dir = "..\\ML-Projects\\Vegetable Images\\test\\"

categories = len(os.listdir(train_dir))

def count_files(rootdir):
    '''counts the number of files in each subfolder in a directory'''
    for path in pathlib.Path(rootdir).iterdir():
        if path.is_dir():
            print("There are " + str(len([name for name in os.listdir(path) \
                                          if os.path.isfile(os.path.join(path, name))])) + " files in " + \
                  str(path.name))


# count_files(os.path.join(validation_dir))

image_folder ='water_spinach'  # The vegetable you want to display
number_of_images = 2  # Number of images to display

def preprocess():
    j = 1
    for i in range(number_of_images):
        folder = os.path.join(validation_dir, image_folder)
        a = random.choice(os.listdir(folder))

        image = Image.open(os.path.join(folder, a))
        image_duplicate = image.copy()
        plt.figure(figsize=(10, 10))

        plt.subplot(number_of_images, 2, j)
        plt.title(label='Orignal', size=17, pad='7.0', loc="center", fontstyle='italic')
        plt.imshow(image)
        plt.show()
        j += 1

        image1 = ImageEnhance.Color(image_duplicate).enhance(1.35)
        image1 = ImageEnhance.Contrast(image1).enhance(1.45)
        image1 = ImageEnhance.Sharpness(image1).enhance(2.5)

        plt.subplot(number_of_images, 2, j)
        plt.title(label='Processed', size=17, pad='7.0', loc="center", fontstyle='italic')
        plt.imshow(image1)
        plt.show()
        j += 1
# preprocess()

train_datagen = image.ImageDataGenerator(rescale = 1./255,
                                         height_shift_range=0.2,
                                         width_shift_range=0.2,
                                         rotation_range=0.5,
                                         shear_range=0.2,
                                         fill_mode='nearest',
                                         horizontal_flip=True,
                                         vertical_flip=True, validation_split=0.1
                                         )
training_set = train_datagen.flow_from_directory(
    train_dir,
    target_size = (224, 224),
    batch_size = 64,
    class_mode = 'categorical', subset='training')

validation_set = train_datagen.flow_from_directory(
    train_dir,
    target_size = (224, 224),
    batch_size = 64,
    class_mode = 'categorical', subset='validation')

IMAGE_SIZE = [224, 224]

inception = InceptionV3(input_shape= (224,224,3), weights='imagenet', include_top=False)

for layer in inception.layers:
    layer.trainable = False

x = inception.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)

prediction = Dense(categories, activation='softmax')(x)

model = Model(inputs=inception.input, outputs=prediction)

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if(logs.get('accuracy')>0.95 and logs.get('val_accuracy')>0.93):
            print("\nAcc dan Val_acc sudah mencapai lebih dari 95%, berhenti training !!!")
            self.model.stop_training=True

r = model.fit_generator(
  training_set,
  validation_data=validation_set,
  epochs=20,
  steps_per_epoch=len(training_set), callbacks= myCallback()
)

model.save("veggiehealth_model_tf_origin.h5")
