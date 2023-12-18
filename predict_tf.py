import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.models import load_model

model = tf.keras.models.load_model('veggiehealth_model_tf_origin.h5')

test_dir = "..\\ML-Projects\\Vegetable Images\\test\\"
train_dir = "..\\ML-Projects\\Vegetable Images\\train\\"
validation_dir = "..\\ML-Projects\\Vegetable Images\\validation\\"

test = 'D:\\Games\\'

validation_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224,224),
    batch_size=64,
    class_mode='categorical')
test_datagen_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(224,224),
    batch_size=64,
    class_mode='categorical')

# loss, acc = model.evaluate(validation_generator, verbose=1)
# print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
#
# print("Done!")

category = dict(enumerate(os.listdir(test_dir)))

def predict_image(filename, model):
    img_ = image.load_img(filename, target_size=(224, 224))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.

    prediction = model.predict(img_processed)
    index = np.argmax(prediction)

    plt.title("Prediction - {}".format(category[index]))
    plt.imshow(img_array)
    plt.show()

predict_image(os.path.join(test,'WhatsApp Image 2023-12-18 at 22.21.02_52022520.jpg'),model)

def predict_dir(filedir, model):
    cols = 3
    pos = 0
    images = []
    total_images = len(os.listdir(filedir))
    rows = total_images // cols + 1

    true = filedir.split('/')[-1]

    for i in sorted(os.listdir(filedir)):
        images.append(os.path.join(filedir, i))

    for subplot, imggg in enumerate(images):
        img_ = image.load_img(imggg, target_size=(224, 224))
        img_array = image.img_to_array(img_)
        img_processed = np.expand_dims(img_array, axis=0)
        img_processed /= 255.
        prediction = model.predict(img_processed)
        index = np.argmax(prediction)

        pred = category.get(index)
        if pred == true:
            pos += 1

    acc = pos / total_images
    print("Accuracy for {orignal}: {:.2f} ({pos}/{total})".format(acc, pos=pos, total=total_images, orignal=true))

# for i in os.listdir(test_dir):
#     print(i)
#     predict_dir(os.path.join(test_dir,i),model)


# def labels_confusion_matrix(validation_folder):
#     folder_path = validation_folder
#
#     mapping = {}
#     for i, j in enumerate(sorted(os.listdir(folder_path))):
#         mapping[j] = i
#
#     files = []
#     real = []
#     predicted = []
#
#     for i in os.listdir(folder_path):
#
#         true = os.path.join(folder_path, i)
#         true = true.split('/')[-1]
#         true = mapping[true]
#
#         for j in os.listdir(os.path.join(folder_path, i)):
#             img_ = image.load_img(os.path.join(folder_path, i, j), target_size=(224, 224))
#             img_array = image.img_to_array(img_)
#             img_processed = np.expand_dims(img_array, axis=0)
#             img_processed /= 255.
#             prediction = model.predict(img_processed)
#             index = np.argmax(prediction)
#
#             predicted.append(index)
#             real.append(true)
#
#     return (real, predicted)


