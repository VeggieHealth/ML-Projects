import tensorflow as tf
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import cv2
from matplotlib import pyplot as plt

# from predict_evaluation import subdirectories

model = tf.keras.models.load_model('veggiehealth_model_grayscale_2.h5')


data_path = r'D:\\Bangkit\\Capstone\\ML-Projects\\Vegetable Images\\test\\'
# categories = sorted(os.listdir(data_path))  # Replace with your actual categories


data_list=[]
# dir=r'c:\test'
def predict_veg(image):
    test_list=os.listdir(data_path) # create a list of the files in the directory
    for f in test_list:  # iterate through the files
        # if os.listdir() == 'Brinjal':
        path = os.path.join(data_path, f)
        fpath=os.listdir(os.path.join (data_path, f)) # create path to the image file
        for i in fpath:
            if i == image:
                imgPath = os.path.join(path,img)
                img=cv2.imread(imgPath) # read image using cv2
                cv2.imshow('Original', img)
                cv2.waitKey(0)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imshow('gray scale', img)
                cv2.waitKey(0)
                img=cv2.resize(img, (224,224)) # resize the image
                data_list.append(img)  # append processed image to the list
            else:
                continue
#         # else:
#         #     continue
        data=np.array(data_list)/255# convert to an np array and rescale images

        predictions = model.predict(data, verbose=0)
        # print(predictions)
        predictions = np.argmax(predictions)


#
        trials = len(test_list)
        print(trials)
        # labels = dict(enumerate(test_list))
        # result = labels[predictions]
        # return result
#
        for i in range(0, 18):
            if test_list[i] == predictions:
                print(test_list[12])
                # return (test_list[i], predictions)  # print file name and class prediction

#
print(predict_veg('IMG_20231214_184306_527.jpg'))





