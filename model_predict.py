import tensorflow as tf
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import cv2
from matplotlib import pyplot as plt

from predict_evaluation import subdirectories

model = tf.keras.models.load_model('veggiehealth_model_5.h5')


data_path = r'D:\\Bangkit\\Capstone\\ML-Projects\\Vegetable Images\\test\\'
# categories = sorted(os.listdir(data_path))  # Replace with your actual categories


data_list=[]
# dir=r'c:\test'
def predict_veg():
    test_list=os.listdir(data_path) # create a list of the files in the directory
    for f in test_list:  # iterate through the files
        # if f == 'Brinjal':
        path = os.path.join(data_path, f)
        fpath=os.listdir(os.path.join (data_path, f)) # create path to the image file
        for i in fpath:
            imgPath = os.path.join(path,i)
            img=cv2.imread(imgPath) # read image using cv2
            img=cv2.resize(img, (224,224)) # resize the image
            data_list.append(img)  # append processed image to the list
        # else:
        #     continue
        data=np.array(data_list)/255# convert to an np array and rescale images
        predictions = model.predict(data, verbose=0)
        # print(predictions)
        # predictions = np.argmax(predictions)
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        y_pred_class = np.argmax(predictions, axis=1)
        y_true_class = test_list
        cm = confusion_matrix(y_true_class, y_pred_class, normalize='true')

        plt.figure(figsize=(12, 8))
        plt.title('Confusion matrix of the classifier')
        sns.heatmap(cm,
                    annot=True,
                    fmt=".2f",
                    cmap=sns.color_palette("Blues", 12),
                    yticklabels=subdirectories,
                    xticklabels=subdirectories)

        plt.xticks(rotation=45)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        # trials = len(predictions)
        labels = dict(enumerate(test_list))
        # result = labels[predictions]
        # return result

        # for i in range(0, 13):
        #     if test_list[i] == predictions:
        #         print(test_list[i], predictions)  # print file name and class prediction

print(predict_veg())





