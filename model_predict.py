import tensorflow as tf
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
model = tf.keras.models.load_model('veggiehealth_model_2.h5')
import cv2
# new_model.summary()
#
# test_images = 'D:\\Bangkit\\Capstone\\ML-Projects\\Vegetable Images\\test\\'
# test_labels = 'D:\\Bangkit\\Capstone\\ML-Projects\\Vegetable Images\\test\\'
#
# # def load_data(data_type):
# #     images = []
# #     labels = []
# #     for idx, category in enumerate(categories):
# #         category_path = os.path.join(data_path, data_type, category)
# #         for img_name in os.listdir(category_path):
# #             img = cv2.imread(os.path.join(category_path, img_name))
# #             img = cv2.resize(img, (128, 128))
# #             images.append(img)
# #             labels.append(idx)
# #     return np.array(images), np.array(labels)
# #
# # X_train, y_train = load_data("training")
# # X_val, y_val = load_data("validation")
test_dir = "..\\ML-Projects\\Vegetable Images\\test\\"
test_datagen = ImageDataGenerator(rescale=1 / 255)
test_generator = test_datagen.flow_from_directory(test_dir,
                                                    batch_size=32,
                                                    target_size=(150,150),
                                                    class_mode='categorical',
                                                    shuffle= False
                                                    )
model.evaluation(test_generator)
# categories = sorted(os.listdir(test_dir))
# prediction = np.expand_dims(test_generator, axis=0)
# loss, acc = new_model.predict(categories[np.argmax(prediction)], verbose=2)
#
#
# file_paths = [os.path.join(test_dir, file) for file in os.listdir(test_dir) if
#               os.path.isfile(os.path.join(test_dir, file))]
#
# for i, file_path in enumerate(file_paths, 1):
#     test_image_path = file_path
#     veg_category, size = new_model.predict(categories[np.argmax(np.expand_dims(test_image_path, axis=0))], verbose=2)
#
#     print(f"Item #{i}: {os.path.basename(file_path)}")
#     print(f"Food Category: {veg_category}")
#     print(f"Estimated Size: {size}")
#     print()
# # print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
#
# # test_folder_path = 'D:\Dunia Perkuliahan\Semester 5\Bangkit\Food_Image_Recognition\Test Model'
# #
# # # List of file paths in the test folder
# # file_paths = [os.path.join(test_dir, file) for file in os.listdir(test_dir) if
# #               os.path.isfile(os.path.join(test_dir, file))]
# #
# # # Test the model on each image in the folder
# # for i, file_path in enumerate(file_paths, 1):
# #     test_image_path = file_path
# #     food_category = new_model.predict(test_generator, verbose=2)
# #
# #     print(f"Item #{i}: {os.path.basename(file_path)}")
# #     print(f"Food Category: {food_category}")
# #     print()
# # # Importing necessary libraries
# # import os
# # import cv2
# # import numpy as np
# # from tensorflow.keras.models import load_model
# #
# # # Load the trained model
# # model = load_model('model_try.h5')
# #
# # # Define food categories
# # data_path = "D:\Dunia Perkuliahan\Semester 5\Bangkit\Food_Image_Recognition\Dataset"
# # categories = sorted(os.listdir(os.path.join(data_path, "training")))  # Replace with your actual categories
#
# # def predict_food_size(image_path):
# #     img = cv2.imread(image_path)
# #     img = cv2.resize(img, (128, 128))
# #     prediction = model.predict(np.expand_dims(img, axis=0))
# #     food_category = categories[np.argmax(prediction)]
# #
# #     # Estimate the size of the dish
# #     area = np.sum(img > 50)  # Thresholding to exclude background pixels
# #     if area < 10000:
# #         size = 'small'
# #     elif area < 30000:
# #         size = 'medium'
# #     else:
# #         size = 'large'
# #
# #     return food_category, size
# #
# # # Path to the folder containing test images
# # test_folder_path = 'D:\Dunia Perkuliahan\Semester 5\Bangkit\Food_Image_Recognition\Test Model'
# #
# # # List of file paths in the test folder
# # file_paths = [os.path.join(test_folder_path, file) for file in os.listdir(test_folder_path) if
# #               os.path.isfile(os.path.join(test_folder_path, file))]
# #
# # # Test the model on each image in the folder
# # for i, file_path in enumerate(file_paths, 1):
# #     test_image_path = file_path
# #     food_category, size = predict_food_size(test_image_path)
# #
# #     print(f"Item #{i}: {os.path.basename(file_path)}")
# #     print(f"Food Category: {food_category}")
# #     print(f"Estimated Size: {size}")
# #     print()

# # Importing necessary libraries
# import os
# import numpy as np
#
#
# # Define food categories
# data_path = "D:\\Bangkit\\Capstone\\ML-Projects\\Vegetable Images\\"
# categories = sorted(os.listdir(os.path.join(data_path, "test")))  # Replace with your actual categories
#
# def predict_food_size(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (150, 150))
#
#
#     prediction = model.predict(np.expand_dims(img, axis=0))
#     food_category = categories[np.argmax(prediction)]

    # # Estimate the size of the dish
    # area = np.sum(img > 50)  # Thresholding to exclude background pixels
    # if area < 10000:
    #     size = 'small'
    # elif area < 30000:
    #     size = 'medium'
    # else:
    #     size = 'large'

#     return food_category
#
# # Path to the folder containing test images
# test_folder_path = "D:\\Bangkit\\Capstone\\ML-Projects\\Vegetable Images\\test\\"
#
# # List of file paths in the test folder
# # file_paths = [os.path.join(test_folder_path, file) for file in os.listdir(test_folder_path) if
# #               os.path.isfile(os.path.join(test_folder_path, file))]
# #
# # # Test the model on each image in the folder
# # for i, file_path in enumerate(file_paths, 1):
# #     test_image_path = file_path
# #     food_category = predict_food_size(test_image_path)
# #
# #     print(f"Item #{i}: {os.path.basename(file_path)}")
# #     print(f"Food Category: {food_category}")
# #     print()
#
# food_category = predict_food_size("0001.jpg")
# print(f"Food Category: {food_category}")