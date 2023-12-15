import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Load the saved model
saved_model_path = "veggiehealth_model_2.h5"
loaded_model = tf.keras.models.load_model(saved_model_path)
train_dir = "..\\ML-Projects\\Vegetable Images\\train\\"


# Path to the image you want to make predictions on
image_path = 'D:\\Bangkit\\Capstone\\ML-Projects\\Vegetable Images\\test\\' # Replace with the actual path to your image

# Load and preprocess the image
img = image.load_img(image_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Rescale to match the training data

# Make predictions
predictions = loaded_model.predict(img_array)

# Get the predicted class index
predicted_class_index = np.argmax(predictions)

train_datagen = ImageDataGenerator(rescale=1 / 255,
                                   height_shift_range=0.2,
                                   width_shift_range=0.2,
                                   rotation_range=10,
                                   shear_range=0.2,
                                   fill_mode='nearest',
                                   horizontal_flip=True,
                                   vertical_flip=True)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=64,
                                                    target_size=(150, 150),
                                                    class_mode='categorical',
                                                    shuffle=True
                                                    )

# Map the index to the class label
categories = train_generator.class_indices
predicted_class_label = [k for k, v in categories.items() if v == predicted_class_index][0]

# Print the predicted class label
print("Predicted class: {}".format(predicted_class_label))
