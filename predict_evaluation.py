import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

model = tf.keras.models.load_model('veggiehealth_model_5.h5')

val_dir = "..\\ML-Projects\\Vegetable Images\\validation\\"
test_dir ="..\\ML-Projects\\Vegetable Images\\test\\"

subdirectories = [f.name for f in os.scandir(val_dir) if f.is_dir()]

print("==> Vegetable names:")
for subdir in subdirectories:
    print(subdir)

print(f"\n==> Number of classes: {len(subdirectories)}")

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

val_data = val_datagen.flow_from_directory(val_dir,
                                           target_size=(224,224),
                                           batch_size=32,
                                           class_mode='categorical',
                                           seed=42)
test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=(224,224),
                                             batch_size=32,
                                             class_mode='categorical',
                                             shuffle=False)


# loss, acc = model.evaluate(val_data, verbose=2)
# print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
labels = dict(enumerate(os.listdir(test_dir)))
print(labels)
predictions = model.predict(test_data)
for i in predictions:
    result = labels[np.argmax(i)]
    print(result)

import seaborn as sns
from sklearn.metrics import confusion_matrix

y_pred_class = np.argmax(predictions, axis=1)
y_true_class = test_data.classes
cm = confusion_matrix(y_true_class, y_pred_class, normalize='true')

plt.figure(figsize=(12,8))
plt.title('Confusion matrix of the classifier')
sns.heatmap(cm,
            annot=True,
            fmt=".2f",
            cmap=sns.color_palette("Blues",12),
            yticklabels=subdirectories,
            xticklabels=subdirectories)

plt.xticks(rotation=45)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()