# import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(13, activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

train_dir = "D:\\Bangkit\\Capstone\\ML-Projects\\Vegetable Images\\train\\"
validation_dir = "D:\\Bangkit\\Capstone\\ML-Projects\\Vegetable Images\\validation\\"
test_dir = "D:\\Bangkit\\Capstone\\ML-Projects\\Vegetable Images\\test\\"

train_datagen = ImageDataGenerator(rescale=1 / 255,
                                   height_shift_range=0.2,
                                   width_shift_range=0.2,
                                   rotation_range=0.2,
                                   shear_range=0.2,
                                   fill_mode='nearest',
                                   horizontal_flip=True,
                                   vertical_flip=True)

validation_datagen = ImageDataGenerator(rescale=1 / 255)
test_datagen = ImageDataGenerator(rescale=1 / 255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=32,
                                                    target_size=(150, 150),
                                                    class_mode='categorical',
                                                    shuffle=True
                                                    )

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              batch_size=32,
                                                              target_size=(150, 150),
                                                              class_mode='categorical',
                                                              shuffle=False
                                                              )

# test_generator = test_datagen.flow_from_directory(validation_dir,
#                                                     batch_size=32,
#                                                     target_size=(150,150),
#                                                     class_mode='categorial',
#                                                     shuffle= False
#                                                     )

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if(logs.get('accuracy')>0.95 and logs.get('val_accuracy')>0.93):
            print("\nAcc dan Val_acc sudah mencapai lebih dari 95%, berhenti training !!!")
            self.model.stop_training=True

history = model.fit(train_generator, epochs=50, validation_data=validation_generator, validation_steps=1, callbacks= myCallback())

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

model.save("veggiehealth_model_1.h5")
