# import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
from keras.optimizers import Adam

train_dir = "..\\ML-Projects\\Vegetable Images\\train\\"
validation_dir = "..\\ML-Projects\\Vegetable Images\\validation\\"
test_dir = "..\\ML-Projects\\Vegetable Images\\test\\"

categories = os.listdir(train_dir)

train_datagen = ImageDataGenerator(rescale=1 / 255,
                                   height_shift_range=0.2,
                                   width_shift_range=0.2,
                                   rotation_range=10,
                                   shear_range=0.2,
                                   fill_mode='nearest',
                                   horizontal_flip=True,
                                   vertical_flip=True,validation_split=0.1)

validation_datagen = ImageDataGenerator(rescale=1. / 255)
# test_datagen = ImageDataGenerator(rescale=1 / 255)

train_batch = 64
validation_batch = 32

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=train_batch,
                                                    target_size=(224, 224),
                                                    class_mode='categorical',
                                                    shuffle=True, seed=42,subset='training'
                                                    )

validation_generator = train_datagen.flow_from_directory(train_dir,
                                                              batch_size=validation_batch,
                                                              target_size=(224, 224),
                                                              class_mode='categorical',
                                                              shuffle=True, seed=42,subset='validation'
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

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(20, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2, padding='valid'),
    tf.keras.layers.Conv2D(48, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(50, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(categories), activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(train_generator, epochs=30, validation_data=validation_generator, steps_per_epoch= 13500 /train_batch, callbacks= myCallback())

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

model.save("veggiehealth_model_origin.h5")
