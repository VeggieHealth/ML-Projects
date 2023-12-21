import matplotlib.pyplot as plt
import os
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dense, Dropout

K.clear_session()

train_dir = "..\\ML-Projects\\Vegetable Images\\train\\"
validation_dir = "..\\ML-Projects\\Vegetable Images\\validation\\"
test_dir = "..\\ML-Projects\\Vegetable Images\\test\\"

categories = len(os.listdir(train_dir))

train_datagen = image.ImageDataGenerator(rescale=1. / 255,
                                         rotation_range=10,
                                         width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         fill_mode='nearest',
                                         horizontal_flip=True,
                                         validation_split=0.1
                                         )
# dir_It = train_datagen.flow_from_directory(
#     train_dir,
#     batch_size=1,
#     save_to_dir="D:\\Games\\",
#     save_prefix="",
#     save_format='png',
# )
#
# for _ in range(5):
#     img, label = dir_It.next()
#     print(img.shape)
#     plt.imshow(img[0])
#     plt.show()

training_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical', subset='training')

validation_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical', subset='validation')

IMAGE_SIZE = [224, 224]

pre_trained = InceptionV3(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

for layer in pre_trained.layers:
    layer.trainable = False

x = pre_trained.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)



prediction = Dense(categories, activation='softmax')(x)

model = Model(inputs=pre_trained.input, outputs=prediction)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if (logs.get('accuracy') > 0.95 and logs.get('val_accuracy') > 0.93):
            print("\nAcc dan Val_acc sudah mencapai lebih dari 95%, berhenti training !!!")
            self.model.stop_training = True


history = model.fit_generator(
    training_set,
    validation_data=validation_set,
    epochs=20,
    steps_per_epoch=len(training_set), callbacks=myCallback()
)
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

model.save("veggiehealth_model_tf_origin_2.h5")
