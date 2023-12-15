import tensorflow as tf

new_model = tf.keras.models.load_model('veggiehealth_model_1.h5')
new_model.summary()

test_images = 'D:\\Bangkit\\Capstone\\ML-Projects\\Vegetable Images\\test\\'
test_labels = 'D:\\Bangkit\\Capstone\\ML-Projects\\Vegetable Images\\test\\'



loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))