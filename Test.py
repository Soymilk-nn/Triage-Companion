import tensorflow as tf
import numpy as np


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


train_images, test_images = train_images / 255.0, test_images / 255.0


history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))


test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
# Consider a new image (new_image) preprocessed to be in the shape (28, 28) and pixel values normalized between 0 and 1.
# For the sake of illustration, let's use an image from the test set.

new_image = test_images[112]

# Add an extra dimension to the image tensor, because the model expects a batch of images as input
new_image = np.expand_dims(new_image, axis=0)

# Use the model to make a prediction
predictions = model.predict(new_image)

# Get the predicted label
predicted_label = np.argmax(predictions)

print('Predicted label:', predicted_label)
#print('\nTest accuracy:', test_acc)

#0: T-shirt/top
#1: Trouser
#2: Pullover
#3: Dress
#4: Coat
#5: Sandal
#6: Shirt
#7: Sneaker
#8: Bag
#9: Ankle boot