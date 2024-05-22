!kaggle datasets download -d salader/dogs-vs-cats


import zipfile
zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout

# Generators divide data into batches
train_ds = keras.utils.image_dataset_from_directory(
    directory='/content/train',
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(256, 256)
)

test_ds = keras.utils.image_dataset_from_directory(
    directory='/content/test',
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(256, 256)
)

# Normalization for scaling
def normalize(image, label):
    image = tf.cast(image / 255, tf.float32)
    return image, label

train_ds = train_ds.map(normalize)
test_ds = test_ds.map(normalize)

# Create a CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))



# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Train the model
model.fit(train_ds, epochs=1)


# Evaluate the model
test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)


# Predict on a single image
import cv2
import numpy as np



test_img = cv2.imread('/content/dog.jpeg')
test_img = cv2.resize(test_img, (256, 256))
test_img = test_img.reshape(1, 256, 256, 3)

prediction = model.predict(test_img)


if prediction[0][0] > 0.5:
    print("It's a dog!")
else:
    print("It's a cat!")