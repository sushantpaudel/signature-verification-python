import csv

import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

## 1. Prepare and create generator

# Save filepaths of images to `samples` to load into generator
samples = []
main_dir = "data/"
genuine_directory = "data/genuine"
forge_directory = "data/forged"
height = 100
width = 100
# The number of channels in our input images (1  for grayscale single channel images, 3  for standard RGB images)
depth = 4
input_shape = (height, width, depth)
# if we are using "channels first", update the input shape
if K.image_data_format() == "channels_first":
    input_shape = (depth, height, width)


def add_to_samples(csv_filepath, samples):
    with open(csv_filepath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples


samples = add_to_samples('dataset_new.csv', samples)

# Remove header - No header in our dataset
# samples = samples[1:]

print("Samples: ", len(samples))

# Split samples into training and validation sets to reduce overfitting
train_samples, validation_samples = train_test_split(samples, test_size=0.1)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            labels = []
            for batch_sample in batch_samples:
                label = batch_sample[1]
                name = format(batch_sample[0])
                if label == "1":
                    name = genuine_directory + "/" + name
                else:
                    name = forge_directory + "/" + name
                center_image = mpimg.imread(name)
                center_image = tf.image.resize_images(center_image, [height, width])
                images.append(center_image)
                labels.append(label)

            x_train = np.array(images)
            y_train = np.array(labels)
            print(x_train)
            yield shuffle(x_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

print("Training Samples: ", len(train_samples))
print("Validation Samples: ", len(validation_samples))

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print("Model summary:\n", model.summary())

## 4. Train model
batch_size = 32
nb_epoch = 50

# Save model weights after each epoch
# checkpointer = ModelCheckpoint(filepath="./tmp/v2-weights.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1, save_best_only=False)

# Train model using generator

print(train_generator)
print(validation_generator)

model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=nb_epoch)

model_json = model.to_json()
with open("model_new_json.json", "w") as json_file:
    json_file.write(model_json)

model.save("model_new_saved.h5")
model.save_weights("model_new_weights.h5")
print("Saved model to disk")
