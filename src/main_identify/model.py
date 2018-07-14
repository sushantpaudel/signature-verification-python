import csv

import matplotlib.image as mpimg
import numpy as np
from keras import backend as K
from keras.engine.saving import load_model
from keras.layers import Conv2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from sklearn.utils import shuffle

## 1. Prepare and create generator

# Save filepaths of images to `samples` to load into generator
samples = []
train_samples = []
validation_samples = []
main_dir = "data/train"
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


train_samples = add_to_samples('dataset_train_small.csv', train_samples)
validation_samples = add_to_samples('dataset_validate_small.csv', validation_samples)


# Split samples into training and validation sets to reduce overfitting
# train_samples, validation_samples = train_test_split(samples, test_size=0.1)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            keys = []
            for batch_sample in batch_samples:
                key = batch_sample[1]
                name = main_dir + "/" + key + "/" + format(batch_sample[0])
                center_image = mpimg.imread(name)
                images.append(center_image)
                keys.append(key)
            x_train = np.array(images)
            y_train = np.array(keys)
            yield shuffle(x_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

print("Training Samples: ", len(train_samples))
print("Validation Samples: ", len(validation_samples))


def create_model():
    model_inside = Sequential()
    model_inside.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model_inside.add(Activation('relu'))
    model_inside.add(MaxPooling2D(pool_size=(2, 2)))

    model_inside.add(Conv2D(32, (3, 3)))
    model_inside.add(Activation('relu'))
    model_inside.add(MaxPooling2D(pool_size=(2, 2)))

    model_inside.add(Conv2D(64, (3, 3)))
    model_inside.add(Activation('relu'))
    model_inside.add(MaxPooling2D(pool_size=(2, 2)))

    model_inside.add(Flatten())
    model_inside.add(Dense(64))
    model_inside.add(Activation('relu'))
    model_inside.add(Dropout(0.5))
    model_inside.add(Dense(32))
    model_inside.add(Activation('relu'))
    model_inside.add(Dense(54))
    model_inside.add(Activation('sigmoid'))

    model_inside.compile(loss='binary_crossentropy',
                         optimizer='rmsprop',
                         metrics=['accuracy'])
    print("Model summary:\n", model_inside.summary())
    return model_inside


def create_model_saved():
    model_inside_saved = load_model("model_with_weights.h5")
    return model_inside_saved


model = create_model_saved()

## 4. Train model
batch_size = 32
nb_epoch = 200

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
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save("model_with_weights.h5")
model.save_weights("model_weights_only.h5")
print("Saved model to disk")
