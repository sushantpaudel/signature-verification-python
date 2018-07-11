from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout


def create_model():
    model_custom_input = Sequential()
    model_custom_input.add(Conv2D(32, (3, 3), input_shape=(100, 100, 4)))
    model_custom_input.add(Activation('relu'))
    model_custom_input.add(MaxPooling2D(pool_size=(2, 2)))

    model_custom_input.add(Conv2D(32, (3, 3)))
    model_custom_input.add(Activation('relu'))
    model_custom_input.add(MaxPooling2D(pool_size=(2, 2)))

    model_custom_input.add(Conv2D(64, (3, 3)))
    model_custom_input.add(Activation('relu'))
    model_custom_input.add(MaxPooling2D(pool_size=(2, 2)))

    model_custom_input.add(Flatten())
    model_custom_input.add(Dense(64))
    model_custom_input.add(Activation('relu'))
    model_custom_input.add(Dropout(0.5))
    model_custom_input.add(Dense(1))
    model_custom_input.add(Activation('sigmoid'))

    return model_custom_input


def add_weights(model_custom_input):
    model_custom_input.load_weights("model.h5")

    model_custom_input.compile(loss='binary_crossentropy',
                               optimizer='rmsprop',
                               metrics=['accuracy'])
    return model_custom_input
