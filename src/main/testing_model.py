import matplotlib.image as mpimg
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import load_model

check_path = "data/check/"


def create_model_input():
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

    model_custom_input.load_weights("model.h5")

    model_custom_input.compile(loss='binary_crossentropy',
                               optimizer='rmsprop',
                               metrics=['accuracy'])
    return model_custom_input


def create_model():
    model_inside = load_model("model_saved.h5")
    model_inside.compile(loss='binary_crossentropy',
                         optimizer='rmsprop',
                         metrics=['accuracy'])
    return


model = create_model_input()


def check_signature(path):
    images = [mpimg.imread(path)]
    x_test = np.array(images)
    print(model.predict(x_test)[0][0])
    return model.predict(x_test)[0][0]


def verify_image(test_image):
    images = [test_image]
    x_test = np.array(images)
    return model.predict(x_test)[0][0]


def output(value):
    if value == 1.0:
        return "This signature is genuine!!"
    else:
        return "This signature is forged!!"


# pre_process = PreProcessing()
# image = pre_process.pre_process(check_path + "forged1.png")
# verify_image(image)


print("Forged: ", output(check_signature(check_path + "forged1.png")))
print("Forged: ", output(check_signature(check_path + "forged2.png")))
print("Forged: ", output(check_signature(check_path + "forged3.png")))
print("Forged: ", output(check_signature(check_path + "forged4.png")))
print("Forged: ", output(check_signature(check_path + "forged5.png")))
print("Forged: ", output(check_signature(check_path + "forged6.png")))
print("Forged: ", output(check_signature(check_path + "forged7.png")))
print("Forged: ", output(check_signature(check_path + "forged8.png")))
print("____________________________________________________________")
print("")
print("Genuine: ", output(check_signature(check_path + "genuine1.png")))
print("Genuine: ", output(check_signature(check_path + "genuine2.png")))
print("Genuine: ", output(check_signature(check_path + "genuine3.png")))
print("Genuine: ", output(check_signature(check_path + "genuine4.png")))
print("Genuine: ", output(check_signature(check_path + "genuine5.png")))
print("Genuine: ", output(check_signature(check_path + "genuine6.png")))
print("Genuine: ", output(check_signature(check_path + "genuine7.png")))
