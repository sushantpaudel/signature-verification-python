import os
from random import shuffle

genuine_directory = "data/genuine"
forge_directory = "data/forged"


def get_genuine_file_names():
    append = [file.title() + "," + "1" for file in os.listdir(genuine_directory)]
    return append


def get_forged_file_names():
    append = [file.title() + "," + "0" for file in os.listdir(forge_directory)]
    return append


data = get_genuine_file_names() + get_forged_file_names()
shuffle(data)
print(data)

file = open("dataset.csv", "w+")
for str in data:
    file.write("%s\n" % str)

file.close
