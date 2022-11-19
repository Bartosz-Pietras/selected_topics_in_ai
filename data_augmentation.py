import os
import shutil

import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import glob
import numpy as np

AUGMENTED_DATA_DIR = "cnn_data_augmented"

DIRECTORY_LIST = [
    "circle",
    "circles",
    "cross",
    "plus",
    "square",
    "triangle",
    "waves",
]


def add_gaussian_noise(image):
    row, col, channel = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, channel))
    gauss = gauss.reshape(row, col, channel)
    noisy = image + gauss
    return noisy


def load_dataset():
    return tf.keras.utils.image_dataset_from_directory(
        directory="cnn_data",
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        image_size=(64, 64),
        shuffle=True,
    )


def group_images() -> dict:
    image_dict = dict()
    for directory in DIRECTORY_LIST:
        images = glob.glob(f"cnn_data/{directory}/*.png")
        image_dict[directory] = images
    return image_dict


def save_augmented_images(augmented_images_dict, augmentation_type):
    for key, value in augmented_images_dict.items():
        for idx, image in enumerate(value):
            image_to_save = Image.fromarray(image.astype(np.uint8))
            image_to_save = ImageOps.grayscale(image_to_save)
            image_to_save.save(f"{AUGMENTED_DATA_DIR}/{key}/{augmentation_type}_{key}_{idx}.png")


def create_dirs_for_augmented_data():
    if os.path.isdir(AUGMENTED_DATA_DIR):
        shutil.rmtree(AUGMENTED_DATA_DIR)
    os.mkdir(AUGMENTED_DATA_DIR)

    for dirpath in DIRECTORY_LIST:
        os.mkdir(f"{AUGMENTED_DATA_DIR}/{dirpath}")


def image_augmentation(num_of_new_images=1000, augmentation_type="rotation"):
    count = 0
    images = group_images()
    augmented_images = dict()

    for key in images.keys():
        augmented_images[key] = []

    while count < num_of_new_images:
        for key, value in images.items():
            for image_path in value:
                image = keras.preprocessing.image.load_img(image_path)
                image_arr = keras.preprocessing.image.img_to_array(image)
                augmented_image = np.array([])
                if augmentation_type == "rotation":
                    augmented_image = keras.preprocessing.image.random_rotation(x=image_arr, rg=30, fill_mode="nearest")
                elif augmentation_type == "gauss":
                    augmented_image = add_gaussian_noise(image_arr)
                elif augmentation_type == "flip":
                    augmented_image = np.flip(image_arr, 0)
                augmented_images[key].append(augmented_image)
                count += 1
                if count == num_of_new_images:
                    return augmented_images


if __name__ == "__main__":
    create_dirs_for_augmented_data()

    gauss_images = image_augmentation(augmentation_type="gauss")
    rotated_images = image_augmentation(augmentation_type="rotation")
    flipped_images = image_augmentation(augmentation_type="flip")

    save_augmented_images(gauss_images, augmentation_type="gauss")
    save_augmented_images(rotated_images, augmentation_type="rotation")
    save_augmented_images(flipped_images, augmentation_type="flip")
