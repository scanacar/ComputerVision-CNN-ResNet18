# NECESSARY PYTHON LIBRARIES
import os
import shutil
from PIL import Image
from ResizeHelper import resize

# DATASET PATH, DATASET PATH THAT CONTAINS RESIZED IMAGES, NEW IMAGES SIZE (128, 128)
DATASET = "C:/Users/can_a/PycharmProjects/Assignment3/Dataset"
DATASET_RESIZED = "C:/Users/can_a/PycharmProjects/Assignment3/DatasetResized"
SIZE = (128, 128)

if os.path.exists(DATASET_RESIZED):
    shutil.rmtree(DATASET_RESIZED)

for root, folders, files in os.walk(DATASET):
    for sub_folder in folders:

        save_folder = os.path.join(DATASET_RESIZED, sub_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        image_names = os.listdir(os.path.join(root, sub_folder))
        for i in image_names:
            image_path = os.path.join(root, sub_folder, i)
            image = Image.open(image_path)
            resized = resize(image, SIZE)
            save_as = os.path.join(save_folder, i)
            resized.save(save_as)
