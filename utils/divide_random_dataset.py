import os
import sys
import shutil
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

# os.chdir(sys.path[4])
sys.path.append("/root/lama-with-refiner/")
# -----------------------------------------------------------------------------------------------#
# the script is used to randomly divide the dataset used by the image inpainting network LaMa
# training data : validation data = 9:1
# train_percent is used to change the ratio of validation data
# -----------------------------------------------------------------------------------------------#
train_val_percent = 0.95
train_percent = 0.9

dataset_path = "data/BSV/bsv_train_256256"
BSV_path = "bsv_train_256256"


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    random.seed(0)
    print("Generate divide the dataset in BSV.")
    train_path = os.path.join(BSV_path, "train")
    validation_path = os.path.join(BSV_path, "val")
    visual_path = os.path.join(BSV_path, "visual_test")
    mkdir(train_path)
    mkdir(validation_path)
    mkdir(visual_path)

    temp_dataset = os.listdir(dataset_path)
    total_dataset = []
    for sample in temp_dataset:
        if sample.endswith(".jpg"):
            total_dataset.append(sample)

    num = len(total_dataset)
    list_num = range(num)
    tv = int(num * train_val_percent)
    tr = int(tv * train_percent)
    vd = int(num * (1 - train_val_percent))
    visual_dataset = random.sample(list_num, vd)
    train_val_dataset = random.sample(list_num, tv)
    train_dataset = random.sample(list_num, tr)
    print("train and val size: ", tv)
    print("train size: ", tr)
    print("visual test: ", vd)
    print("val size: ", num - vd - tr)

    for i in tqdm(list_num):
        if i in visual_dataset:
            shutil.copy(os.path.join(dataset_path, total_dataset[i]), os.path.join(BSV_path, "visual_test",
                                                                                   total_dataset[i]))
        elif i in train_val_dataset:
            if i in train_dataset:
                shutil.copy(os.path.join(dataset_path, total_dataset[i]), os.path.join(BSV_path, "train",
                                                                                       total_dataset[i]))
            else:
                shutil.copy(os.path.join(dataset_path, total_dataset[i]), os.path.join(BSV_path, "val",
                                                                                       total_dataset[i]))
    print("Generate divide the dataset in BSV done.")
