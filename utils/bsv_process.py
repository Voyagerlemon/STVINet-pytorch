import os
import argparse
from glob import glob
from tqdm import tqdm
from PIL import Image


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_argparse():
    parser = argparse.ArgumentParser(description="BSV processing")
    parser.add_argument("--input", type=str, default="./data/BSV", help="path to Dataset")
    parser.add_argument("--view", type=str, default="test", help="path to child folder")
    parser.add_argument("--size", type=float, default=1024, help="crop image size")
    return parser


def concat1024files(parent_folder, subfolder):
    total_img = []
    img_name = []

    for view in os.listdir(parent_folder):
        if view == subfolder:
            imgs = glob(os.path.join(parent_folder, view) + "/*.png")
            for img in imgs:
                total_img.append(img)
                image_file_name = [file_name for file_name in img if
                                   file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
                img_name.append(image_file_name)

    num = len(total_img)
    list_img = range(num)

    if subfolder == "train":
        new_path = parent_folder + "/bsv_train_2561024"
        mkdir(new_path)

    new_path = parent_folder + "/bsv_test_2561024"
    mkdir(new_path)
    for i in tqdm(list_img):
        img_path = total_img[i]
        img = Image.open(img_path)
        width, height = img.size
        if not os.path.exists(img_path) or width != 1024:
            raise ValueError(
                "BSV %s is not detected, please check whether the file exists in the specific path and whether the "
                "suffix is png/jpg and the width must be 1024." % img_path)

        file_prefix = os.path.basename(img_path)[:-4]
        img = Image.open(img_path)
        width, height = img.size
        # -------------------------------------------------------------#
        # Crop the image to remove the ground portion of the vehicles
        # -------------------------------------------------------------#
        cropped_image = img.crop((0, 0, width, int(height / 2)))
        cropped_image.save(f"{new_path}/{file_prefix}.jpg")
    print(f"the original image has been cropped from 512*1024 to 256*1024, total: {num}")


def concat256files(parent_folder):
    total_img = []
    img_name = []

    for view in os.listdir(parent_folder):
        if view == "bsv_train_2561024":
            imgs = glob(os.path.join(parent_folder, view) + "/*.jpg")
            for img in imgs:
                total_img.append(img)
                image_file_name = [file_name for file_name in img if
                                   file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
                img_name.append(image_file_name)

    num = len(total_img)
    list_img = range(num)

    new_path = parent_folder + "/bsv_train_256256"
    mkdir(new_path)

    for i in tqdm(list_img):
        img_path = total_img[i]
        if not os.path.exists(img_path):
            raise ValueError(
                "BSV %s is not detected, please check whether the file exists in the specific path and whether the "
                "suffix is png/jpg and the width must be 1024." % img_path)

        file_prefix = os.path.basename(img_path)[:-4]
        img = Image.open(img_path)
        width, height = img.size
        # -------------------------------------#
        # Crop the image, 256×1024-->256×256
        # -------------------------------------#
        if width == 1024 and height == 256:
            for i in range(4):
                left = i * 256
                upper = 0
                right = left + 256
                lower = 256
                cropped_image = img.crop((left, upper, right, lower))
                cropped_image.save(f"{new_path}/{file_prefix}_{i}.jpg")
    print(f"the original image has been cropped from 256*1024 to 256*256, total: {num * 4}")


def main():
    # ---------------------------------------------------------------------------------------------#
    opts = get_argparse().parse_args()
    if opts.size == 1024:
        concat1024files(opts.input, opts.view)
    elif opts.size == 256:
        concat256files(opts.input)
    else:
        raise ValueError("the parameter is incorrect, please re-enter the correct parameter.")
    # ---------------------------------------------------------------------------------------------#


if __name__ == "__main__":
    main()
