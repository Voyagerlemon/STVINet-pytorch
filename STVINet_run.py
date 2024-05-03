import os
import cv2
import csv
import argparse
import numpy as np
from tqdm import tqdm
from SVF.SVFProcessing import BSVCalculation
from nets.lama.predict_single import lamaOcclusion
from nets.segmentator.segmentator import Segmentator


root_directory = os.path.abspath(os.path.dirname(__file__))
fisheye_folder = root_directory + "/data/fisheye/"
csv_path = os.path.join(fisheye_folder, "test_original.csv")


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, choices=["deeplabv3", "deeplabv3plus"], help="please enter a model name")
    return parser


def createCsv(row):
    existing_row = []
    csv_file_exist = os.path.exists(csv_path)
    headers = ["CODE", "SVF", "TVF", "BVF"]

    try:
        with open (csv_path, "r") as file:
            reader = csv.DictReader(file)
            existing_row = list(reader)
    except FileNotFoundError:
        pass
    
    back_code = dict(zip(headers, row))["CODE"]
    for item in existing_row:
        next_code = item["CODE"] 
        if int(next_code) == int(back_code):
            return
        with open (csv_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            if not csv_file_exist:
                writer.writeheader()
            writer.writerow(dict(zip(headers, row)))
    
    if len(existing_row) == 0:
        with open (csv_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            if not csv_file_exist:
                writer.writeheader()
            writer.writerow(dict(zip(headers, row)))


def main():
    opts = get_argparse().parse_args()
    seg_folder = root_directory + "/data/deeplabv3plus_seg/whole"
    lama_folder = root_directory + "/data/lama/"
    inference_folder = root_directory + "/data/inference/"

    if opts.model == "deeplabv3":
        seg_results = Segmentator.deeplabv3(root_directory)
        length   = len(seg_results[0])
        list_all = range(length)

        for i in tqdm(list_all):
            csv_data = []
            height, width, channels = seg_results[0][i].shape
            img_h = height // 2
            img = seg_results[0][i][:img_h, :, :]
            seg_path = root_directory + "/data/deeplabv3_seg/whole/" + seg_results[1][i][:-4] + "_seg.png"
            cv2.imwrite(seg_path, img)

            img_code = seg_results[1][i].split("_")
            csv_data.append(img_code[0])

            fisheye_path = fisheye_folder + seg_results[1][i][:-4] + "_fisheye_seg.png"
            results = BSVCalculation.panorama2fisheyeDeepLab(seg_path, fisheye_path, True)
            csv_data.append(results[0])
            csv_data.append(results[1])
            csv_data.append(results[2])

            createCsv(csv_data)

            mask_image = cv2.imread(seg_path)
        
            sky_color = (180, 130, 70)
            lower_sky_bound = np.array([sky_color[2] - 10, sky_color[1] - 10, sky_color[0] - 10])
            upper_sky_bound = np.array([sky_color[2] + 10, sky_color[1] + 10, sky_color[0] + 10])
            color_sky_mask = cv2.inRange(mask_image, lower_sky_bound, upper_sky_bound)
            sky_image = cv2.bitwise_and(mask_image, mask_image, mask=color_sky_mask)

            sky_path = root_directory + "/data/deeplabv3_seg/single/" + seg_results[1][i][:-4] + "_sky_seg.png"
            cv2.imwrite(sky_path, sky_image)

            tree_color = (35, 142, 107)
            lower_tree_bound = np.array([tree_color[2] - 10, tree_color[1] - 10, tree_color[0] - 10])
            upper_tree_bound = np.array([tree_color[2] + 10, tree_color[1] + 10, tree_color[0] + 10])
            color_tree_mask = cv2.inRange(mask_image, lower_tree_bound, upper_tree_bound)
            tree_image = cv2.bitwise_and(mask_image, mask_image, mask=color_tree_mask)

            tree_path = root_directory + "/data/deeplabv3_seg/single/" + seg_results[1][i][:-4] + "_tree_seg.png"
            cv2.imwrite(tree_path, tree_image)

            building_color = (70, 70, 70)
            lower_building_bound = np.array([building_color[2] - 10, building_color[1] - 10, building_color[0] - 10])
            upper_building_bound = np.array([building_color[2] + 10, building_color[1] + 10, building_color[0] + 10])
            color_building_mask = cv2.inRange(mask_image, lower_building_bound, upper_building_bound)
            building_image = cv2.bitwise_and(mask_image, mask_image, mask=color_building_mask)

            building_path = root_directory + "/data/deeplabv3_seg/single/" + seg_results[1][i][:-4] + "_building_seg.png"
            cv2.imwrite(building_path, building_image)

            stb_path = root_directory + "/data/deeplabv3_seg/single/" + seg_results[1][i][:-4] + "_stb_seg.png"
            stb_image = sky_image + tree_image + building_image
            cv2.imwrite(stb_path, stb_image)

            mask_image[color_tree_mask > 0]  = [255, 255, 255]
            mask_image[color_tree_mask == 0] = [0, 0, 0]
            tree_mask_path = lama_folder + seg_results[1][i][:-4] + "_mask.png"
            cv2.imwrite(tree_mask_path, mask_image)

            lamaOcclusion(lama_folder, inference_folder)
        
    elif opts.model == "deeplabv3plus":
        seg_results = os.listdir(seg_folder)
        length   = len(seg_results)
        list_all = range(length)
        # lamaOcclusion(lama_folder, inference_folder)

        for i in tqdm(list_all):
            # lamaOcclusion(lama_folder, inference_folder)

            seg_path = os.path.join(seg_folder, seg_results[i])
            mask_image = cv2.imread(seg_path)
        
            sky_color = (70, 130, 180)
            lower_sky_bound = np.array([sky_color[2] - 10, sky_color[1] - 10, sky_color[0] - 10])
            upper_sky_bound = np.array([sky_color[2] + 10, sky_color[1] + 10, sky_color[0] + 10])
            color_sky_mask = cv2.inRange(mask_image, lower_sky_bound, upper_sky_bound)
            sky_image = cv2.bitwise_and(mask_image, mask_image, mask=color_sky_mask)

            sky_path = root_directory + "/data/deeplabv3plus_seg/single/" + seg_results[i][:-13] + "_sky_seg.png"
            cv2.imwrite(sky_path, sky_image)

            tree_color = (107, 142, 35)
            lower_tree_bound = np.array([tree_color[2] - 10, tree_color[1] - 10, tree_color[0] - 10])
            upper_tree_bound = np.array([tree_color[2] + 10, tree_color[1] + 10, tree_color[0] + 10])
            color_tree_mask = cv2.inRange(mask_image, lower_tree_bound, upper_tree_bound)
            tree_image = cv2.bitwise_and(mask_image, mask_image, mask=color_tree_mask)

            tree_path = root_directory + "/data/deeplabv3plus_seg/single/" + seg_results[i][:-13] + "_tree_seg.png"
            cv2.imwrite(tree_path, tree_image)

            building_color = (70, 70, 70)
            lower_building_bound = np.array([building_color[2] - 10, building_color[1] - 10, building_color[0] - 10])
            upper_building_bound = np.array([building_color[2] + 10, building_color[1] + 10, building_color[0] + 10])
            color_building_mask = cv2.inRange(mask_image, lower_building_bound, upper_building_bound)
            building_image = cv2.bitwise_and(mask_image, mask_image, mask=color_building_mask)

            building_path = root_directory + "/data/deeplabv3plus_seg/single/" + seg_results[i][:-13] + "_building_seg.png"
            cv2.imwrite(building_path, building_image)

            stb_path = root_directory + "/data/deeplabv3plus_seg/single/" + seg_results[i][:-13] + "_stb_seg.png"
            stb_image = sky_image + tree_image + building_image
            cv2.imwrite(stb_path, stb_image)

            mask_image[color_tree_mask > 0]  = [255, 255, 255]
            mask_image[color_tree_mask == 0] = [0, 0, 0]
            tree_mask_path = lama_folder + seg_results[i][:-13] + "_panorama_mask.png"
            cv2.imwrite(tree_mask_path, mask_image)
            lamaOcclusion(lama_folder, inference_folder)

            
    # for segmentation_img, img_path, panorama in zip(seg_results[0], seg_results[1], seg_results[2]):
    #     height, width, channels = segmentation_img.shape
    #     img_h = height // 2
    #     img = segmentation_img[:img_h, :, :]
    #     seg_path = root_directory + "/data/deeplabv3_seg/" + img_path[:-4] + "_seg.png"
    #     cv2.imwrite(seg_path, img)

    #     height_p, width_p, channels_p = panorama.shape
    #     img_p_h = height_p // 2
    #     img_p = panorama[:img_p_h, :, :]
    #     panorama_path = root_directory + "/data/lama/" + img_path[:-4] + ".png"
    #     cv2.imwrite(panorama_path, img_p)



if __name__ == '__main__':
    main()
