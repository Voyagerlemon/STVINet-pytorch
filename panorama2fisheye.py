import os
import csv
from tqdm import tqdm
from SVF.SVFProcessing import BSVCalculation


root_directory = os.path.abspath(os.path.dirname(__file__))
panorama_folder = root_directory + "/data/streetview256_seg_masterpaper"
fisheye_folder = root_directory + "/data/fisheye256_seg_masterpaper"
csv_path = os.path.join(fisheye_folder, "masterpaper_svf.csv")

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def numeric_sort(filename):
    num = int(filename.split('_')[0])
    return num

def createCsv(row):
    headers = ["CODE", "SVF", "TVF", "BVF"]
    csv_file_exist = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        if not csv_file_exist:
            writer.writeheader()
        writer.writerow(dict(zip(headers, row)))

def panorama2_seg_fisheye(seg_folder, isSegment):
    for i in tqdm(range(len(seg_folder))):
        csv_data = []
        img_code = seg_folder[i].split("_")
        csv_data.append(img_code[0])
        fisheye_name = img_code[0] + "_" + img_code[1] + "_" + img_code[2] + "_fisheye_seg.png"
        seg_path = os.path.join(panorama_folder, seg_folder[i])
        fisheye_path = os.path.join(fisheye_folder, fisheye_name)
        results = BSVCalculation.panorama2fisheyeDeepLabPlus(seg_path, fisheye_path, isSegment)
        csv_data.append(results[0])
        csv_data.append(results[1])
        csv_data.append(results[2])
        createCsv(csv_data)

def panorama2fisheye(seg_folder, isSegment):
    for i in tqdm(range(len(seg_folder))):
        img_code = seg_folder[i].split("_")
        fisheye_name = img_code[0] + "_" + img_code[1] + "_" + img_code[2] + "_fisheye.png"
        seg_path = os.path.join(panorama_folder, seg_folder[i])
        fisheye_path = os.path.join(fisheye_folder, fisheye_name)
        BSVCalculation.panorama2fisheyeDeepLabPlus(seg_path, fisheye_path, isSegment)


if __name__ == "__main__":
    panorama_files = os.listdir(panorama_folder)
    sorted_filenames = sorted(panorama_files, key=numeric_sort)
    mkdir(fisheye_folder)
    print("Start converting to fisheye images...")
    panorama2fisheye(sorted_filenames, False)