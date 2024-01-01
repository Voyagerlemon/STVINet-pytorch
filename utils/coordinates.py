import os
import csv
from glob import glob
from tqdm import tqdm


def duplicates(coordinates, new_coordinate):
    for coord in coordinates:
        if coord == new_coordinate:
            return True
    return False


csv_filename = "bsv_coordinates.csv"
locations = []


def Coordinates():
    parent_folder = "data/BSV/bsv_train_256256"
    total_img = []

    imgs = glob(parent_folder + "/*.jpg")
    for img in imgs:
        total_img.append(img)

    for i in tqdm(range(len(total_img))):
        img_path = total_img[i]
        if not os.path.exists(img_path):
            raise ValueError(
                "BSV %s is not detected, please check whether the file exists in the specific path and whether the "
                "suffix is png/jpg." % img_path)
        file_prefix = os.path.basename(img_path)[:-5].split("_")
        lng_lat = (file_prefix[2], file_prefix[3])
        locations.append(lng_lat)


if __name__ == "__main__":
    Coordinates()
    csv_path = "data/" + csv_filename
    with open(csv_path, mode="w", newline='') as file:
        writer = csv.writer(file)

        coordinates_written = set()
        for lng, lat in locations:
            coordinate_pair = (lng, lat)
            if coordinate_pair not in coordinates_written:
                writer.writerow(coordinate_pair)
                coordinates_written.add(coordinate_pair)
    print("the coordinate pairs have been written to the csv file.")
