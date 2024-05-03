import os
import csv
from tqdm import tqdm
from PIL import Image
from SVF.SVFProcessing import BSVCalculation


root_directory = os.path.abspath(os.path.dirname(__file__))
output512 = root_directory + "/data/Qinhuai/streetview512/"
output256 = root_directory + "/data/Qinhuai/streetview256/"


def panoramic_mosaic():

    bsv = BSVCalculation()
    csv_file = "data/Nanjing_qinhuai_75.csv"
    with open(csv_file, newline='', encoding="UTF-8") as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        rows_list = list(csv_reader)
        length = len(rows_list)

        for i in tqdm(range(length)):
            row = rows_list[i]
            output512_file_path = output512 + row[0] + "_" + row[1] + "_" + row[2] + "_panorama.png"
            output256_file_path = output256 + row[0] + "_" + row[1] + "_" + row[2] + "_panorama.png"
            bsv.getMosaicImage(row[9], output512_file_path, output256_file_path)
        # for row in csv_reader:
        #     print(row[9])



if __name__ == "__main__":
    print("开始爬取")
    panoramic_mosaic()