import os
from tqdm import tqdm
from PIL import Image

image_folder = "example/Qinhuai/buildings"
images = os.listdir(image_folder)
length   = len(images)
list_all = range(length)

for i in tqdm(list_all):
    seg_path = os.path.join(image_folder, images[i])
    original_image = Image.open(seg_path)
    resized_image = original_image.resize((1024, 256), Image.ANTIALIAS)
    resized_image.save(os.path.join(image_folder, images[i]))