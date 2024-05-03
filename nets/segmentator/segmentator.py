'''
Author: Voyagerlemon xuhaiyangw@163.com
Date: 2023-11-01 19:52:13
LastEditors: Voyagerlemon xuhaiyangw@163.com
LastEditTime: 2024-04-24 09:30:39
FilePath: \STVINet-pytorch\nets\segmentator\segmentator.py
Description: 语义分割网络
'''
import cv2
import time
import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
from nets.segmentator.deeplabv3plus.deeplab import DeeplabV3_plus
from nets.segmentator.deeplabv3.deeplabv3 import DeepLabV3
from nets.segmentator.deeplabv3.datasets import DatasetSeq
from utils.color import label_img_to_color


class Segmentator:
    """semantic segmentation: deeplabv3 and deeplabv3+"""
    """root_directory: Predicts the upper directory address of the image."""
    def deeplabv3(root_directory):
        batch_size = 2
        network = DeepLabV3("data", project_dir=root_directory).cuda()
        network.load_state_dict(torch.load("model_data/deeplabv3/model_1_epoch_1000.pth"))

        
        val_dataset = DatasetSeq(cityscapes_data_path=root_directory)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        network.eval()

        unsorted_img_ids = []
        segmentation_imgs = []
        for step, (img_all, img_ids) in enumerate(val_loader):
            with torch.no_grad():
                img_all = Variable(img_all).cuda()
                outputs = network(img_all)
                outputs = outputs.data.cpu().numpy()
                pred_label_imgs = np.argmax(outputs, axis=1)
                pred_label_imgs = pred_label_imgs.astype(np.uint8)

                for i in range(pred_label_imgs.shape[0]):
                    pred_label_img = pred_label_imgs[i]
                    img_id = img_ids[i]
                    pred_label_img_color = label_img_to_color(pred_label_img)
                    segmentation_imgs.append(pred_label_img_color)
                    unsorted_img_ids.append(img_id)
        return segmentation_imgs, unsorted_img_ids
    
    def deeplabv3plus(root_directory, out_directory):
        deeplab = DeeplabV3_plus()
        #----------------------------------------------------------------------------------------------------------#
        # mode用于指定测试的模式：
        # 'predict'    :   表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
        # 'dir_predict':   表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释
        # 'export_onnx':   表示将模型导出为onnx，需要pytorch1.7.1以上
        #----------------------------------------------------------------------------------------------------------#
        mode = "dir_predict"
        #---------------------------------------------------------------------------#
        # count       : 指定了是否进行目标的像素点计数（即面积）与比例计算
        # name_classes: 区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
        #
        # count、name_classes仅在mode='predict'时有效
        #---------------------------------------------------------------------------#
        count           = False
        name_classes    = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
                       "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle",
                       "bicycle"]
        #------------------------------------------------------#
        # dir_origin_path:  指定了用于检测的图片的文件夹路径
        # dir_save_path  :  指定了检测完图片的保存路径
        #------------------------------------------------------#
        dir_origin_path = root_directory
        dir_save_path   = out_directory
        #------------------------------------------------------#
        # simplify       :  使用Simplify onnx
        # onnx_save_path :  指定了onnx的保存路径
        #------------------------------------------------------#
        simplify        = True
        onnx_save_path  = "model_data/deeplabv3plus/models.onnx"

        if mode == "predict":
            while True:
                img = input('Input image filename:')
                try:
                    image = Image.open(img)
                except:
                    print('Open Error! Try again!')
                    continue
                else:
                    r_image = deeplab.detect_image(image, count=count, name_classes=name_classes)
                    r_image.show()

        elif mode == "dir_predict":
            import os
            from tqdm import tqdm
    
            img_names = os.listdir(dir_origin_path)
            for img_name in tqdm(img_names):
                if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    image_path  = os.path.join(dir_origin_path, img_name)
                    image       = Image.open(image_path)
                    r_image     = deeplab.detect_image(image)
                    if not os.path.exists(dir_save_path):
                        os.makedirs(dir_save_path)
                    r_image.save(os.path.join(dir_save_path, img_name))
        elif mode == "export_onnx":
            deeplab.convert_to_onnx(simplify, onnx_save_path)
            
        else:
            raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
