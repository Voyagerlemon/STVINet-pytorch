import torch
import numpy as np
from torch.autograd import Variable
from nets.segmentator.deeplabv3.deeplabv3 import DeepLabV3
from nets.segmentator.deeplabv3.datasets import DatasetSeq
from utils.color import label_img_to_color


class Segmentator:
    """semantic segmentation: deeplabv3 and deeplabv3+"""
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
    
    # def deeplabv3plus():
