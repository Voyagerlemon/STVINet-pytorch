import cv2
import numpy as np


sky_mask = cv2.imread('data/deeplabv3plus_seg/single/286_118.7891733_32.0263545_sky_seg.png', cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(sky_mask, threshold1=30, threshold2=100)  

# 轮廓提取
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在原始图像上绘制轮廓
original_image = cv2.imread('original_image.png')  # 假设原始图像是'original_image.png'
cv2.drawContours(original_image, contours, -1, (0, 255, 0), 2)  # 在原始图像上绘制所有轮廓，使用绿色

# 显示结果
cv2.imshow('Original Image with Sky Contours', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
