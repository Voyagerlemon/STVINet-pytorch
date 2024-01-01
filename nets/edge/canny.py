import cv2
import numpy as np

img = cv2.imread(r"E:\\Master\\Papers\\SkyViewFactor\\software\\STVINet-pytorch\\data\\lama\\138_118.8245508486703_32.02957875349542.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 30, 100)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(gray)

for i, contour in enumerate(contours):
    # 创建单独的掩码
    contour_mask = np.zeros_like(gray)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # 提取每个元素
    element = cv2.bitwise_and(img, img, mask=contour_mask)
    
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
# out_x = cv2.Canny(img, 50, 150)
# out_y = cv2.Canny(img, 100, 150)

# tack_img = np.hstack((out_x, out_y))
# width, height = tack_img.shape
# canny_img = tack_img[:, :height // 2]
# output_img = "data/inference/138_118.8245508486703_32.02957875349542_edge.png"
# cv2.imwrite(output_img, canny_img)