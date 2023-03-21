import cv2
from matplotlib import pyplot as plt
import numpy as np


def draw_circle(image, t_points):

    for i in range(4):
        cv2.circle(image, (t_points[i, 0], t_points[i, 1]), 20, (255, 0, 0))
    
    return image


def test(img):

    IMAGE_W = img.shape[1]
    IMAGE_H = img.shape[0]

    src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    dst = np.float32([[400, IMAGE_H], [600, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    
    img = img[300:IMAGE_H, 0:IMAGE_W] 
    
    M = cv2.getPerspectiveTransform(dst, src) 
    Minv = cv2.getPerspectiveTransform(src, dst) 
    
    warped_img = cv2.warpPerspective(img, Minv, (IMAGE_W, IMAGE_H)) 
    
    plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) 
    plt.show()
    
    return None


if __name__ == "__main__":
    
    image = cv2.imread("../data/view1.png")
    print(image.shape)
    test(image)
    
