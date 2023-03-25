import cv2
from matplotlib import pyplot as plt
import numpy as np


def draw_circle(image, t_points):

    for i in range(4):
        cv2.circle(image, (t_points[i, 0], t_points[i, 1]), 20, (255, 0, 0))
    
    return image


def test(image_front, image_right, image_left, image_back):

    IMAGE_W = image_right.shape[1]
    IMAGE_H = image_right.shape[0]

    src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    dst = np.float32([[500, IMAGE_H], [640, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    
    image_front = image_front[450:IMAGE_H, 0:IMAGE_W] 
    image_right = image_right[450:IMAGE_H, 0:IMAGE_W] 
    
    M = cv2.getPerspectiveTransform(src, dst) 

    warped_image_front = cv2.warpPerspective(image_front, M, (IMAGE_W*2, IMAGE_H*2))
    warped_image_right = cv2.warpPerspective(image_right, M, (IMAGE_W*2, IMAGE_H*2))

    warped_image_right = cv2.rotate(warped_image_right, cv2.ROTATE_90_CLOCKWISE)
    
    translate = np.array([[1,0,400],[0,1,0],[0,0,1]], dtype=np.float32)
    
    warped_image_front = cv2.warpPerspective(warped_image_front, translate, (IMAGE_W*2, IMAGE_H*2))
    
    translate = np.array([[1,0,380],[0,1,80],[0,0,1]], dtype=np.float32)

    warped_image_right = cv2.warpPerspective(warped_image_right, translate, (IMAGE_W*2, IMAGE_H*2))

    camera_output = cv2.addWeighted(warped_image_front, 1, warped_image_right, 1, 0)

    plt.imshow(cv2.cvtColor(camera_output, cv2.COLOR_BGR2RGB)) 
    plt.show()
    
    return None


if __name__ == "__main__":
    
    image_front = cv2.imread("../assets/front_camera.png")
    image_right = cv2.imread("../assets/right_camera.png")
    image_left = cv2.imread("../assets/left_camera.png")
    image_back = cv2.imread("../assets/back_camera.png")
    test(image_front, image_right, image_left, image_back)
    
