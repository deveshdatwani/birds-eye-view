import cv2
from matplotlib import pyplot as plt
import numpy as np


def draw_circle(image, t_points):

    for i in range(4):
        cv2.circle(image, (t_points[i, 0], t_points[i, 1]), 20, (255, 0, 0))
    
    return image


def birds_eye_view(image: np.ndarray) -> np.ndarray:
   
    WIDTH = image.shape[1]
    HEIGHT = image.shape[0]
    
    x1, y1 = 685, 560
    x2, y2 = 750, 560
    x3, y3 = 25, 860
    x4, y4 = 1350, 860
    
    points = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    
    X1, Y1 = 0, 0
    X2, Y2 = WIDTH, 0
    X3, Y3 = 0, HEIGHT 
    X4, Y4 = WIDTH, HEIGHT

    t_points = np.float32([[X1, Y1], [X2, Y2], [X3, Y3], [X4, Y4]])
    

    M = cv2.getPerspectiveTransform(points, t_points)

    warpped = cv2.warpPerspective(image.copy(), M, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)

    plt.imshow(warpped)
    plt.show()

    return image


def test(image):

    IMAGE_W = image.shape[1]
    IMAGE_H = image.shape[0]

    src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    M = cv2.getPerspectiveTransform(src, dst) 
    Minv = cv2.getPerspectiveTransform(dst, src)

    img = cv2.imread('../assets/road.jpg')
    img = img[550:550+IMAGE_H, 0:IMAGE_W] 
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) 
    

    plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) 
    plt.show()
    
    return None


if __name__ == "__main__":
    
    image = cv2.imread("../assets/road.jpg")
    test(image)
    
