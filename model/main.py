import cv2
from matplotlib import pyplot as plt
import numpy as np


class birdsEyeView():

    def __init__(self, views=4, width=1073, height=605):
        self.views = 4
        self.views = ['front', 'left', 'right', 'back']
        self.canvas = None
        self.IMAGE_W = width
        self.IMAGE_H = height
        self.src = np.float32([[0, self.IMAGE_H], [self.IMAGE_W, self.IMAGE_H], [0, 0], [self.IMAGE_W, 0]])
        self.dst = np.float32([[500, self.IMAGE_H], [640, self.IMAGE_H], [0, 0], [self.IMAGE_W, 0]])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst) 
        self.translateFront = np.array([[1,0,1000],[0,1,0],[0,0,1]], dtype=np.float32)
        self.translateRight = np.array([[1,0,390],[0,1,75],[0,0,1]], dtype=np.float32)
        self.translateLeft = np.array([[1,0,920],[0,1,-2020],[0,0,1]], dtype=np.float32)
        self.translateBack = np.array([[1,0,-1100],[0,1,-530],[0,0,1]], dtype=np.float32)
        self.clip = 550

    def read_images(self, imageAddress):
        '''
        Args: str // Address to the image
        Return: An image as an np.ndarray 
        '''
        image = cv2.imread(imageAddress)

        return image
    
    def createImageSet(self, images):
        '''
        Args: set of 4 images => <4 x np.ndarray>   
        Return: dictionary with keys as name of views and their corresponding images => np.ndarray 
        '''
        imageSet = {}
        for i, image in enumerate(images):
            imageSet[self.views[i]] = images[i]

        return imageSet

    def transformImage(self, image, view):

        image = image[self.clip:self.IMAGE_H, 0:self.IMAGE_W]

        if view == 'front':
            warpedImage = cv2.warpPerspective(image, self.M, (self.IMAGE_W*3, self.IMAGE_H*3))
            warpedImage = cv2.warpPerspective(warpedImage, self.translateFront, (self.IMAGE_W*3, self.IMAGE_H*3)) 
            
        if view == 'left':
            warpedImage = cv2.warpPerspective(image, self.M, (self.IMAGE_W*3, self.IMAGE_H*3))
            warpedImage = cv2.rotate(warpedImage, cv2.ROTATE_90_COUNTERCLOCKWISE)
            warpedImage = cv2.warpPerspective(warpedImage, self.translateLeft, (self.IMAGE_W*3, self.IMAGE_H*3))

        if view == 'right':
            warpedImage = cv2.warpPerspective(image, self.M, (self.IMAGE_W*3, self.IMAGE_H*3))
            warpedImage = cv2.rotate(warpedImage, cv2.ROTATE_90_CLOCKWISE)
            warpedImage = cv2.warpPerspective(warpedImage, self.translateRight, (self.IMAGE_W*3, self.IMAGE_H*3))
        
        if view == 'back':
            warpedImage = cv2.warpPerspective(image, self.M, (self.IMAGE_W*3, self.IMAGE_H*3))
            warpedImage = cv2.rotate(warpedImage, cv2.ROTATE_180)
            warpedImage = cv2.warpPerspective(warpedImage, self.translateBack, (self.IMAGE_W*3, self.IMAGE_H*3))

        return warpedImage
        
    def basicBlend(self, transformedImageSet):
        blendedImage = np.zeros(shape=(self.IMAGE_H*3, self.IMAGE_W*3, 3), dtype=np.uint8)

        for i in range(0, 4):
            blendedImage = cv2.addWeighted(blendedImage, 1, transformedImageSet[i], 1, 0)

        return blendedImage


if __name__ == "__main__":
    bev = birdsEyeView()
    imagesAddress = ["../assets/front_camera.png", "../assets/left_camera.png", "../assets/right_camera.png", "../assets/back_camera.png"]
    imageSet = [bev.read_images(i) for i in imagesAddress]
    imageSet = bev.createImageSet(imageSet)
    transformedImageSet = []

    for view, img in imageSet.items():
        transformedImageSet.append(bev.transformImage(img, view))

    blendedImage = bev.basicBlend(transformedImageSet)
    plt.imshow(cv2.cvtColor(blendedImage, cv2.COLOR_BGR2RGB))
    plt.show()
    
    
    