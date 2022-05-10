import os
import cv2
from PIL import Image
array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name


def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+directory_name):
        #print(filename) #just for test
        #img is used to store the image data 
        if filename.endswith(".jpg"):
            print(filename)
            im1 = Image.open(filename)
            im1.save(filename.split('.')[0]+".png")
            os.remove(filename)
            #array_of_img.append(img)
            #print(img)
            #print(array_of_img)



read_directory(".")