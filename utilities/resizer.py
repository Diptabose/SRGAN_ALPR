import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

train_dir = "C:\\Users\\dipta\\Downloads\\train-detector-master\\train-detector-master\\eu"
dest_dir = "C:\\Users\\dipta\\Downloads\\train-detector-master"

for img in os.listdir( train_dir):
    
    img_array = cv2.imread(os.path.join(train_dir , img))
    img_array = cv2.resize(img_array, (128,128) )
    lr_img_array = cv2.resize(img_array,(32,32) )
    print('Saving')
    cv2.imwrite(os.path.join(dest_dir , "hr_images" , img), img_array)
    cv2.imwrite(os.path.join(dest_dir , "lr_images" , img), lr_img_array)