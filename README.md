# SRGAN_ALPR


Install the required packages from requirements.txt

This project contains the code for training the SRGAN model with License plate dataset (Demonstartion purpose)

A generalised model is used RRDB_ESRGAN_x4.pth for super resolution. Download here: https://drive.google.com/uc?id=1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene



Any step in main.py can be invoked by commenting other methods after first complete run of all methods.
Intermediate images such as denoised images, threshold_inversion can stored in respective folders , so that any method can be invoked after first complete run.


**Folder Structure**
architecture_diagram  ==> Contains the architecture diagram of SRGAN
train_srgan ==> A Jupyter notbook from Google colab to train the model

Please create these folders in the root directory of the project, if not present
LR ==> Stores the cropped license plates 
results ==> Stores the super resolution images
grayscale ==> Part of post Processing, stores grayscaled images of cropped license plates
gray_denoise ==> Part of post processing, stores denoised images after gray_scaling
gray_denoise_thresh ==> Part of post processing, stores threshold inversion image of the above image
0.5accurate ==> Stores the cropped images whose detection accuracy is atleast 0.5
utilities ==> Consists of helper methods to process the datasets


**Train dataset**
Train dataset was taken from the eu folder of the following github repository https://github.com/openalpr/train-detector
Train dataset will be reshaped to 32x32 and 128x128 using the resizer.py from utilities.
Make sure the train dataset in google drive should have 'lr_images' 'hr_images' in a folder 'SRGAN_TEST'.
Use the above resizer and upload the images to google drive.

**Vehicles image**
Vehicles Images were taken from the kaggle dataset having various Indian LPs.

**Model**
This folder contains the model and weights RRDB_ESRGAN_x4.pth and yolov3 related models
model
   cfg
     darknet-yolov3.cfg
   weights
     model.weights
   class.names
   RRDB_ESRGAN_x4.pth    