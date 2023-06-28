from project import SRGAN

class Main(SRGAN):
    def __init__(self, test_folder:str) -> None:
        super().__init__(test_folder)


#Test vehicle images ********* 78 Images Obtained Average Acuuracy- 98.002% *********
vehicles_test_folder="vehicles_test_images"      

#Test cases images ( far , font , skew , blur , randomly-googled)
testcase_folder = "execution_test_cases"


# Run this in order , to get the license plate characters

#Initialise the Main with folder containing images
srgan = Main(vehicles_test_folder)

# Invoke this to extract the license plates
srgan._license_plates_extract()

# Invoke this to perform super_resolution imaging here
srgan._super_resolution()

# Invoke this to start text extraction
srgan._text_extract()

# Invoke this to get the matplotlib representation of cropped image, denoised image and threshold_inversion
srgan._start_plot()

# Invoke this to get the accuracy plot Original vs Predicted 
srgan._start_comparison()



'''
Any step can be invoked by commenting other methods after first complete run of all methods.
Intermediate images such as denoised images, threshold_inversion can stored in respective folders , so that any method can be invoked after first complete run.
'''


'''
Folder Structure


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
'''

'''
Train dataset

Train dataset was taken from the eu folder of the following github repository https://github.com/openalpr/train-detector
Train dataset will be reshaped to 32x32 and 128x128 using the resizer.py from utilities.
Make sure the train dataset in google drive should have 'lr_images' 'hr_images' in a folder 'SRGAN_TEST'.
Use the above resizer and upload the images to google drive.

Vehicles image
Vehicles Images were taken from the kaggle dataset having various Indian LPs.

'''




