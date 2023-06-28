import os
import cv2
import util
import numpy as np
import matplotlib.pyplot as plt
import random

import glob
import torch
import RRDBNet_arch as arch

from text_extractor import TextExtractor
from matplotlib import pyplot as plt, ticker

import json

class SRGAN:

    def __init__(self,test_folder) -> None:
        # Initialising model paths and test folders
        self.__test_folder= os.path.join(os.getcwd() , test_folder)
        self.__model_cfg_path = os.path.join(os.getcwd(), 'model', 'cfg', 'darknet-yolov3.cfg')
        self.__model_weights_path = os.path.join(os.getcwd(), 'model', 'weights', 'model.weights')
        self.__class_names_path = os.path.join(os.getcwd() , 'model', 'class.names')
        self.__srgan_model_path = os.path.join(os.getcwd() , 'model' , "RRDB_ESRGAN_x4.pth")
        self.__half_accurate_path = os.path.join(os.getcwd() , "0.5accurate")
        self.__low_res_folder= 'LR/*'
        

    def _license_plates_extract(self):

        for index , img_name in enumerate(os.listdir(self.__test_folder)):
            print("Test folder is " , self.__test_folder);
            print("Img name" , img_name)
            img_path = os.path.join(self.__test_folder, img_name)


            if not img_name.endswith('.jpg'):
                continue
            print("Img path is" , img_path)
            with open(self.__class_names_path, 'r') as f:
                class_names = [j[:0] for j in f.readlines() if len(j) > 2]
            net = cv2.dnn.readNetFromDarknet(self.__model_cfg_path, self.__model_weights_path)
            img = cv2.imread(img_path)
            H, W, _ = img.shape
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)
            net.setInput(blob)
            detections = util.get_outputs(net)
            bboxes = []
            class_ids = []
            scores = []

            
            for detection in detections:
                bbox = detection[:4]
                xc, yc, w, h = bbox
                bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]
                bbox_confidence = detection[4]
                class_id = np.argmax(detection[5:])
                score = np.amax(detection[5:])
                bboxes.append(bbox)
                class_ids.append(class_id)
                scores.append(score)    
            
            bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

            print("Score after util " , scores)

            for bbox_, bbox in enumerate(bboxes):
                xc, yc, w, h = bbox
                cv2.putText(img,class_names[class_ids[bbox_]],(int(xc - (w / 2)), int(yc + (h / 2) - 20)),cv2.FONT_HERSHEY_SIMPLEX,7,(0, 255, 0),15)
                license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()
                print("Saving the cropped images")
                self._save_image(f"LR\\{img_name}" , license_plate)


    def _save_image(self , folder_name , image):
        cv2.imwrite(folder_name , image)

    def _super_resolution(self):

        device = torch.device('cpu')
        model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        model.load_state_dict(torch.load(self.__srgan_model_path), strict=True)
        model.eval()
        model = model.to(device)

        print('Model path {:s}. \nTesting...'.format(self.__srgan_model_path))
        idx=0
        for path in glob.glob(self.__low_res_folder):
            idx += 1
            base = os.path.splitext(os.path.basename(path))[0]
            print(idx, base , path)
            # read images
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = img * 1.0 / 255
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_LR = img.unsqueeze(0)
            img_LR = img_LR.to(device)

            with torch.no_grad():
                output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
                output = (output * 255.0).round()
                # cv2.imwrite('results/{:s}'.format(base), output)
                try:
                    cv2.imwrite(f'results/{base}.jpg', output)
                except:
                    print("Error")


    def _text_extract(self):
        extractor = TextExtractor()
        extractor._init_text_extraction()
        
    def _start_plot(self):
        
        acc_json = open('accuracy.json',"r")
        accuracy= json.load(acc_json) 
        acc_json.close()

        total_accuracy = 0

        total_images = 0
        # Pick image from the origin destructure folder , results , cropped_lp, and perform gray scaling and thresholding on results folder images , pick up the accuracy from the above dictionary and plot it
        for image in os.listdir(self.__half_accurate_path):
            try:
                org_img = cv2.imread(os.path.join(self.__test_folder , image) , cv2.COLOR_BGR2RGB)
                sup_res_img = cv2.imread(os.path.join('results' , image) , cv2.COLOR_BGR2RGB)
                cropped_lp = cv2.imread(os.path.join('LR', image) , cv2.COLOR_BGR2RGB)
                super_res_gray_img= cv2.imread(os.path.join("grayscale" , image))
                super_res_denoise = cv2.imread(os.path.join("gray_denoise" , image))
                super_res_gray_thres_img = cv2.imread(os.path.join("gray_denoise_thresh" , image))


                if (image in accuracy):
                    details = accuracy[image]
                    acc = details["accuracy"]
                    #total_accuracy = total_accuracy+int(acc.split("%")[0])
                    total_accuracy = total_accuracy + acc
                    total_images = total_images+1
                    generated_lp = details["generated_num"]
                    actual_lp = details["actual_num"]
                    
                    super_res_gray_img_ = cv2.cvtColor(sup_res_img, cv2.COLOR_BGR2GRAY)
                    
                    org_gray_img_ = cv2.cvtColor(org_img , cv2.COLOR_BGR2GRAY) 
                    # 64 ,
                    _,org_gray_thres_img_ = cv2.threshold(org_gray_img_, 127, 255, cv2.THRESH_BINARY)

                    super_res_denoise_=None
                    for i in range(0 , 1):
                        if(i==0):
                            super_res_denoise_= cv2.fastNlMeansDenoising(super_res_gray_img_, None, 20, 7, 21)
                        else:
                            super_res_denoise_ = cv2.fastNlMeansDenoising(super_res_denoise, None, 20, 7, 21)
                    


                    _, super_res_gray_thres_img_ = cv2.threshold(super_res_denoise_, 127, 255, cv2.THRESH_BINARY)
                    print(org_gray_thres_img_.shape , super_res_gray_thres_img_.shape)
                    psnr_value = util.PSNR(org_gray_thres_img_,super_res_gray_thres_img_)          
                    # Start plotting , can actualy get slow
                    plt.figure(figsize=(16, 7))    
                    plt.subplot(231)
                    plt.title('Original Image with bounding box')
                    plt.imshow(org_img)
                    plt.subplot(232)
                    plt.title('License plate Cropped')
                    plt.imshow(cropped_lp)
                    plt.subplot(233)
                    plt.title("Super resolution Image")
                    plt.imshow(sup_res_img)
                    plt.subplot(234)
                    plt.title("Super resolution Gray Image")
                    plt.imshow(super_res_gray_img)
                    plt.subplot(235)
                    plt.title("Super resolution Gray Denoise Image")
                    plt.imshow(super_res_denoise )
                    plt.subplot(236)
                    plt.title("Super resolution Gray threshold After Denoising")
                    plt.imshow(super_res_gray_thres_img)
                    plt.text(-550 , 180 , image)
                    plt.text(-550, 200 , f'Generated LP: {generated_lp}')
                    plt.text(-550, 220 , f'%Matching based on Levinhsteins distance: {acc}')
                    plt.text(-550 , 240 , f'PSNR: Noise Reduced by: {psnr_value}')
                    # manager = plt.get_current_fig_manager()
                    # manager.full_screen_toggle()
                    plt.show()
                else:
                    print("Image not in accuracy data")  
            except KeyboardInterrupt:
                print("Keyboard Exception")
                os._exit(1)    
              
        try:
            # Average accuracy obtained 98.01001411293949
            print("Average Acuuracy " , total_accuracy/total_images)
        except:
            print("Exception occured at the Average acuuracy level")

            
    def _start_comparison(self):
        try:
            acc_json = open('accuracy.json',)
            accuracy= json.load(acc_json) 
            acc_json.close()
            pred =[]
            org =[]
            x=[]
            i=0

            for image in os.listdir(self.__test_folder):
                if image in accuracy:
                    accu = accuracy[image]
                    pred.append(accu["accuracy"])
                    org.append(0)
                    x.append(i+1)
                    i=i+1

            org_np = np.array(org)
            pred_np = np.array(pred)
            x_np = np.array(x)
            
            plt.plot(x, org_np, label='Original')
            plt.plot(x, pred_np, label='Predicted')
            plt.yscale('linear')  # Use linear scale
            plt.ylim(0, 200)


            plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(10))

            plt.xlabel('No.of plates')
            plt.ylabel('Accuracy')
            plt.title('Plot Original vs Pred')

            plt.legend()
            plt.show()

        except KeyboardInterrupt:
            print("Keyboard Interrupt")
            os._exit(1)    
        except:
            print("Some exception occured")        







        


        

 

        
        
