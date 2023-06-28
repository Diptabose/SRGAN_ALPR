import os 
import cv2
import easyocr
import json



class TextExtractor:

    def __init__(self) -> None:
        os.system('cls')
        self.__reader = easyocr.Reader(['en'])
        self.__final_license_plate_dir = "results"

        path_json = open('path_number.json',"r")
        self.__image_dict= json.load(path_json) 
        path_json.close()

        

    def _prettified_text(self, text):
        mod_text = ''.join(e for e in text if e.isalnum()).upper()
        return mod_text
       
        
    def _custom_accuracy_v1(self , org_text , generated_text):
        org_text_len = len(org_text)
        generated_text_len = len(generated_text)
        matched=0
        for index,letter in enumerate(org_text):
            if(index>generated_text_len):
                return matched / org_text_len
            if(letter == generated_text[index]):
                matched=matched+1
        return str((matched / org_text_len)*100)+'%'


    def _custom_accuracy_v2(self , org_text , generated_text):
        generated_text_len = len(generated_text)
        ascii_org = 0
        ascii_gen = 0
        for index,letter in enumerate(org_text):
            if(index>generated_text_len):
                return ascii_org/ascii_gen
            ascii_org=ascii_org+ ord(letter)
            ascii_gen= ascii_gen + ord(generated_text[index])   
        return str((ascii_gen/ascii_org)*100)+'%' if ascii_gen<=ascii_org else str((ascii_org/ascii_gen)*100)+'%'
    
    # Based Levihnstein's distance == >Extra: Added Ascii based character decoding to mathematically calculate acuuracy.
    def _custom_accuracy_v3(self , org_text , generated_text):
        generated_text_len = len(generated_text)
        original_text_len = len(org_text)
        difference = original_text_len - generated_text_len
        ascii_org = 0
        ascii_gen = 0
        for letter in org_text:
            ascii_org = ascii_org + ord(letter)
        
        for letter in generated_text:
            ascii_gen = ascii_gen + ord(letter)

        return (ascii_gen/ascii_org)*100 if ascii_gen<=ascii_org else (ascii_org/ascii_gen)*100 

            
    def _init_text_extraction(self):
        acc={}
        total_accuracy = 0
        total_images = 0


        for index , image in enumerate(os.listdir(self.__final_license_plate_dir)):
            cv_img_org =  cv2.imread(os.path.join(self.__final_license_plate_dir , image))
            # Postprocessing images here 
            license_plate_gray = cv2.cvtColor(cv_img_org, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f'grayscale/{image}' , license_plate_gray)
            #denoise_img=cv2.fastNlMeansDenoising(license_plate_gray, None, 20, 7, 21)
            denoise_img= None
            for i in range(0 , 1):
                if(i==0):
                    denoise_img= cv2.fastNlMeansDenoising(license_plate_gray, None, 20, 7, 21)
                else:
                    denoise_img= cv2.fastNlMeansDenoising(denoise_img, None, 20, 7, 21) 
            cv2.imwrite(f'gray_denoise/{image}' , denoise_img)

            _,cv_img = cv2.threshold(denoise_img, 127, 255, cv2.THRESH_BINARY_INV)   
            cv2.imwrite(f'gray_denoise_thresh/{image}' , cv_img)        
           # _, cv_img = cv2.threshold(denoise_img, 64, 255, cv2.THRESH_BINARY_INV)


            output = self.__reader.readtext(cv_img)
            scores =[]
            texts=[]
            
            try:


                # for result in output:
                #     result[1]= result[1][::-1]

                # output = sorted(output , keys = lambda r: r[0][0])

                for out in output:           
                    text_bbox, text, text_score = out
                    texts.append(text)
                    scores.append(text_score)

                  
                if max(scores) >= 0:    
                    print(image)
                    if image in self.__image_dict:
                        actual_lp = self.__image_dict[image]
                        generated_lp=self._prettified_text(''.join(texts))   
                        custom_v3_acc = self._custom_accuracy_v3(actual_lp , generated_lp)
                        
                        print("Actual lp " , actual_lp) 
                        print("Text and score using custom_accuracy_v3 is " ,actual_lp ,  generated_lp, str(custom_v3_acc)+"%")        
                        # print("Text and score value is" ,actual_lp , generated_lp, str(max(scores)*100)+'%')
                        # print("Text and score value with custom_acc v1 is" ,actual_lp , generated_lp, self._custom_accuracy_v1(actual_lp , generated_lp))
                        # print("Text and score value with custom_acc v2 is" ,actual_lp , generated_lp, self._custom_accuracy_v2(actual_lp , generated_lp))
                        if custom_v3_acc >= 95:
                            total_accuracy = total_accuracy + custom_v3_acc
                            total_images = total_images + 1
                            cv2.imwrite(f"0.5accurate/{image}" , cv_img);
                          
                            acc[image]= {"accuracy":self._custom_accuracy_v3(actual_lp , generated_lp) , "generated_num":generated_lp , "actual_num":actual_lp}
                    else:
                        print("Image not annotated , Please annotate the image")    
            except IndexError as ie:
                print("Index error occured" , ie)
            except TypeError as te:
                print("Type error " , te)
            except ValueError as ve:
                print("Value error" , ve)
            except ZeroDivisionError as zde:
                print("Zerro Division error" , zde)    
            except ArithmeticError as ae:
                print("Arithmetic err" , ae)
            except:
                print("General Error no idea"); 
                       

        try:
            print("Total Average accuracy is " , total_accuracy/total_images)
        except:
            print("Exception occured in image extraction")          
        #dump here
        with open("accuracy.json", 'w') as file_object:  #open the file in write mode
            json.dump(acc, file_object)              
















       
