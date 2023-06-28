import os
import json
import re
import xml.etree.ElementTree as ET
import cv2



folder_path = "C:\\Users\\dipta\\Downloads\\State-wise_OLX-20230415T144054Z-001\\State-wise_OLX"


path_number_dict={

}

count=0
# Sort and insert into the path_number_dict
def custom_sort(image_arr , state_name):
   global count
   number_arr = []
   count= count+len(image_arr)
   for path in image_arr:
      number_arr.append(int(re.findall(r'\d+' ,path )[0]))
   number_arr.sort()
   #print("the sortd arr is  "  , number_arr)
   for number in number_arr:
      file_name = state_name + str(number)+".jpg"
      path_number_dict[file_name]=0

for folder in os.listdir(folder_path):
    #print(folder)
    image_arr = []
    for image in os.listdir(os.path.join(folder_path , folder)):
        if not image.endswith(".xml"):
            # print(image)
            image_arr.append(image)
    custom_sort(image_arr , folder)



for image_name in path_number_dict.keys():
    xml_file_name = image_name[0:2]
    if(xml_file_name in os.listdir(folder_path)):
       for file in os.listdir(os.path.join(folder_path , xml_file_name)):
          if not file.endswith(".jpg") and file==image_name.split(".")[0]+'.xml':
             tree= ET.parse(os.path.join(folder_path , xml_file_name , file))
             root = tree.getroot()
             path_number_dict[image_name]=root[6][0].text
                

        
with open("path_number.json", 'w') as file_object:  #open the file in write mode
 json.dump(path_number_dict, file_object) 

# for folder in os.listdir(folder_path):
#     for image in os.listdir(os.path.join(folder_path , folder)):
#         print(image)
#         if image.endswith(".jpg"):
#             x=cv2.imread(os.path.join(folder_path , folder , image))
#             cv2.imwrite(f"C:\\Users\\dipta\\Downloads\\State-wise_OLX-20230415T144054Z-001\\Destructure_States\\{image}",x)
