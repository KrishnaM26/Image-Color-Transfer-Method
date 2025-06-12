import numpy as np
import os 
import cv2


dir_path = '/Users/krishnamehta/Desktop/Image Color Transfer/Kodak Color Plus 200'
files = os.listdir(dir_path)

image_paths=[]
for i in files:
    path = os.path.join(dir_path,i)
    image_paths.append(path)


unique_colors = set()
for img in image_paths:
    image = cv2.imread(img);
    h, w = image.shape[:2]
    full_size = w * h
    print([w,h], full_size)
    for i in range(w):
        for j in range(h):
            B,G,R = image[j,i]
            rgb_arr = " ".join([str(R).zfill(3), str(G).zfill(3), str(B).zfill(3)])
            unique_colors.add(rgb_arr)

    print("Total Unique Colors: ", len(unique_colors))
    print(f"image {img} complete")   

#print(unique_colors)

image_to_convert_path = '/Users/krishnamehta/Desktop/Image Color Transfer/IMG_2559.JPG' 
imageTC = cv2.imread(image_to_convert_path)
'''
h, w = image.shape[:2]

imageTC_rgb = []

for i in range(w):
    rgb_row = [] 
    for j in range(h):
        B,G,R = imageTC[j,i]
        R1, G1, B1 = str(R).zfill(3), str(G).zfill(3), str(B).zfill(3)
        rgb_row.append([int(R1), int(G1), int(B1)])
    imageTC_rgb.append(rgb_row)
   '''     
img_rgb = cv2.cvtColor(imageTC, cv2.COLOR_BGR2RGB)
cv2.imshow("RGB Image", img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
#print(imageTC_rgb)
