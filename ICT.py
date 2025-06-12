import numpy as np
import os 
import cv2


dir_path = '/Users/krishnamehta/Desktop/Image Color Transfer/Kodak Color Plus 200'
files = os.listdir(dir_path)

image_paths=[]
for i in files:
    path = os.path.join(dir_path,i)
    image_paths.append(path)


colors = []
for img in image_paths:
    image = cv2.imread(img);
    h, w = image.shape[:2]
    full_size = w * h
    print([w,h], full_size)
    for i in range(w):
        for j in range(h):
            B,G,R = image[j,i]
            colors.append([R,G,B])
    print([np.size(colors,0), np.size(colors,1)])
    print(f"image {img} complete")   
