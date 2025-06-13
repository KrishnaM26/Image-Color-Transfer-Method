import numpy as np
import os 
import cv2


dir_path = '/Users/krishnamehta/Desktop/Image Color Transfer/Kodak Color Plus 200'
image_path = '/Users/krishnamehta/Desktop/Image Color Transfer/IMG_2559.JPG' 


def get_color_palette(path):
    
    files = os.listdir(path)
    
    image_paths=[]
    for i in files:
        path = os.path.join(dir_path,i)
        image_paths.append(path)

    unique_colors = set()
    for img in image_paths:
        image = cv2.imread(img)
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

    color_palette = np.array(list(unique_colors))
    return color_palette


def get_image_rgb(path):

    imageTC = cv2.imread(path)
    h, w = imageTC.shape[:2]

    imageTC_rgb = []
    for i in range(w):
        rgb_row = [] 
        for j in range(h):
            B,G,R = imageTC[j, i]
            R, G, B = str(R).zfill(3), str(G).zfill(3), str(B).zfill(3) 
            #rgb_str2 = " ".join([str(R).zfill(3), str(G).zfill(3), str(B).zfill(3)])
            rgb_row.append([int(R), int(G), int(B)])
        imageTC_rgb.append(rgb_row)
    return np.array(imageTC_rgb)
        
'''
# Displays Image #
cv2.imshow("RGB Image", np.array(imageTC_rgb))
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# print(get_image_rgb(image_path))
print(get_color_palette(dir_path))
