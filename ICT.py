from sklearn.neighbors import KDTree
import numpy as np
import os 
import cv2


dir_path = '/Users/krishnamehta/Desktop/Image Color Transfer/NewDir'
image_path = '/Users/krishnamehta/Desktop/Image Color Transfer/IMG_2559.JPG' 

def get_image_props(path):
    image_rgb = cv2.imread(path)
    h, w = image_rgb.shape[:2]
    return image_rgb, h, w


def get_color_palette(path):
    
    files = os.listdir(path)
    print(files)
    
    image_paths=[]
    for i in files:
        path = os.path.join(dir_path,i)
        image_paths.append(path)

    unique_colors = []
    for img in image_paths:
        image, h, w = get_image_props(img)
        full_size = w * h
       # print([w,h], full_size)
        for i in range(w):
            for j in range(h):
                B,G,R = image[j,i]
                #R, G, B  = str(R).zfill(3), str(G).zfill(3), str(B).zfill(3)
                #rgb_arr = " ".join([str(R).zfill(3), str(G).zfill(3), str(B).zfill(3)])
                unique_colors.append([int(B), int(G), int(R)])
            
        #print("Total Unique Colors: ", len(unique_colors))
        print(f"image {img} complete")
    
    color_palette = np.array(unique_colors)
    unique_palette = np.unique(color_palette, axis=0)
    return unique_palette


def get_image_rgb(path):

    imageTC, h, w = get_image_props(path) 
    imageTC_rgb = []
    for i in range(w):
        rgb_row = [] 
        for j in range(h):
            B,G,R = imageTC[j, i]
            #R, G, B = str(R).zfill(3), str(G).zfill(3), str(B).zfill(3) 
            #rgb_str2 = " ".join([str(R).zfill(3), str(G).zfill(3), str(B).zfill(3)])
            rgb_row.append([int(B), int(G), int(R)])
        imageTC_rgb.append(rgb_row)

    image_2np =np.array(imageTC_rgb) 
    return image_2np 
  


colors = get_color_palette(dir_path)
og_image = get_image_rgb(image_path)
og_img_shape = og_image.shape

threed_twod = og_image.reshape((og_img_shape[0] * og_img_shape[1]), og_img_shape[2])

'''
for i in range(len(colors)):
    if colors[i] in threed_twod:
'''        

#print(colors.shape)
#print( og_image, threed_twod)
A = colors
B = threed_twod

tree = KDTree(A, leaf_size=40)

dist, ind = tree.query(B, k=1)

print(dist,ind)

new_image_map = []

for i in ind:
    for j in i:
        new_image_map.append(colors[i])

new_image_map = np.array(new_image_map)

output_image = new_image_map.reshape(og_img_shape)

print(output_image.dtype)

final_image = output_image.astype(np.uint8) 
final_img_rotate = cv2.rotate(final_image, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow("RGB Image", final_img_rotate)
cv2.waitKey(0)
cv2.destroyAllWindows()
