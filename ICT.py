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


#gets unique colors to train on
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
        for i in range(w):
            for j in range(h):
                B,G,R = image[j,i]
                unique_colors.append([B, G, R])
        print("Total Unique Colors: ", len(unique_colors))
        print(f"image {img} complete")
    
    color_palette = np.array(unique_colors)
    unique_palette = np.unique(color_palette, axis = 0)
    return unique_palette


## converts image to edit to BGR codes
def get_image_rgb(path):

    imageTC, h, w = get_image_props(path) 
    imageTC_bgr = []
    for i in range(w):
        bgr_row = [] 
        for j in range(h):
            B,G,R = imageTC[j, i]
            bgr_row.append([B, G, R])
        imageTC_bgr.append(bgr_row)

    image_2np = np.array(imageTC_bgr) 
    return image_2np 
  


colors = get_color_palette(dir_path)
og_image = get_image_rgb(image_path)
print(colors,colors.shape) 
print(og_image, og_image.shape)



'''
og_img_shape = og_image.shape

threed_twod = og_image.reshape((og_img_shape[0] * og_img_shape[1]), og_img_shape[2])

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
'''

