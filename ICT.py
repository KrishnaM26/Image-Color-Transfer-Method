from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
import numpy as np
import colorsys
import os 
import cv2


dir_path = '/Users/krishnamehta/Desktop/Image Color Transfer/newDir7'
#image_path = '/Users/krishnamehta/Desktop/Image Color Transfer/IMG_2559.JPG'
image_path='/Users/krishnamehta/Desktop/Image Color Transfer/ICT_1.jpg'
download_dir = r'/Users/krishnamehta/Desktop/Image Color Transfer/img_download'
k=30
l=1000
add_noise = False 
#j=0.1

def get_image_props(path):
    image_rgb = cv2.imread(path)
    h, w = image_rgb.shape[:2]
    return image_rgb, h, w


#gets unique colors to train on
def get_color_palette(path):
    
    files = os.listdir(path)
    print(f'Training Data: {files}')
    
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
        print(f"image {img} training complete")
    
    color_palette = np.array(unique_colors)
    unique_palette = np.unique(color_palette, axis = 0)
    print("Total Unique Colors: ", len(unique_colors))
    #print(unique_palette)
    return unique_palette


## converts image to edit to BGR codes
def get_image_rgb(path, noise):

    #get image from path
    imageTC, h, w = get_image_props(path) 
    
    if noise:
        #add gaussian noise
        mean,std = 0, 25
        gaussian_noise = np.random.normal(mean, std, imageTC.shape).astype(np.float32)
        dst = cv2.add(imageTC.astype(np.float32), gaussian_noise)

        dst = np.clip(dst, 0, 255).astype(np.uint8)
        imageTC = dst

    imageTC_bgr = []
    for i in range(w):
        bgr_row = [] 
        for j in range(h):
            B,G,R = imageTC[j, i]
            bgr_row.append([B, G, R])
        imageTC_bgr.append(bgr_row)

    image_2np = np.array(imageTC_bgr) 
    return image_2np 
  

def get_lrange(arr, l):

    unique_colors = []
    for i in arr:
         B,G,R = i
         rgb = (R / 255.0, G/255.0, B/255.0) 
         H, L, S = colorsys.rgb_to_hls(*rgb)

         l_values = np.linspace(0.0, 1.0, l)

         for _, l_new in enumerate(l_values):
            r, g, b = colorsys.hls_to_rgb(H, l_new, S)
            unique_colors.append([int(b * 255), int(g*255), int(r*255)])

    color_palette = np.array(unique_colors)
    unique_palette_range = np.unique(color_palette, axis = 0)
    print('palette created!') 
    return unique_palette_range



colors = get_color_palette(dir_path)
og_image = get_image_rgb(image_path, add_noise)
#print(colors,colors.shape) 
#print(og_image, og_image.shape)

kmeans = KMeans(n_clusters = k, random_state = 0)
kmeans.fit(colors)

#labels = kmeans.labels_
centroids = kmeans.cluster_centers_
centroids_int = centroids.astype(int)




color_palette_range = get_lrange(centroids_int, l)

print('editing image...')

og_img_shape = og_image.shape

threed_twod = og_image.reshape((og_img_shape[0] * og_img_shape[1]), og_img_shape[2])
A = color_palette_range
B = threed_twod

tree = KDTree(A, leaf_size=40)

dist, ind = tree.query(B, k=1)

new_image_map = []
for i in ind:
    for j in i:
        new_image_map.append(A[i])

new_image_map = np.array(new_image_map)

output_image = new_image_map.reshape(og_img_shape)

final_image = output_image.astype(np.uint8) 
#print(final_image[1])
final_img_rotate = cv2.rotate(final_image, cv2.ROTATE_90_CLOCKWISE)
flip_image = cv2.flip(final_img_rotate, 1)

print('image complete.')

cv2.imshow("BGR Image", flip_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#os.chdir(download_dir)
#file_name = f'KCP_2-9_{k}_{l}_.jpg'
#cv2.imwrite(file_name, flip_image)
#print(f'image: {k}, {l}  saved')
