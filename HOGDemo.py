import numpy as np
import cv2
import matplotlib.pyplot as plt
from ImageEngine import ImageEngine

# Normalize image channel
def normalize_channel(channel):
    channel[channel < 0] = 0
    channel[channel > 255] = 255
    channel = np.uint8(channel)
            
    return channel

# Get images for HOG feature visualization
def getHOGVis(img):
    imgEng = ImageEngine(hog_channels = [0,1,2])

    #channel0 = normalize_channel(np.array(img[:, :, 0], dtype = np.int32) + np.array(img[:, :, 1], dtype = np.int32) - np.array(img[:, :, 2], dtype = np.int32))
    #channel1 = normalize_channel(255 - img[:, :, 1])
    #channel2 = normalize_channel(img[:, :, 2])
    #img = np.dstack((channel0, channel1, channel2))
    
    img = imgEng.convertColorSpace(img, tgtColorSpace = 'YUV')

    features, visualised = imgEng.getImageFeatures(img, visualise = True)

    return img, visualised

# Visualize
img1 = cv2.imread('./img_data/vehicles/GTI_MiddleClose/image0190.png')
#img1 = cv2.imread('./test_images/test05.png')
img1, visualised1 = getHOGVis(img1)

img2 = cv2.imread('./img_data/non_vehicles/Extras/extra251.png')
#img2 = cv2.imread('./test_images/test07.png')
img2, visualised2 = getHOGVis(img2)

f, graph_matr = plt.subplots(3, 4, figsize=(20,20))

graph_matr[0, 0].set_title('V {}'.format(0), fontsize = 30)
graph_matr[0, 1].set_title('HOG V {}'.format(1), fontsize = 30)
graph_matr[0, 2].set_title('N-V {}'.format(2), fontsize = 30)
graph_matr[0, 3].set_title('HOG N-V {}'.format(3), fontsize = 30)

for i in range(3):
    graph_matr[i, 0].set_title('Vh P:{}'.format(i), fontsize = 30)
    graph_matr[i, 0].imshow(img1[:,:,i], cmap='gray')
    graph_matr[i, 1].set_title('HOG Vh P:{}'.format(i), fontsize = 30)
    graph_matr[i, 1].imshow(visualised1[0][:,:,i], cmap='gray')
    graph_matr[i, 2].set_title('N-Vh P:{}'.format(i), fontsize = 30)
    graph_matr[i, 2].imshow(img2[:,:,i], cmap='gray')
    graph_matr[i, 3].set_title('HOG N-Vh P:{}'.format(i), fontsize = 30)
    graph_matr[i, 3].imshow(visualised2[0][:,:,i], cmap='gray')
plt.show()
#f.savefig('./test_images_output/hog_sample.png')
