from darkflow.net.build import TFNet
import cv2
import os
import matplotlib.pyplot as plt

import time

def processFrame(src_name, show_diagram = False):
    imgcv = cv2.imread(input_dir_path + src_name)
    result = tfnet.return_predict(imgcv)

    res_img = imgcv.copy()

    for detected_obj in result:
        object_label = detected_obj["label"]
        confidence = detected_obj["confidence"]
        box0 = (detected_obj["topleft"]["y"], detected_obj["topleft"]["x"])
        box1 = (detected_obj["bottomright"]["y"], detected_obj["bottomright"]["x"])
    
        annotation_color = (0, 0, 255)
        annotationWindowThickness = 3

        cv2.rectangle(res_img, box0[::-1], box1[::-1], annotation_color, annotationWindowThickness)

        cv2.putText(
            res_img,
            'Object {}: {:.1f}%'.format(object_label, confidence * 100.0),
            (box0[1] + 1, box0[0] - 10 + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            thickness = 1,
            lineType = cv2.LINE_AA)

        cv2.putText(
            res_img,
            'Object {}: {:.1f}%'.format(object_label, confidence * 100.0),
            (box0[1], box0[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            annotation_color,
            thickness = 1,
            lineType = cv2.LINE_AA)

    cv2.imwrite(output_dir_path + src_name, res_img)

    if show_diagram:
        res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        plt.imshow(res_img)
        plt.show()

def processImage(image_name):
    image_path = input_dir_path + image_name
    if os.path.exists(image_path) and os.path.isfile(image_path):
        t = time.time()
        print("Processing image {0} ...".format(image_name))
        processFrame(image_name)
        print("    Processed in {:.2f} sec.".format(time.time() - t))

    print()

#input_dir_path = "./sample_img/"
#output_dir_path = "./sample_img/output/"

input_dir_path = "./test_images/"
output_dir_path = "./test_images_output/darkflow_output/"

images_dir = None
try: images_dir = os.listdir(input_dir_path)
except: print("Cannot read list of images")

try: os.makedirs(output_dir_path)
except: pass

options = {"config":"./config", "model": "./config/yolo.cfg", "load": "weights/yolo.weights", "threshold": 0.4}

if __name__ == '__main__':
    if images_dir is not None:
        global tfnet
        tfnet = TFNet(options)

        for image_name in images_dir:
            processImage(image_name)