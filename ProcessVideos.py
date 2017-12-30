import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip

from CameraManager import CameraManager
from LaneLine import LaneLine
from FrameProcessor import FrameProcessor
from ImageClassifier import ImageClassifier
from ImageEngine import ImageEngine

import keras
from darkflow.net.build import TFNet

import multiprocessing

def createTFNet():
    global tfnet
    try:
        tfnet
    except:
        tfnet = TFNet({"config":"./config", "model": "./config/yolo.cfg", "load": "weights/yolo.weights", "threshold": 0.3})

def process_image(img):
    global num_frames_global
    
    num_frames_global += 1
    processor.processFrame(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    #return cv2.cvtColor(processor.visLaneDetectImage, cv2.COLOR_BGR2RGB)
    #return cv2.cvtColor(processor.visBinaryImage, cv2.COLOR_BGR2RGB)
    #return cv2.cvtColor(processor.visHeatMapImage, cv2.COLOR_BGR2RGB)
    #return cv2.cvtColor(processor.visVehicleBoxesImage, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(processor.visImageAnnotated, cv2.COLOR_BGR2RGB)

def process_video(file_name, sub_clip_from = 0, sub_clip_to = 0, visualization = False):
    global num_frames_global
    num_frames_global = 0

    camera = CameraManager('center')
    camera.initPerspectiveTransformation(srcPoints, dstPoints, dtsPlaneSizePx, dtsPlaneSizeM)

    leftLane = LaneLine()
    rightLane = LaneLine()

    imgEng = ImageEngine(load_setup = True)
    clfLinearSVC = ImageClassifier('clf_linear')
    clfSVC = ImageClassifier('clf_rbf')
    clfCNN = keras.models.load_model('./config/clf_cnn.dat')

    createTFNet()

    global processor
    processor = FrameProcessor(
        camera = camera,
        leftLane = leftLane,
        rightLane = rightLane,
        imageEngine = imgEng,
        baseWindowSize = baseWindowSize,
        detectionRegions = detectionRegions,
        classifierFast = clfLinearSVC,
        classifierAccurate = clfSVC,
        classifierCNN = clfCNN,
        classifierDarkFlow = tfnet,
        visualization = visualization,
        heatMapFrames = 5,
        heatMapThreshold = 10,
        heatMapTotalMin = 500,
        heatMapTotalMinFrame = 100)

    v_clip = VideoFileClip(input_dir_path + file_name)
    if sub_clip_to > 0:
        v_clip = v_clip.subclip(sub_clip_from, sub_clip_to)

    white_clip = v_clip.fl_image(process_image)
    white_clip.write_videofile(output_dir_path + file_name, audio=False)
    print("Video is processed. Frames: {0}.".format(num_frames_global))
    return

input_dir_path = "./test_videos/"
output_dir_path = "./test_videos_output/"

try:
    os.makedirs(output_dir_path)
except:
    pass

# Number of parallel threads
threads_num = 1

# Perspective transformation parameters
srcPoints = [[246, 700], [570, 468], [715, 468], [1079, 700]]
dstPoints = [[400, 719], [400, 0], [1200, 0], [1200, 719]]
dtsPlaneSizePx = (720, 1600)
dtsPlaneSizeM = (30.0, 7.4)

# Sliding window size
baseWindowSize = 64
# Set of detection regions and feature sizes used for vehicle detection
detectionRegions = [
    #[(360, 0), (445, 1280), (32,), (0.5, 0.5)],
    [(360, 0), (445, 1280), (48,), (2.0/3.0, 2.0/3.0)],
    [(360, 0), (490, 1280), (64,), (0.75, 0.75)],
    [(360, 0), (655, 1280), (96,), (5.0/6.0, 5.0/6.0)],
    [(360, 0), (655, 1280), (128,), (0.75, 0.75)],
    [(360, 0), (655, 1280), (192,), (5.0/6.0, 5.0/6.0)],
    ]

if __name__ == '__main__':
    #args = ["test_video.mp4", "project_video.mp4", "challenge_video.mp4", "harder_challenge_video.mp4"]
    
    args = ["project_video.mp4"]

    if threads_num > 1:
        pool = multiprocessing.Pool(processes = threads_num)
        pool.map(process_video, args)
        pool.close()
    else:
        for file_name in args:
            process_video(file_name)

        #process_video("test_video.mp4")

        #process_video("project_video.mp4", sub_clip_from = 27, sub_clip_to = 31)

        #process_video("challenge_video.mp4")