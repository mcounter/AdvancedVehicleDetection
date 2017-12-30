import numpy as np
import os
import pickle
import cv2
import skimage.feature

class ImageEngine:
    """
    Responsible for work with images, including feature vectors calculation.
    """

    def __init__(
        self,
        load_setup = True, # Load configuration from file if exists
        config_file = './config/image_eng.dat', # Path to configuration file
        color_space = 'YUV', # Color space used: BGR, RGB, HSV, HLS, LUV, YUV, YCrCb
        hog_features = True, # Include HOG features
        hog_channels = [0,2], # Calculate HOG features for selected channels
        hog_orientations = 15, # Number of orientations
        hog_pix_per_cell = 8, # Pixels per cell
        hog_cells_per_block = 2, # Cells per block
        hog_block_norm = 'L2-Hys', # HOG block normalization function
        hog_transform_sqrt = True, # HOG do SQRT transformation first
        spatial_features = True, # Include spatial features
        spatial_channels = [0,1,2], # Calculate spatial features for selected channels
        spatial_size = 32, # Spatial feature size
        histogram_features = True, # Include histogram features
        histogram_channels = [0,1,2], # Calculate histogram features for selected channels
        histogram_bins = 64, # Size of histogram feature
        ):
        """
        Initialize class instance
        """
        
        self.config_file = config_file

        self.color_space = color_space

        self.hog_features = hog_features
        self.hog_channels = hog_channels
        self.hog_orientations = hog_orientations
        self.hog_pix_per_cell = hog_pix_per_cell
        self.hog_cells_per_block = hog_cells_per_block
        self.hog_block_norm = hog_block_norm
        self.hog_transform_sqrt = hog_transform_sqrt

        self.spatial_features = spatial_features
        self.spatial_channels = spatial_channels
        self.spatial_size = spatial_size

        self.histogram_features = histogram_features
        self.histogram_channels = histogram_channels
        self.histogram_bins = histogram_bins

        if (load_setup):
            self._unpickle_parameters()

    def _unpickle_parameters(self):
        """
        Load parameters.
        """

        is_loaded = False

        try:
            with open(self.config_file, mode='rb') as f:
                data_set = pickle.load(f)
    
            self.color_space = data_set['color_space']

            self.hog_features = data_set['hog_features']
            self.hog_channels = data_set['hog_channels']
            self.hog_orientations = data_set['hog_orientations']
            self.hog_pix_per_cell = data_set['hog_pix_per_cell']
            self.hog_cells_per_block = data_set['hog_cells_per_block']
            self.hog_block_norm = data_set['hog_block_norm']
            self.hog_transform_sqrt = data_set['hog_transform_sqrt']

            self.spatial_features = data_set['spatial_features']
            self.spatial_channels = data_set['spatial_channels']
            self.spatial_size = data_set['spatial_size']

            self.histogram_features = data_set['histogram_features']
            self.histogram_channels = data_set['histogram_channels']
            self.histogram_bins = data_set['histogram_bins']

            is_loaded = True
        except:
            is_loaded = False

        return is_loaded

    def _pickle_parameters(self):
        """
        Save parameters.
        """

        is_saved = False

        try:
            file_dir = os.path.dirname(self.config_file)
            os.makedirs(file_dir)
        except:
            pass

        try:
            with open(self.config_file, mode='wb') as f:
                data_set = {
                    'color_space' : self.color_space,

                    'hog_features' : self.hog_features,
                    'hog_channels' : self.hog_channels,
                    'hog_orientations' : self.hog_orientations,
                    'hog_pix_per_cell' : self.hog_pix_per_cell,
                    'hog_cells_per_block' : self.hog_cells_per_block,
                    'hog_block_norm' : self.hog_block_norm,
                    'hog_transform_sqrt' : self.hog_transform_sqrt,

                    'spatial_features' : self.spatial_features,
                    'spatial_channels' : self.spatial_channels,
                    'spatial_size' : self.spatial_size,

                    'histogram_features' : self.histogram_features,
                    'histogram_channels' : self.histogram_channels,
                    'histogram_bins' : self.histogram_bins}

                pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)

            is_saved = True;
        except:
            is_saved = False

        return is_saved

    def saveParameters(self):
        return self._pickle_parameters()

    def loadParameters(self):
        return self._unpickle_parameters()

    def convertColorSpace(
        self,
        img, # Image in some color space
        srcColorSpace = 'BGR', # Source color space
        tgtColorSpace = 'RGB', # Traget color space
        ):
        """
        Convert between color spaces
        """

        if srcColorSpace == tgtColorSpace:
            return img

        if srcColorSpace == 'BGR':
            img_bgr = img
        elif srcColorSpace == 'RGB':
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif srcColorSpace == 'HSV':
            img_bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        elif srcColorSpace == 'HLS':
            img_bgr = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)
        elif srcColorSpace == 'LUV':
            img_bgr = cv2.cvtColor(img, cv2.COLOR_LUV2BGR)
        elif srcColorSpace == 'YUV':
            img_bgr = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        elif srcColorSpace == 'YCrCb':
            img_bgr = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
        else:
            raise Exception("Incorrect color space: {}".format(srcColorSpace))

        if tgtColorSpace == 'BGR':
            img_tgt = img_bgr
        elif tgtColorSpace == 'RGB':
            img_tgt = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif tgtColorSpace == 'HSV':
            img_tgt = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        elif tgtColorSpace == 'HLS':
            img_tgt = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
        elif tgtColorSpace == 'LUV':
            img_tgt = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LUV)
        elif tgtColorSpace == 'YUV':
            img_tgt = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        elif tgtColorSpace == 'YCrCb':
            img_tgt = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        else:
            raise Exception("Incorrect color space: {}".format(tgtColorSpace))

        return img_tgt

    def calcHOGFeatures(
        self,
        img, # Image in some color space
        orientations, # Number of orientations
        pix_per_cell, # Pixels per cell
        cell_per_block, # Cells per block
        block_norm = 'L2-Hys', # Block normalization function
        transform_sqrt = False, # Do SQRT transformation first
        visualise = False # Visualise HOG features
        ):
        """
        Calculate HOG feature matrix based on parameters
        """

        res = skimage.feature.hog(
            img,
            orientations = orientations,
            pixels_per_cell = (pix_per_cell, pix_per_cell),
            cells_per_block = (cell_per_block, cell_per_block),
            block_norm = block_norm,
            transform_sqrt = transform_sqrt,
            visualise = visualise,
            feature_vector = False)

        return res

    def getImageFeatures(
        self,
        img, # Image in OpenCV BGR color space
        img_window = ((None, None), (None, None)), # Image window to analyze - ((y1, x1), (y2, x2)) - top left and bottom right image corners +1. Default - whole image
        featureSize = (None, None), # Feature size (height, width). Default - whole image = one feature.
        featureScale = 1.0, # Feature scale - in case < 1 - image will be increased in size, > 1 - decreased.
        overlap = (0.0, 0.0), # Overlap between feature windows (y, x) in range [0, 1)
        visualise = False, # Visualise HOG features
        imgCNN = [], # Image for CNN feature (must be same size as img)
        ):
        """
        Extract feature vectors for whole image
        """

        # Normalize one of image channel
        def normalize_channel(channel):
            channel[channel < 0] = 0
            channel[channel > 255] = 255
            channel = np.uint8(channel)
            
            return channel

        img_shape = img.shape
        img_window = [img_window[0][0], img_window[0][1], img_window[1][0], img_window[1][1]]

        if img_window[0] is None:
            img_window[0] = 0
        else:
            img_window[0] = max(0, img_window[0])

        if img_window[1] is None:
            img_window[1] = 0
        else:
            img_window[1] = max(0, img_window[1])

        if img_window[2] is None:
            img_window[2] = img_shape[0]
        else:
            img_window[2] = min(img_shape[0], img_window[2])

        if img_window[3] is None:
            img_window[3] = img_shape[1]
        else:
            img_window[3] = min(img_shape[1], img_window[3])

        if len(imgCNN) <= 0:
            img_cnn = []
        else:
            img_cnn = imgCNN[img_window[0]:img_window[2], img_window[1]:img_window[3]]

        img_yuv = img[img_window[0]:img_window[2], img_window[1]:img_window[3]]
       
        # Transformation between BGR color space and adjusted YUV (empirical approach)
        channel0 = normalize_channel(np.array(img_yuv[:, :, 0], dtype = np.int32) + np.array(img_yuv[:, :, 1], dtype = np.int32) - np.array(img_yuv[:, :, 2], dtype = np.int32))
        channel1 = normalize_channel(255 - img_yuv[:, :, 1])
        channel2 = normalize_channel(img_yuv[:, :, 2])
        img_yuv = np.dstack((channel0, channel1, channel2))

        img_yuv = self.convertColorSpace(img_yuv, tgtColorSpace = self.color_space)

        img_shape = img_yuv.shape

        # Resize image region to normalize sliding window size 
        if featureScale != 1.0:
            img_yuv = cv2.resize(img_yuv, (int(img_shape[1] / featureScale), int(img_shape[0] / featureScale)), interpolation = cv2.INTER_CUBIC)
            img_shape = img_yuv.shape
            if len(img_cnn) > 0:
                img_cnn = cv2.resize(img_cnn, (img_shape[1], img_shape[0]), interpolation = cv2.INTER_CUBIC)

        featureSize = [featureSize[0], featureSize[1]]

        if featureSize[0] is None:
            featureSize[0] = img_shape[0]

        if featureSize[1] is None:
            featureSize[1] = img_shape[1]

        if featureScale != 1.0:
            featureSizeSrc = [float(featureSize[0]) * featureScale, float(featureSize[1]) * featureScale]
        else:
            featureSizeSrc = [float(featureSize[0]), float(featureSize[1])]

        feature_stride = [featureSize[0], featureSize[1]]

        for idx in range(len(feature_stride)):
            if overlap[idx] > 0.0:
                feature_stride[idx] = int((1.0 - overlap[idx]) * featureSize[idx])

            if feature_stride[idx] <= 0:
                feature_stride[idx] = 1

            if self.hog_features:
                feature_stride[idx] = (feature_stride[idx] // self.hog_pix_per_cell) * self.hog_pix_per_cell
                if feature_stride[idx] < self.hog_pix_per_cell:
                    feature_stride[idx] = self.hog_pix_per_cell

        # Calculate HOG features for whole image region
        if self.hog_features:
            hog_feature_data_size = [(featureSize[0] // self.hog_pix_per_cell) - self.hog_cells_per_block + 1, (featureSize[1] // self.hog_pix_per_cell) - self.hog_cells_per_block + 1]

            hog_features = []
            
            if visualise:
                hog_vis = []

            for channel in self.hog_channels:
                res = self.calcHOGFeatures(
                    img_yuv[:,:,channel],
                    self.hog_orientations,
                    self.hog_pix_per_cell,
                    self.hog_cells_per_block,
                    block_norm = self.hog_block_norm,
                    transform_sqrt = self.hog_transform_sqrt,
                    visualise = visualise)

                if visualise:
                    hog_features += [res[0]]
                    hog_vis += [res[1]]
                else:
                    hog_features += [res]

        res_features = []
        if visualise:
            res_vis = []

        # Slide window across whole image region and extract separate feature vectors from each
        for y_pos in range(0, img_shape[0], feature_stride[0]):
            if featureScale != 1.0:
                y_pos_src = float(y_pos) * featureScale
            else:
                y_pos_src = float(y_pos)

            for x_pos in range(0, img_shape[1], feature_stride[1]):
                if featureScale != 1.0:
                    x_pos_src = float(x_pos) * featureScale
                else:
                    x_pos_src = float(x_pos)

                features_sub = []
                if visualise:
                    vis_sub = []

                # Extract spatial color feature vector
                if self.spatial_features:
                    sub_img = img_yuv[y_pos : (y_pos + featureSize[0]), x_pos : (x_pos + featureSize[1])]
                    sub_img_shape = sub_img.shape
                    if (sub_img_shape[0] != self.spatial_size) or (sub_img_shape[1] != self.spatial_size):
                        sub_img = cv2.resize(sub_img, (self.spatial_size, self.spatial_size), interpolation = cv2.INTER_CUBIC)
                        sub_img_shape = sub_img.shape

                        if (sub_img_shape[0] != self.spatial_size) or (sub_img_shape[1] != self.spatial_size):
                            continue

                    spatial_features_sub = []
                    for channel in self.spatial_channels:
                        if len(spatial_features_sub) <= 0:
                            spatial_features_sub = sub_img[:,:,channel].ravel()
                        else:
                            spatial_features_sub = np.hstack((spatial_features_sub, sub_img[:,:,channel].ravel()))

                    if len(features_sub) <= 0:
                        features_sub = spatial_features_sub
                    else:
                        features_sub = np.hstack((features_sub, spatial_features_sub))

                # Extract color histogram feature vector
                if self.histogram_features:
                    sub_img = img_yuv[y_pos : (y_pos + featureSize[0]), x_pos : (x_pos + featureSize[1])]

                    histogram_features_sub = []
                    for channel in self.histogram_channels:
                        cur_hist = np.histogram(sub_img[:,:,channel], bins=self.histogram_bins, range=(0, 256))
                         
                        if len(histogram_features_sub) <= 0:
                            histogram_features_sub = cur_hist[0].ravel()
                        else:
                            histogram_features_sub = np.hstack((histogram_features_sub, cur_hist[0].ravel()))

                    if len(features_sub) <= 0:
                        features_sub = histogram_features_sub
                    else:
                        features_sub = np.hstack((features_sub, histogram_features_sub))

                # Extract HOG feature sub-vector for each window, based on HOG matrix
                if self.hog_features:
                    y_pos_cells = y_pos // self.hog_pix_per_cell
                    x_pos_cells = x_pos // self.hog_pix_per_cell

                    hog_features_sub = []
                    
                    if visualise:
                        hog_vis_sub = []

                    isOK = True

                    for idx in range(len(hog_features)):
                        cur_feature = hog_features[idx][y_pos_cells : (y_pos_cells + hog_feature_data_size[0]), x_pos_cells : (x_pos_cells + hog_feature_data_size[1])]
                        cur_feature_shape = cur_feature.shape
                        if (cur_feature_shape[0] != hog_feature_data_size[0]) or (cur_feature_shape[1] != hog_feature_data_size[1]):
                            isOK = False
                            break

                        if len(hog_features_sub) <= 0:
                            hog_features_sub = cur_feature.ravel()
                        else:
                            hog_features_sub = np.hstack((hog_features_sub, cur_feature.ravel()))

                        if visualise:
                            cur_hog_vis = hog_vis[idx][y_pos : (y_pos + featureSize[0]), x_pos : (x_pos + featureSize[1])]
                            if len(hog_vis_sub) <= 0:
                                hog_vis_sub = cur_hog_vis
                            else:
                                hog_vis_sub = np.dstack((hog_vis_sub, cur_hog_vis))

                    if not isOK:
                        continue

                    if len(features_sub) <= 0:
                        features_sub = hog_features_sub
                    else:
                        features_sub = np.hstack((features_sub, hog_features_sub))

                    if visualise:
                        vis_sub = hog_vis_sub

                if len(img_cnn) > 0:
                    cnn_feature = img_cnn[y_pos : (y_pos + featureSize[0]), x_pos : (x_pos + featureSize[1])]
                    cnn_feature_shape = cnn_feature.shape
                    if (cnn_feature_shape[0] != featureSize[0]) or (cnn_feature_shape[1] != featureSize[1]):
                        cnn_feature = cv2.resize(cnn_feature, (featureSize[1], featureSize[0]), interpolation = cv2.INTER_CUBIC)
                        cnn_feature_shape = cnn_feature.shape

                        if (cnn_feature_shape[0] != featureSize[0]) or (cnn_feature_shape[1] != featureSize[1]):
                            continue

                # Combine features in one vector
                feature_vec = [(y_pos_src + img_window[0], x_pos_src + img_window[1]), (y_pos_src + img_window[0] + featureSizeSrc[0], x_pos_src + img_window[1] + featureSizeSrc[1]), features_sub]
                if len(img_cnn) > 0:
                    feature_vec += [cnn_feature]

                res_features += [feature_vec]

                if visualise:
                    res_vis += [vis_sub]

        if visualise:
            return res_features, res_vis

        return res_features
