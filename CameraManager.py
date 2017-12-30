import numpy as np
import glob
import pickle
import cv2
import os
import matplotlib.pyplot as plt

class CameraManager:
    """
    Manage camera parameters
    """

    def __init__(
        self,
        instanceName, # Camrea instance name
        storageDir = './camera' # Location of camera parameters
        ):
        """
        Initialize camera instance
        """

        self.storageDir = storageDir
        self.instanceName = instanceName
        self.perspectiveMatr = None
        self.perspectiveMatrInv = None
        self.perspectivePlaneSizePx = None
        self.perspectivePlaneSizeM = None

        self._unpickle_parameters()

    def _unpickle_parameters(self):
        """
        Load camera parameters.
        """

        self.isCalibrated = False
        self.cameraMatrix = None
        self.distortionParam = None

        try:
            with open('{}/{}.dat'.format(self.storageDir, self.instanceName), mode='rb') as f:
                data_set = pickle.load(f)
    
            self.cameraMatrix = data_set['cam_matrix']
            self.distortionParam = data_set['cam_dist']
            self.isCalibrated = data_set['cam_calibrated']
        except:
            self.isCalibrated = False

        return self.isCalibrated

    def _pickle_parameters(self):
        """
        Save camera parameters.
        """

        try:
            os.makedirs(self.storageDir)
        except:
            pass

        with open('{}/{}.dat'.format(self.storageDir, self.instanceName), mode='wb') as f:
            data_set = {
                'cam_matrix' : self.cameraMatrix,
                'cam_dist' : self.distortionParam,
                'cam_calibrated' : self.isCalibrated}

            pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)

    def calibrateOnChessboard(
        self,
        imageDir, # Location of files for calibration
        imageNameMask, # Calibration file name mask (example: calibration*.jpg)
        sizeColsRows, # number of columns and rows on checkboard for calibration (cols, rows)
        minImagesNum = 4, # Minimal number success images for calibration
        colRowFlexibility = ((0, 0), (0, 0)), # Images can be captured not in full format, so need be more flexible
        whiteBorderSize = 0 # Add white border around image as recommended by OpenCV in complex cases
        ):
        """
        Calibrate camera on chessboard image set
        """

        print('Camera calibration:')

        self.isCalibrated = False
        self.cameraMatrix = None
        self.distortionParam = None

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        images = glob.glob('{}/{}'.format(imageDir, imageNameMask))

        print('    Images found: {}'.format(len(images)))

        images_success = 0
        images_error = 0

        if len(images) >= max(1, minImagesNum):
            # Prepare object points, like [[0, 0, 0], [1, 0 ,0], [2, 0, 0], ..., [sizeColsRows[0] - 1, sizeColsRows[1] - 1, 0]]
            objp = np.zeros((sizeColsRows[1], sizeColsRows[0], 3), np.float32)
            objp[:, :, :2] = np.mgrid[0:sizeColsRows[0], 0:sizeColsRows[1]].T.reshape(-1, sizeColsRows[0], 2)

            isFlexAdjustment = False
            min_col_size = sizeColsRows[0]
            min_row_size = sizeColsRows[1]

            # Step through the list and search for chessboard corners
            for idx, fname in enumerate(images):
                isSuccess = False

                try:
                    img = cv2.imread(fname)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # Find the chessboard corners
                    ret, corners = cv2.findChessboardCorners(gray, sizeColsRows, None)

                    if ret:
                        objpoints.append(objp)
                        imgpoints.append(corners.reshape(-1, sizeColsRows[0], 2))

                        isSuccess = True
                    
                    if not isSuccess:
                        if whiteBorderSize > 0:
                            img_bdr = cv2.copyMakeBorder(img, whiteBorderSize, whiteBorderSize, whiteBorderSize, whiteBorderSize, cv2.BORDER_CONSTANT, None, value = (255, 255, 255))
                            gray_bdr = cv2.cvtColor(img_bdr, cv2.COLOR_BGR2GRAY)

                            # Find the chessboard corners
                            ret, corners = cv2.findChessboardCorners(gray_bdr, sizeColsRows, None)

                            if ret:
                                objpoints.append(objp)
                                imgpoints.append(corners.reshape(-1, sizeColsRows[0], 2))

                                isSuccess = True

                    if not isSuccess:
                        for col_shift in range(colRowFlexibility[0][1], colRowFlexibility[0][0] - 1,  -1):
                            for row_shift in range(colRowFlexibility[1][1], colRowFlexibility[1][0] - 1, - 1):
                                if (col_shift != 0) or (row_shift != 0):
                                    # Find the chessboard corners
                                    ret, corners = cv2.findChessboardCorners(gray, (sizeColsRows[0] + col_shift, sizeColsRows[1] + row_shift), None)

                                    if (not ret) and (whiteBorderSize > 0):
                                        ret, corners = cv2.findChessboardCorners(gray_bdr, (sizeColsRows[0] + col_shift, sizeColsRows[1] + row_shift), None)

                                    if ret == True:
                                        objpoints.append(objp)
                                        imgpoints.append(corners.reshape(-1, sizeColsRows[0] + col_shift, 2))

                                        min_col_size = min(min_col_size, sizeColsRows[0] + col_shift)
                                        min_row_size = min(min_row_size, sizeColsRows[1] + row_shift)
                                        isFlexAdjustment = True
                                        isSuccess = True
                                        break;

                            if isSuccess:
                                break;
                except:
                    isSuccess = False

                if isSuccess:
                    images_success += 1
                else:
                    images_error += 1
                    print('    Image "{}" cannot be recognized.'.format(fname))

            print('    Successfully recognized: {}'.format(images_success))
            print('    Cannot recognize: {}'.format(images_error))

            # If minimal number of success images was detected, try to calibrate camera
            if images_success >= max(1, minImagesNum):
                # Not all images have been detected in normal way
                objpoints_adj = []
                imgpoints_adj = []

                for idx in range(len(imgpoints)):
                    objp = objpoints[idx]
                    imgpnt = imgpoints[idx]

                    objpoints_adj.append(objp[:min_row_size, :min_col_size].reshape(-1, 3))
                    imgpoints_adj.append(imgpnt[:min_row_size, :min_col_size].reshape(-1, 1, 2))
                        
                objpoints = objpoints_adj
                imgpoints = imgpoints_adj

                # Do camera calibration given object points and image points
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (min_col_size, min_row_size), None, None)

                self.cameraMatrix = mtx
                self.distortionParam = dist
                self.isCalibrated = True

        isSuccess = self.isCalibrated

        if isSuccess:
            self._pickle_parameters() # Save parameters in case of success

            print('    Calibration is finished successfully.')
        else:
            self._unpickle_parameters() # Try to restore parameters in case of error

            print('    Calibration error.')

        return isSuccess

    def undistort(
        self,
        img # Source image
        ):
        """
        Image undistortion.
        """
        if not self.isCalibrated:
            raise Exception("Camera is not calibrated.")

        return cv2.undistort(img, self.cameraMatrix, self.distortionParam)

    def makeBinary(
        self,
        img, # Source image in OpenCV BGR color space
        blurKernel = 5, # Gaussian blur kernel size
        sobelKernel = 5, # Kernel size for sobel operator
        saturationThr = (150, 50, 50), # Saturation threshold (for all 3 channels, but first 2 are used now)
        gradThr = (30, 30, 30), # Gradient threshold (for all 3 channels, but first 2 are used now)
        gradDirectionThr = (0.3, 0.3, 0.3), # Gradient direction threshold (for all 3 channels, but first 2 are used now)
        undistort = False # Undistort first
        ):
        """
        Apply binary filters to the image
        """

        # Normalize one of image channel
        def normalize_channel(channel):
            channel[channel < 0] = 0
            channel[channel > 255] = 255

            channel_min = np.min(channel)

            if channel_min != 0:
                channel = channel - channel_min

            channel_max = np.max(channel)
            if (channel_max != 0) and (channel_max != 255):
                channel = np.uint8(255.0 * channel / channel_max)
            else:
                channel = np.uint8(channel)
            
            return channel

        # Calculate gradient magnitude and direction for one of image channel
        def get_channel_gradient(channel):
            grad_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize = sobelKernel)
            grad_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize = sobelKernel)
            grad = np.sqrt(np.power(grad_x, 2) + np.power(grad_y, 2))

            # Scale gradient to [0..255] range
            grad_max = np.max(grad)
            if grad_max != 0:
                grad = np.uint8(255.0 * grad / grad_max)
            else:
                grad = np.uint8(grad)

            grad_dir = np.arctan2(np.absolute(grad_x), np.absolute(grad_y))

            return grad, grad_dir

        if undistort:
            img = self.undistort(img)

        # Convert in YUV color space
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        ############### Channel Y:

        # Correct with secondary channel
        y_channel = normalize_channel(np.array(img_yuv[:, :, 0], dtype = np.int32) + np.array(img_yuv[:, :, 1], dtype = np.int32) - np.array(img_yuv[:, :, 2], dtype = np.int32))
        # Bluring
        y_channel = normalize_channel(cv2.GaussianBlur(y_channel, (blurKernel, blurKernel), 0))

        # Do more contrast
        for i in range(3):
            y_channel[1:-1, 1:-1] = np.maximum(np.maximum(np.maximum(y_channel[:-2, 1:-1], y_channel[2:,1:-1]), y_channel[1:-1, :-2]), y_channel[1:-1, 2:])

        # Calculate gradient
        y_grad, y_grad_dir = get_channel_gradient(y_channel)
        
        # Filter by saturation level
        y_filter = y_channel < saturationThr[0]

        # Expand filter
        for i in range(2):
            y_filter[1:,:] = y_filter[1:,:] | y_filter[:-1,:]
            y_filter[:,1:] = y_filter[:,1:] | y_filter[:,:-1]

        # Apply filter
        y_grad[y_filter] = 0

        # Filter by gradient magnitude and direction
        y_grad[(y_grad <= gradThr[0]) | (y_grad_dir < gradDirectionThr[0])] = 0
        y_grad[y_grad > 0] = 255

        # Do result thicker
        for i in range(3):
            y_grad[1:-1, 1:-1] = np.maximum(np.maximum(np.maximum(y_grad[:-2, 1:-1], y_grad[2:,1:-1]), y_grad[1:-1, :-2]), y_grad[1:-1, 2:])

        ############### Channel U:

        # Reverse channel to be similar with channel Y
        u_channel = normalize_channel(255 - img_yuv[:, :, 1])
        # Bluring
        u_channel = normalize_channel(cv2.GaussianBlur(u_channel, (blurKernel, blurKernel), 0))

        # Do more contrast
        for i in range(1):
            u_channel[1:-1, 1:-1] = np.maximum(np.maximum(np.maximum(u_channel[:-2, 1:-1], u_channel[2:,1:-1]), u_channel[1:-1, :-2]), u_channel[1:-1, 2:])

        # Calculate gradient
        u_grad, u_grad_dir = get_channel_gradient(u_channel)

        # Filter by saturation level
        u_filter = u_channel < saturationThr[1]

        # Expand filter
        for i in range(2):
            u_filter[1:,:] = u_filter[1:,:] | u_filter[:-1,:]
            u_filter[:,1:] = u_filter[:,1:] | u_filter[:,:-1]

        # Apply filter
        u_grad[u_filter] = 0

        # Filter by gradient magnitude and direction
        u_grad[(u_grad <= gradThr[1]) | (u_grad_dir < gradDirectionThr[1])] = 0
        u_grad[u_grad > 0] = 255

        # Do result thicker
        for i in range(1):
            u_grad[1:-1, 1:-1] = np.maximum(np.maximum(np.maximum(u_grad[:-2, 1:-1], u_grad[2:,1:-1]), u_grad[1:-1, :-2]), u_grad[1:-1, 2:])

        # Combine results for Y and U layers and make binary output
        binary_output = (y_grad >= 128) | (u_grad >= 128)

        
        # For test purpose only

        #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        #ax1.imshow(y_channel, cmap="gray")
        #ax1.set_title('Original Image', fontsize = 30)
        #ax2.imshow(y_grad, cmap="gray")
        #ax2.set_title('Processed Image', fontsize = 30)
        #plt.show()

        #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        #ax1.imshow(u_channel, cmap="gray")
        #ax1.set_title('Original Image', fontsize = 30)
        #ax2.imshow(u_grad, cmap="gray")
        #ax2.set_title('Processed Image', fontsize = 30)
        #plt.show()

        return binary_output

    def initPerspectiveTransformation(
        self,
        srcPoints, # Array of points on source plane (usually 4 points)
        dstPoints, # Array of point on destination plane (usually 4 points)
        dtsPlaneSizePx, # Size of destination plane in pixels
        dtsPlaneSizeM, # Size of destination plane in meters
        ):
        """
        Initialize perspective transformation matrices and parameters.
        """

        srcPoints_np = np.array(srcPoints, dtype = np.float32)
        dstPoints_np = np.array(dstPoints, dtype = np.float32)
        self.perspectiveMatr = cv2.getPerspectiveTransform(srcPoints_np, dstPoints_np)
        self.perspectiveMatrInv = cv2.getPerspectiveTransform(dstPoints_np, srcPoints_np)
        
        self.perspectivePlaneSizePx = dtsPlaneSizePx
        self.perspectivePlaneSizeM = dtsPlaneSizeM

    def perspectiveTransform(
        self,
        img # Source image
        ):
        """
        Apply perspective transformation to the image
        """

        if (self.perspectiveMatr is None) or (self.perspectivePlaneSizePx is None):
            raise Exception("Perspective transformation is not initialized.")

        return cv2.warpPerspective(img, self.perspectiveMatr, (self.perspectivePlaneSizePx[1], self.perspectivePlaneSizePx[0]))

    def binaryToImg(
        self,
        binary # Binary array
        ):
        """
        Convert binary matrix to image
        """

        img = np.zeros((binary.shape[0], binary.shape[1], 3), np.uint8)
        img[binary] = 255
        
        return img

    def imgToBinary(
        self,
        img, # Image (gray or all channels are the same)
        threshold = 128 # Threshold value
        ):
        """
        Convert image to binary matrix
        """

        binary = img[:, :, 0] >= threshold
        
        return binary

    def perspectiveTransformBinary(
        self,
        binary # Binary array for transformation
        ):
        """
        Apply perspective transformation to the image
        """

        img = self.binaryToImg(binary)
        img_res = self.perspectiveTransform(img)

        return self.imgToBinary(img_res)

    def perspectiveTransformPoints(
        self,
        points, # Array of points
        inverse = False, # Inverse transformation
        ):
        """
        Apply perspective transformation to array of points
        """

        if inverse:
            matrix = self.perspectiveMatrInv
        else:
            matrix = self.perspectiveMatr

        points_ext = points.T
        points_ext = np.append(points_ext, [np.ones_like(points_ext[0])], axis = 0)
        
        point_res = np.matmul(matrix, points_ext)
        point_res[0] /= point_res[2]
        point_res[1] /= point_res[2]
        return point_res[0:2].T

    def getWorldCorrection(self):
        return (float(self.perspectivePlaneSizeM[0]) / float(self.perspectivePlaneSizePx[0]), float(self.perspectivePlaneSizeM[1]) / float(self.perspectivePlaneSizePx[1]))
