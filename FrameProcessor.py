import numpy as np
import cv2
import LaneLine
import scipy.ndimage.measurements

class FrameProcessor():
    """
    Frame processor - implement pipeline to detect lane lines in frame
    """

    def __init__(
        self,
        camera, # Camera instance
        leftLane, # LaneLine instance for left lane detection
        rightLane, # LaneLine instance for right lane detection
        imageEngine, # Image engine
        baseWindowSize, # Base window size for detection
        detectionRegions, # Detection regions list with elements: [(y1,x1), (y2,x2), (sz1, sz2, sz3), (ovlp_y, ovlp_x)], where (y1,x1), (y2,x2) - left top and bottom right coordinates of region, sz1, sz2,... - windowses used for detection, (ovlp_y, ovlp_x) - window overlap [0, 1)
        classifierFast = None, # Image classifier for initial classification
        classifierAccurate = None, # Image classifier for additional classification (can be None)
        classifierCNN = None, # CNN used for classification (can be None)
        classifierDarkFlow = None, # Darkflow classifier
        useDarkFlowMode = 2, # Use Darkflow scenarios: 0 - separate flow, 1 - combination of Darkflow and other detectors, 2 - Darkflow as detector
        useDarkFlowMult = 6.0, # Multiplier for Darkflow prebabilities
        useDarkFlowLabels = ["person", "bicycle", "car", "motorbike", "bus", "truck"], # Limited set of DarkFlow labels, other will be ignored
        visualization = False, # If this parameter is enabled - produce vizualization of frame processing step by step.
        detectorInitSize = (0.3, 0.3), # Window size to detect initial bottom position of lane line - (height, width)
        detectorWinSize = (20, 50, 180), # (Height, Bottom width, Top width) - convolution windows size
        detectorWinMarging = (100, 50, 180), # (Initial bottom, Bottom, Top) - ranges where convolution operation will be applied
        detectorEmptyThreshold = 0.1, # Threshold to recognize window as empty, range [0..1]
        detectorEmptySeq = 10, # Maximal number of consecutive empty windows to stop lane detection
        detectorMinWinPerLine = 8,  # Minimal number of windows must be detected for successfull recognition
        detectorMinApproxBoxes = 5, # Minimal number of windows detected to start lane approximation
        detectorMinApproxBoxesSq = 7, # Minimal number of windows detected to start curve approximation
        detectorMutualLaneShapeCorrection = False, # Use mutual lane shape correction or not
        detectorUsePreviousLine = True, # Use previous line shape as start point
        detectorWeightFactor = 4.0, # Detector weight factor - weight of bottom pixels in comparison to top pixels have always weight 1.0
        detectorMaxAllowedCurvDiff = 6.0, # Maximal allowed cuvrature difference between lines. This is logarithmic empirical value
        heatMapFrames = 1, # Number of historical frames used in heatmap, must be 1 for single images
        heatMapThreshold = 2, # Filter less number of matches
        heatMapTotalMin = 200, # Filter number of total heat points for correct detection
        heatMapTotalMinFrame = 200, # Filter number of total heat points for correct detection (one frame)
        heatMapEdgeThereshold = 0.75, # Parameter to detect multiple objects with overlap edges
        heatMapRegionThreshold = 0.5, # Parameter to recognize that some square region belongs to object
        heatMapConvolutionWindowSize = 64, # Convolution window size to split big region on horizontal sub-regions
        objsDetFrames = 7, # Number of frames where object will be tracked if possible
        objsDetAvgFrames = 7, # Number of frames where object frame will be averaged
        objsDetCrossFrameMaxDist = 48, # Max distance between objects in sequential frames
        objMergeThreshold = 0.6, # If some objects overlap, it can be merged if overlap percent exceed this parameter
        objHeightWidthLimit = 1.5, # Limit of height to width proportion
        annotationColorBGR = (0, 255, 0), # Annotation BGR color for space between lines
        annotationColorLineLeftBGR = (0, 0, 255), # Annotation BGR color for left line
        annotationColorLineRightBGR = (255, 0, 0), # Annotation BGR color for right line
        annotationColorTextBGR = (255, 255, 255), # Annotation BGR color for text
        annotationWeight = 0.3, # Transparence factor for annotation lanes and space between. In comparison to main image.
        annotationDelayFrames = 7, # Delay of annotation information refresh (in video frames)
        annotationBufferSize = 15, # Size of annotation buffer for averaging (in video frames)
        annotationWindowСolor = (255, 0, 0), # Annotation color of windows
        annotationWindowСolorList = [(0, 255, 0), (0, 0, 255), (255, 0, 255)], # List of colors to annotate tracked objects with color
        annotationWindowThickness = 3, # Annotation thickness of window
        annotationMaxDistanceFilterLR = 30, # Maximal distance filter left-right
        annotationMaxDistanceFilterF = 80, # Maximal distance filter forward
        ):
        
        self.camera = camera
        self.laneLines = (leftLane, rightLane)

        self.imageEngine = imageEngine
        self.classifierFast = classifierFast
        self.classifierAccurate = classifierAccurate
        self.classifierCNN = classifierCNN
        self.classifierDarkFlow = classifierDarkFlow
        self.useDarkFlowMode = useDarkFlowMode
        self.useDarkFlowMult = useDarkFlowMult

        self.useDarkFlowLabels = set()
        for cur_darkflow_label in useDarkFlowLabels:
            if len(cur_darkflow_label) > 0:
                self.useDarkFlowLabels.add(cur_darkflow_label)

        self.detectorInitSize = detectorInitSize
        self.detectorWinSize = detectorWinSize
        self.detectorWinMarging = detectorWinMarging
        self.detectorEmptyThreshold = detectorEmptyThreshold
        self.detectorEmptySeq = detectorEmptySeq
        self.detectorMinWinPerLine = detectorMinWinPerLine
        self.detectorMinApproxBoxes = detectorMinApproxBoxes
        self.detectorMinApproxBoxesSq = detectorMinApproxBoxesSq
        self.detectorMutualLaneShapeCorrection = detectorMutualLaneShapeCorrection
        self.detectorUsePreviousLine = detectorUsePreviousLine
        self.detectorWeightFactor = detectorWeightFactor
        self.detectorMaxAllowedCurvDiff = detectorMaxAllowedCurvDiff
        
        self.baseWindowSize = baseWindowSize
        self.detectionRegions = detectionRegions
        
        self.heatMapFrames = heatMapFrames
        self.heatMapThreshold = heatMapThreshold
        self.heatMapTotalMin = heatMapTotalMin
        self.heatMapTotalMinFrame = heatMapTotalMinFrame
        self.heatMapEdgeThereshold = heatMapEdgeThereshold
        self.heatMapRegionThreshold = heatMapRegionThreshold
        self.heatMapConvolutionWindowSize = heatMapConvolutionWindowSize

        self.objsDetFrames = objsDetFrames
        self.objsDetAvgFrames = objsDetAvgFrames
        self.objsDetCrossFrameMaxDist = objsDetCrossFrameMaxDist
        self.objMergeThreshold = objMergeThreshold
        self.objHeightWidthLimit = objHeightWidthLimit

        self.annotationColorBGR = annotationColorBGR
        self.annotationColorLineLeftBGR = annotationColorLineLeftBGR
        self.annotationColorLineRightBGR = annotationColorLineRightBGR
        self.annotationColorTextBGR = annotationColorTextBGR
        self.annotationWeight = annotationWeight
        self.annotationDelayFrames = annotationDelayFrames
        self.annotationBufferSize = annotationBufferSize

        self.annotationWindowСolor = annotationWindowСolor
        self.annotationWindowСolorList = annotationWindowСolorList
        self.annotationWindowThickness = annotationWindowThickness

        self.annotationMaxDistanceFilterLR = annotationMaxDistanceFilterLR
        self.annotationMaxDistanceFilterF = annotationMaxDistanceFilterF
        
        self.visualization = visualization
        self.visOrigImage = None
        self.visUndistortImage = None
        self.visBinaryImage = None
        self.visTopViewBinaryImage = None
        self.visLaneDetectImage = None

        self.visAllBoxesImage = None
        self.visVehicleBoxesImage = None
        self.visHeatMapImage = None
        
        self.isImageAnnotated = False
        self.visImageAnnotated = None
        self.annotationRadiusBuf = []
        self.annotationRadius = 0
        self.annotationCenterShiftBuf = []
        self.annotationCenterShift = 0
        self.annotationDelayCnt = -1

        self.regionHistory = []
        self.objectsHistory = []
        self.objLastLabel = 0

    def _detectLines(
        self,
        binary # Binary top view matrix 
        ):
        """
        Detect lane lines - main algorithm of lane line detector on binary top-view image
        """

        # Adjust windows size to not exceed image shape
        def make_win_adj(win):
            return (
                int(min(bin_shape[0], max(0, win[0]))),
                int(min(bin_shape[1], max(0, win[1]))),
                int(min(bin_shape[0], max(0, win[2]))),
                int(min(bin_shape[1], max(0, win[3]))))

        # Process windows - do vertical sum, convolve and calculate fulfillment of window
        def process_win(win):
            win_adj = make_win_adj(win)

            if (win_adj[3] - win_adj[1]) > conv_win_size:
                vert_sum = np.sum(binary[win_adj[0]:win_adj[2], win_adj[1]:win_adj[3]], axis=0)
                win_center = np.argmax(np.convolve(vert_sum, conv_win, mode = 'valid')) + win_adj[1] + conv_win_half
            else:
                win_center = (win_adj[1] + win_adj[3]) // 2

            win_adj = make_win_adj((win_adj[0], win_center - conv_win_half, win_adj[2], win_center + conv_win_half))
            win_sum = np.sum(binary[win_adj[0]:win_adj[2], win_adj[1]:win_adj[3]])
            win_sq = (win_adj[2] - win_adj[0]) * (win_adj[3] - win_adj[1])

            if win_sq > 0:
                win_pct = float(win_sum) / float(win_sq)
            else:
                win_pct = -1.0

            return win_adj, win_center, win_sum, win_pct

        # Convert set of windows to array of points these windows contain
        def combine_line_points(win_matrix, color_plane_bgr):
            line_points = []

            if len(win_matrix) >= self.detectorMinWinPerLine:
                for y1, x1, y2, x2 in win_matrix:
                    idx_grid = np.mgrid[y1:y2, x1:x2]
                    idx_filter = np.array(binary[y1:y2, x1:x2], dtype=bool)
                    points = idx_grid[:, idx_filter]
                    points_w = np.array(1 + np.around(points[0] / float(bin_shape[0]) * float(self.detectorWeightFactor - 1)), dtype=np.int32)
                    points = np.append(points, [points_w], axis=0).T

                    # Vizualize windows
                    if self.visualization:
                        self.visLaneDetectImage[y1:y2, x1:x2] = self.visLaneDetectImage[y1:y2, x1:x2] * 0.75
                        self.visLaneDetectImage[y1:y2, x1:x2, color_plane_bgr] = 255
                    
                    if len(line_points) > 0:
                        line_points = np.append(line_points, points, axis=0)
                    else:
                        line_points = points

            return line_points

        # Vizualuze lane lines if any detected
        def visualize_lane_line(lane_line, color_bgr = (0, 255, 255), thickness = 5):
            if self.visualization:
                if lane_line.isLineDetected:
                    vect_y = np.array(np.mgrid[0:bin_shape[0]], dtype = np.float64)
                    vect_x = lane_line.lineShape[0] * (vect_y**2) + lane_line.lineShape[1] * vect_y + lane_line.lineShape[2]

                    vect_filter = (vect_x >= 0) & (vect_x < bin_shape[1])
                    vect_y = vect_y[vect_filter]
                    vect_x = vect_x[vect_filter]

                    points = np.array([vect_x, vect_y], dtype = np.int32).T

                    cv2.polylines(self.visLaneDetectImage, [points], 0, color_bgr, thickness = thickness)

        # Vizualuze history points to see total data set for videos
        def visualize_lane_points(lane_line, color_plane_bgr):
            if self.visualization:
                for points in lane_line.histPoints:
                    if len(points) > 0:
                        for y1, x1, w in points:
                            self.visLaneDetectImage[y1, x1, color_plane_bgr] = 255

        if self.visualization:
            self.visLaneDetectImage = self.camera.binaryToImg(binary)

        bin_shape = binary.shape

        window_matrix_l = [] # (x1, y1, x2, y2) - coordinates of windowses related to left lane line
        window_matrix_r = [] # (x1, y1, x2, y2) - coordinates of windowses related to right lane line
        conv_win_size = self.detectorWinSize[1]
        conv_win_half = conv_win_size // 2 # Half of convolution window size
        conv_win = np.ones(conv_win_size) # Convolution windows - contains all values 1.0
        detectorWinMarging = self.detectorWinMarging[1]

        # First step - roughly detect bottom start position for both lines
        # For this purpose we select 2 windows at the bottom of the screen centered by left and right image halfs
    
        win_pos_x_w = int(bin_shape[1] * self.detectorInitSize[1])
        win_pos_x_l = int(((bin_shape[1] / 2.0) - win_pos_x_w) / 2.0)
        if win_pos_x_l < 0:
            win_pos_x_l = 0

        win_pos_x_r = int((bin_shape[1] / 2.0) + win_pos_x_l)

        win_pos_y = int(bin_shape[0] * (1.0 - self.detectorInitSize[0]))
        img_pos_y = bin_shape[0]

        if self.detectorUsePreviousLine and self.laneLines[0].isLineDetected:
            # Define start point with lane line detected from previous video frames
            l_center = self.laneLines[0].lineShape[0] * (img_pos_y ** 2) + self.laneLines[0].lineShape[1] * img_pos_y + self.laneLines[0].lineShape[2]
        else:
            # Define start point with convolution of bottom left and right part of image.
            win_adj, l_center, win_sum, win_pct = process_win((win_pos_y, win_pos_x_l, bin_shape[0], win_pos_x_l + win_pos_x_w))

        if self.detectorUsePreviousLine and self.laneLines[1].isLineDetected:
            # Define start point with lane line detected from previous video frames
            r_center = self.laneLines[1].lineShape[0] * (img_pos_y ** 2) + self.laneLines[1].lineShape[1] * img_pos_y + self.laneLines[1].lineShape[2]
        else:
            # Define start point with convolution of bottom left and right part of image.
            win_adj, r_center, win_sum, win_pct = process_win((win_pos_y, win_pos_x_r, bin_shape[0], win_pos_x_r + win_pos_x_w))

        det_marging_l = self.detectorWinMarging[0]
        l_detect = True
        l_empty = 0
        l_center_points = []
        
        det_marging_r = self.detectorWinMarging[0]
        r_detect = True
        r_empty = 0
        r_center_points = []

        while (l_detect or r_detect) and (img_pos_y > 0):
            # Left line detection
            if l_detect:
                # Extrapolation to detect next window center
                l_shape = []
                if len(l_center_points) > self.detectorMinApproxBoxes:
                    points = np.array(l_center_points, dtype = np.int32).T
                    points_w = np.array(1 + np.around(points[0] / float(bin_shape[0]) * float(self.detectorWeightFactor - 1)), dtype=np.int32)

                    if len(window_matrix_l) >= self.detectorMinApproxBoxesSq:
                        l_shape = np.polyfit(points[0], points[1], 2, w = points_w)
                    else:
                        l_shape = np.polyfit(points[0], points[1], 1, w = points_w)
                elif self.detectorUsePreviousLine and self.laneLines[0].isLineDetected:
                    l_shape = self.laneLines[0].lineShape

                if len(l_shape) >= 3:
                    l_center = l_shape[0] * (img_pos_y ** 2) + l_shape[1] * img_pos_y + l_shape[2]
                    det_marging_l = detectorWinMarging
                elif len(l_shape) >= 2:
                    l_center = l_shape[0] * img_pos_y + l_shape[1]
                    det_marging_l = detectorWinMarging

                if ((l_center - conv_win_half) < 0) | ((l_center + conv_win_half) >= bin_shape[1]):
                    # If left or right side of image reached, stop detection process to avoid lane line deformation
                    l_detect = False
                else:
                    # Use convolution to detect next segment of lane line
                    win_adj_l, l_center_new, win_sum, win_pct = process_win(
                        (img_pos_y - self.detectorWinSize[0],
                         l_center - conv_win_half - det_marging_l,
                         img_pos_y,
                         l_center + conv_win_half + det_marging_l))

                    if win_pct >= self.detectorEmptyThreshold:
                        l_empty = 0
                        l_center = l_center_new
                        window_matrix_l += [win_adj_l]
                        l_center_points += [[img_pos_y, l_center_new]]
                    else:
                        l_empty += 1
                        l_center_points += [[img_pos_y, l_center]]
                        #if l_empty >= self.detectorEmptySeq:
                        #    l_detect = False

            # Right line detection
            if r_detect:
                # Extrapolation to detect next window center
                r_shape = []
                if len(r_center_points) > self.detectorMinApproxBoxes:
                    points = np.array(r_center_points, dtype = np.int32).T
                    points_w = np.array(1 + np.around(points[0] / float(bin_shape[0]) * float(self.detectorWeightFactor - 1)), dtype=np.int32)

                    if len(window_matrix_r) >= self.detectorMinApproxBoxesSq:
                        r_shape = np.polyfit(points[0], points[1], 2, w = points_w)
                    else:
                        r_shape = np.polyfit(points[0], points[1], 1, w = points_w)
                elif self.detectorUsePreviousLine and self.laneLines[1].isLineDetected:
                    r_shape = self.laneLines[1].lineShape

                if len(r_shape) >= 3:
                    r_center = r_shape[0] * (img_pos_y ** 2) + r_shape[1] * img_pos_y + r_shape[2]
                    det_marging_r = detectorWinMarging
                elif len(r_shape) >= 2:
                    r_center = r_shape[0] * img_pos_y + r_shape[1]
                    det_marging_r = detectorWinMarging

                if ((r_center - conv_win_half) < 0) | ((r_center + conv_win_half) >= bin_shape[1]):
                    # If left or right side of image reached, stop detection process to avoid lane line deformation
                    r_detect = False
                else:
                    # Use convolution to detect next segment of lane line
                    win_adj_r, r_center_new, win_sum, win_pct = process_win(
                        (img_pos_y - self.detectorWinSize[0],
                         r_center - conv_win_half - det_marging_r,
                         img_pos_y,
                         r_center + conv_win_half + det_marging_r))

                    if win_pct >= self.detectorEmptyThreshold:
                        r_empty = 0
                        r_center = r_center_new
                        window_matrix_r += [win_adj_r]
                        r_center_points += [[img_pos_y, r_center_new]]
                    else:
                        r_empty += 1
                        r_center_points += [[img_pos_y, r_center]]
                        #if r_empty >= self.detectorEmptySeq:
                        #    r_detect = False

            img_pos_y -= self.detectorWinSize[0]

            if img_pos_y > 0:
                # Calculate size of next convolution window. This size is increasing from bottom to top of the image.
                conv_win_size_new = int(np.ceil(self.detectorWinSize[1] + ((float(bin_shape[0]) - float(img_pos_y)) / float(bin_shape[0])) * float(self.detectorWinSize[2] - self.detectorWinSize[1])))
                detectorWinMarging_new = int(np.ceil(self.detectorWinSize[1] + ((float(bin_shape[0]) - float(img_pos_y)) / float(bin_shape[0])) * float(self.detectorWinMarging[2] - self.detectorWinMarging[1])))

                if conv_win_size_new != conv_win_size:
                    conv_win_size = conv_win_size_new
                    conv_win_half = conv_win_size // 2
                    conv_win = np.ones(conv_win_size)

                detectorWinMarging_new = int(np.ceil(self.detectorWinSize[1] + ((float(bin_shape[0]) - float(img_pos_y)) / float(bin_shape[0])) * float(self.detectorWinMarging[2] - self.detectorWinMarging[1])))
                if detectorWinMarging_new != detectorWinMarging:
                    detectorWinMarging = detectorWinMarging_new

            det_marging_l = detectorWinMarging
            det_marging_r = detectorWinMarging

        # Retrive lane points from set of windows
        line_points_l = combine_line_points(window_matrix_l, 0)
        line_points_r = combine_line_points(window_matrix_r, 2)

        # Validate lane lines separately
        valid_pos_y = bin_shape[0] - 1
        checkCurvaturePoints = np.mgrid[0:bin_shape[0]]

        if self.detectorUsePreviousLine and self.laneLines[0].isLineDetected:
            l_shape = self.laneLines[0].lineShape
            l_center = l_shape[0] * (valid_pos_y ** 2) + l_shape[1] * valid_pos_y + l_shape[2]
            check_res_l, line_points_l, line_shape_l = self.laneLines[0].checkFilterLinePoints(line_points_l, (bin_shape[0], max(win_pos_x_l, l_center - self.detectorWinMarging[0]), min(win_pos_x_l + win_pos_x_w, l_center + self.detectorWinMarging[0])), checkCurvaturePoints, self.detectorMaxAllowedCurvDiff)
        else:
            check_res_l, line_points_l, line_shape_l = self.laneLines[0].checkFilterLinePoints(line_points_l, (valid_pos_y, win_pos_x_l, win_pos_x_l + win_pos_x_w), checkCurvaturePoints, self.detectorMaxAllowedCurvDiff)

        if self.detectorUsePreviousLine and self.laneLines[1].isLineDetected:
            r_shape = self.laneLines[1].lineShape
            r_center = r_shape[0] * (valid_pos_y ** 2) + r_shape[1] * valid_pos_y + r_shape[2]
            check_res_r, line_points_r, line_shape_r = self.laneLines[1].checkFilterLinePoints(line_points_r, (bin_shape[0], max(win_pos_x_r, r_center - self.detectorWinMarging[0]), min(win_pos_x_r + win_pos_x_w, r_center + self.detectorWinMarging[0])), checkCurvaturePoints, self.detectorMaxAllowedCurvDiff)
        else:
            check_res_r, line_points_r, line_shape_r = self.laneLines[1].checkFilterLinePoints(line_points_r, (valid_pos_y, win_pos_x_r, win_pos_x_r + win_pos_x_w), checkCurvaturePoints, self.detectorMaxAllowedCurvDiff)

        # Validate both lane lines
        addPoints = False
        if check_res_l and check_res_r:
            # Validate if lane lines intersect within double image height.
            a1 = line_shape_l[0] - line_shape_r[0]
            b1 = line_shape_l[1] - line_shape_r[1]
            c1 = line_shape_l[2] - line_shape_r[2]
            det = b1 ** 2 - 4 * a1 * c1
            if det >= 0:
                y1 = (-b1 - np.sqrt(det)) / (2 * a1)
                y2 = (-b1 + np.sqrt(det)) / (2 * a1)
            else:
                y1 = 0
                y2 = 0

            if (det < 0) or ((abs(y1) > bin_shape[0]) and (abs(y2) > bin_shape[0])):
                # Validate if lane line curvature is comparable
                if (line_shape_l[0] != 0) and (line_shape_r[0] != 0):
                    rad1 = np.log10(np.min(((1 + (2 * line_shape_l[0] * checkCurvaturePoints + line_shape_l[1]) ** 2) ** 1.5) / np.absolute(2 * line_shape_l[0])))
                    rad2 = np.log10(np.min(((1 + (2 * line_shape_r[0] * checkCurvaturePoints + line_shape_r[1]) ** 2) ** 1.5) / np.absolute(2 * line_shape_r[0])))

                    if (rad1 != 0) and (rad2 != 0):
                        change_power = 100.0**(max(rad1, rad2) / min(rad1, rad2)) / 100.0
                        if change_power <= self.detectorMaxAllowedCurvDiff:
                            addPoints = True
                else:
                    addPoints = True

        # Add detected points to lane line history. If validation was not successful, empty values are added
        if addPoints:
            self.laneLines[0].addLinePoints(line_points_l)
            self.laneLines[1].addLinePoints(line_points_r)
        else:
            self.laneLines[0].addLinePoints([])
            self.laneLines[1].addLinePoints([])

        # Update shape of lane lines
        world_correction = self.camera.getWorldCorrection()
        self.laneLines[0].updateLineShape(world_correction)
        self.laneLines[1].updateLineShape(world_correction)

        # if mutual lane shape correction enabled, do correction to calculate average lane lines shape for both lines
        if self.detectorMutualLaneShapeCorrection:
            if self.visualization:
                visualize_lane_line(self.laneLines[0], thickness = 2)
                visualize_lane_line(self.laneLines[1], thickness = 2)

            # Lane shape must be approxumately the same, so we can mutually correct it both
            fix_point_y = bin_shape[0] * (1.0 - self.detectorInitSize[0])
            fix_point_world_y = fix_point_y * world_correction[0]
            LaneLine.LaneLine.mutualLaneShapeCorrection(self.laneLines[0], self.laneLines[1], fix_point_y, fix_point_world_y)

        # Perform vizualization
        if self.visualization:
            visualize_lane_line(self.laneLines[0], thickness = 7)
            visualize_lane_line(self.laneLines[1], thickness = 7)

        if self.visualization:
            visualize_lane_points(self.laneLines[0], 1)
            visualize_lane_points(self.laneLines[1], 1)

    def _calculateParameters(
        self,
        img, # Image for final vizualization
        top_shape # Shape of top-view image
        ):
        """
        Calculate parameters based on detected lane lines.
        """

        def add_vehicle_detection():
            bottom_center_point = self.camera.perspectiveTransformPoints(np.array((img_shape[1] / 2.0, img_shape[0]), dtype = np.float64), False)
            bottom_center_point = (bottom_center_point[0] * world_correction[1], bottom_center_point[1] * world_correction[0])

            for obj_idx, is_assigned, box0, box1, hist_center, hist_metrics, size_history, object_label, object_class in self.objectsHistory:
                if is_assigned and (object_label > 0):
                    annotation_color = self.annotationWindowСolor

                    if object_label <= len(self.annotationWindowСolorList):
                        annotation_color = self.annotationWindowСolorList[object_label - 1]

                    vehicle_point = self.camera.perspectiveTransformPoints(np.array(((box0[1] + box1[1]) / 2.0, (box0[0] + 4.0 * box1[0]) / 5.0), dtype = np.float64), False)
                    vehicle_point = (vehicle_point[0] * world_correction[1], vehicle_point[1] * world_correction[0])

                    distance_to_vehicle_lr = vehicle_point[0] - bottom_center_point[0]
                    if distance_to_vehicle_lr >= 0:
                        distance_to_vehicle_dir = "R"
                    else:
                        distance_to_vehicle_dir = "L"

                    distance_to_vehicle_lr = int(abs(distance_to_vehicle_lr))
                    distance_to_vehicle_f = int(abs(vehicle_point[1] - bottom_center_point[1]))

                    if (distance_to_vehicle_lr <= self.annotationMaxDistanceFilterLR) and (distance_to_vehicle_f <= self.annotationMaxDistanceFilterF):
                        cv2.rectangle(res_img, box0[::-1], box1[::-1], annotation_color, self.annotationWindowThickness)

                        cv2.putText(
                            res_img,
                            '{} {}'.format(object_class, object_label),
                            (box0[1] + 1, box0[0] - 35 + 1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            thickness = 1,
                            lineType = cv2.LINE_AA)

                        cv2.putText(
                            res_img,
                            '{} {}'.format(object_class, object_label),
                            (box0[1], box0[0] - 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            annotation_color,
                            thickness = 1,
                            lineType = cv2.LINE_AA)

                        cv2.putText(
                            res_img,
                            '{}: {}m'.format(distance_to_vehicle_dir, distance_to_vehicle_lr),
                            (box0[1] + 1, box0[0] - 20 + 1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            thickness = 1,
                            lineType = cv2.LINE_AA)

                        cv2.putText(
                            res_img,
                            '{}: {}m'.format(distance_to_vehicle_dir, distance_to_vehicle_lr),
                            (box0[1], box0[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            annotation_color,
                            thickness = 1,
                            lineType = cv2.LINE_AA)

                        cv2.putText(
                            res_img,
                            'F: {}m'.format(distance_to_vehicle_f),
                            (box0[1] + 1, box0[0] - 5 + 1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            thickness = 1,
                            lineType = cv2.LINE_AA)

                        cv2.putText(
                            res_img,
                            'F: {}m'.format(distance_to_vehicle_f),
                            (box0[1], box0[0] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            annotation_color,
                            thickness = 1,
                            lineType = cv2.LINE_AA)

        res_img = img.copy()
        img_shape = res_img.shape
        world_correction = self.camera.getWorldCorrection()

        if (not self.laneLines[0].isLineDetected) or (not self.laneLines[1].isLineDetected):
            add_vehicle_detection()
            return False, res_img

        # Create set of points from lane line shape and perform inverse perspective transformation
        vect_y = np.array(np.mgrid[0:top_shape[0]], dtype = np.float64)
        vect_x1 = self.laneLines[0].lineShape[0] * (vect_y**2) + self.laneLines[0].lineShape[1] * vect_y + self.laneLines[0].lineShape[2]
        vect_x2 = self.laneLines[1].lineShape[0] * (vect_y**2) + self.laneLines[1].lineShape[1] * vect_y + self.laneLines[1].lineShape[2]

        points1 = np.array([vect_x1, vect_y], dtype = np.int32).T
        points1 = np.array(self.camera.perspectiveTransformPoints(points1, True), dtype = np.int32)

        points2 = np.array([vect_x2, vect_y], dtype = np.int32).T
        points2 = np.array(self.camera.perspectiveTransformPoints(points2, True), dtype = np.int32)

        checkCurvaturePoints = np.mgrid[0:top_shape[0]] * world_correction[0]

        self.annotationDelayCnt = (self.annotationDelayCnt + 1) % self.annotationDelayFrames

        # Calculate curvature of lane lines
        if self.laneLines[0].isLineDetected and (self.laneLines[0].lineShapeWorld[0] != 0):
            cur_rad1 = np.min(((1 + (2 * self.laneLines[0].lineShapeWorld[0] * checkCurvaturePoints + self.laneLines[0].lineShapeWorld[1]) ** 2) ** 1.5) / np.absolute(2 * self.laneLines[0].lineShapeWorld[0]))
            # Straight lines can have different values, so makes sense limit maximum value.
            cur_rad1 = min(5000, cur_rad1)
        else:
            cur_rad1 = 0

        if self.laneLines[1].isLineDetected and (self.laneLines[1].lineShapeWorld[0] != 0):
            cur_rad2 = np.min(((1 + (2 * self.laneLines[1].lineShapeWorld[0] * checkCurvaturePoints + self.laneLines[1].lineShapeWorld[1]) ** 2) ** 1.5) / np.absolute(2 * self.laneLines[1].lineShapeWorld[0]))
            # Straight lines can have different values, so makes sense limit maximum value.
            cur_rad1 = min(5000, cur_rad1)
        else:
            cur_rad2 = 0

        if (cur_rad1 <= 0) & (cur_rad2 <= 0):
            cur_rad = 0
        elif cur_rad1 <= 0:
            cur_rad = cur_rad2
        elif cur_rad2 <= 0:
            cur_rad = cur_rad1
        else:
            cur_rad = min(cur_rad1, cur_rad2)

        self.annotationRadiusBuf += [cur_rad]
        if len(self.annotationRadiusBuf) > self.annotationBufferSize:
            self.annotationRadiusBuf = self.annotationRadiusBuf[1:]

        if (self.annotationDelayCnt == 0) or (self.annotationRadius <= 0):
            buff = np.array(self.annotationRadiusBuf, dtype = np.float64)
            buff = buff[buff > 0]

            if len(buff) > 0:
                self.annotationRadius = np.mean(buff)
            else:
                self.annotationRadius = 0

        # Calculate road center shift
        lane_btm_y = top_shape[0] * world_correction[0]
        if self.laneLines[0].isLineDetected and self.laneLines[1].isLineDetected:
            lane_btm_x1 = self.laneLines[0].lineShapeWorld[0] * (lane_btm_y ** 2) + self.laneLines[0].lineShapeWorld[1] * lane_btm_y + self.laneLines[0].lineShapeWorld[2]
            lane_btm_x2 = self.laneLines[1].lineShapeWorld[0] * (lane_btm_y ** 2) + self.laneLines[1].lineShapeWorld[1] * lane_btm_y + self.laneLines[1].lineShapeWorld[2]
            lane_size = lane_btm_x2 - lane_btm_x1
        else:
            lane_size = -1

        if lane_size > 0:
            lane_btm_y = top_shape[0]
            lane_btm_x1 = self.laneLines[0].lineShape[0] * (lane_btm_y ** 2) + self.laneLines[0].lineShape[1] * lane_btm_y + self.laneLines[0].lineShape[2]
            lane_btm_x2 = self.laneLines[1].lineShape[0] * (lane_btm_y ** 2) + self.laneLines[1].lineShape[1] * lane_btm_y + self.laneLines[1].lineShape[2]
            lane_image_points = self.camera.perspectiveTransformPoints(np.array([[lane_btm_x1, lane_btm_y], [lane_btm_x2, lane_btm_y]], dtype = np.float64), True)
            x1 = lane_image_points[0, 0]
            x2 = lane_image_points[1, 0]

            if x1 != x2:
                cur_shift = (img_shape[1] / 2.0 - (x1 + x2) / 2.0) / np.absolute(x2 - x1) * lane_size
            else:
                cur_shift = -10000000
        else:
            cur_shift = -10000000

        self.annotationCenterShiftBuf += [cur_shift]
        if len(self.annotationCenterShiftBuf) > self.annotationBufferSize:
            self.annotationCenterShiftBuf = self.annotationCenterShiftBuf[1:]

        if (self.annotationDelayCnt == 0) or (self.annotationCenterShift <= -1000000):
            buff = np.array(self.annotationCenterShiftBuf, dtype = np.float64)
            buff = buff[buff > -1000000]

            if len(buff) > 0:
                self.annotationCenterShift = np.mean(buff)
            else:
                self.annotationCenterShift = 0

        # Do vizualuzation
        image_mask = np.zeros_like(res_img)
        color_mask = np.zeros_like(res_img)
        color_mask[:] = self.annotationColorBGR

        points = np.append(points1, points2[::-1], axis = 0)

        cv2.fillPoly(image_mask, [points], (255, 255, 255))

        image_outside_mask = cv2.bitwise_and(res_img, ~image_mask)
        image_inside_mask = cv2.bitwise_and(res_img, image_mask)
        color_mask = cv2.bitwise_and(color_mask, image_mask)
        
        image_inside_mask = cv2.addWeighted(image_inside_mask, 1 - self.annotationWeight, color_mask, self.annotationWeight, 0)
        res_img = cv2.addWeighted(image_outside_mask, 1.0, image_inside_mask, 1.0, 0)

        image_mask = np.zeros_like(res_img)
        color_mask = np.zeros_like(res_img)
        color_mask[:] = self.annotationColorLineLeftBGR

        cv2.polylines(image_mask, [points1], 0, (255, 255, 255), thickness = 10)

        image_outside_mask = cv2.bitwise_and(res_img, ~image_mask)
        image_inside_mask = cv2.bitwise_and(res_img, image_mask)
        color_mask = cv2.bitwise_and(color_mask, image_mask)
        
        image_inside_mask = cv2.addWeighted(image_inside_mask, 1 - self.annotationWeight, color_mask, self.annotationWeight, 0)
        res_img = cv2.addWeighted(image_outside_mask, 1.0, image_inside_mask, 1.0, 0)

        image_mask = np.zeros_like(res_img)
        color_mask = np.zeros_like(res_img)
        color_mask[:] = self.annotationColorLineRightBGR

        cv2.polylines(image_mask, [points2], 0, (255, 255, 255), thickness = 10)

        image_outside_mask = cv2.bitwise_and(res_img, ~image_mask)
        image_inside_mask = cv2.bitwise_and(res_img, image_mask)
        color_mask = cv2.bitwise_and(color_mask, image_mask)
        
        annotationRadiusMul = max(1, int(np.round(self.annotationRadius / 100.0)))
        image_inside_mask = cv2.addWeighted(image_inside_mask, 1 - self.annotationWeight, color_mask, self.annotationWeight, 0)
        res_img = cv2.addWeighted(image_outside_mask, 1.0, image_inside_mask, 1.0, 0)

        cv2.putText(
            res_img,
            'Radius of curvature = {:.1f} km'.format(annotationRadiusMul / 10.0),
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            self.annotationColorTextBGR,
            thickness = 2,
            lineType = cv2.LINE_AA)

        annotationCenterShiftMult = int(self.annotationCenterShift * 10.0)
        if annotationCenterShiftMult > 0:
            txt = 'Vehicle is {:.1f} m right of center'.format(annotationCenterShiftMult / 10.0)
        elif annotationCenterShiftMult < 0:
            txt = 'Vehicle is {:.1f} m left of center'.format(-annotationCenterShiftMult / 10.0)
        else:
            txt = 'Vehicle is by center'

        cv2.putText(
            res_img,
            txt,
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            self.annotationColorTextBGR,
            thickness = 2,
            lineType = cv2.LINE_AA)

        add_vehicle_detection()

        return True, res_img

    def _labelVehiclesInHistory(self):
        # Label tracked objects with labels reusing
        next_obj_label = 1
        used_labels = set()
        for obj_idx, is_assigned, box0, box1, hist_center, hist_metrics, size_history, object_label, object_class in self.objectsHistory:
            if object_label > 0:
                used_labels.add(object_label)

        for idx in range(len(self.objectsHistory)):
            obj_idx, is_assigned, box0, box1, hist_center, hist_metrics, size_history, object_label, object_class = self.objectsHistory[idx]
            if is_assigned and (object_label <= 0):
                while next_obj_label in used_labels:
                    next_obj_label += 1

                object_label = next_obj_label
                next_obj_label += 1

                self.objectsHistory[idx] = (obj_idx, is_assigned, box0, box1, hist_center, hist_metrics, size_history, object_label, object_class)

    def _vehicleTracking(
        self,
        img # Source image in OpenCV BGR format
        ):
        """
        Vehicle tracking algorithm
        """

        if self.visualization:
            self.visAllBoxesImage = img.copy()
            self.visVehicleBoxesImage = img.copy()

            self.visHeatMapImage = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
            self.visHeatMapImage[:,:,0] = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY) // 2
            self.visHeatMapImage[:,:,1] = self.visHeatMapImage[:,:,0]
            self.visHeatMapImage[:,:,2] = self.visHeatMapImage[:,:,0]
        
        isFirstPrint = True
        
        conv_win_half = self.heatMapConvolutionWindowSize // 2 # Half of convolution window size
        conv_win = np.ones(self.heatMapConvolutionWindowSize) # Convolution windows - contains all values 1.0

        region_set = []
        heatMap = np.zeros_like(img[:,:,0], dtype = np.float)

        if self.classifierDarkFlow != None:
            img_shape = img.shape
            img_dark = np.zeros(shape=(max(img_shape[0], img_shape[1]), max(img_shape[0], img_shape[1]), img_shape[2]), dtype = np.uint8)
            img_dark[:img_shape[0], :img_shape[1]] = img

            img_dark_mask = np.zeros(shape=(img_shape[0], img_shape[1]), dtype = np.uint8)
        
            detectedObjects_dark = []
            for detected_obj in self.classifierDarkFlow.return_predict(img_dark):
                object_label = detected_obj["label"]
                confidence = detected_obj["confidence"]
                box0 = (detected_obj["topleft"]["y"], detected_obj["topleft"]["x"])
                box1 = (detected_obj["bottomright"]["y"], detected_obj["bottomright"]["x"])

                if ((object_label in self.useDarkFlowLabels) and
                    (box0[0] <= img_shape[0]) and
                    (box0[1] <= img_shape[1]) and
                    (box1[0] <= img_shape[0]) and
                    (box1[1] <= img_shape[1])):

                    detectedObjects_dark += [[box0, box1, (int((box0[0] + box1[0]) / 2.0), int((box0[1] + box1[1]) / 2.0)), object_label, confidence]]
                    img_dark_mask[box0[0]:box1[0], box0[1]:box1[1]] = 1

        if (self.useDarkFlowMode == 2) and (self.classifierDarkFlow != None):
            for box0, box1, object_center, object_label, confidence in detectedObjects_dark:
                # Add region in list of regions and on heat map
                region_set += [[box0, box1, confidence * self.useDarkFlowMult]]
                heatMap[box0[0]:box1[0], box0[1]:box1[1]] += confidence * self.useDarkFlowMult

                if self.visualization:
                    cv2.rectangle(self.visVehicleBoxesImage, box0[::-1], box1[::-1], self.annotationWindowСolor, self.annotationWindowThickness)
                    cv2.rectangle(self.visHeatMapImage, box0[::-1], box1[::-1], (127, 127, 0), 1)
        else:
            # For each feature size and region on image find all features with sliding window
            for detection_region in self.detectionRegions:
                for feature_size in detection_region[2]:
                    blur_kernel_size = (feature_size // 8) - 1
                    img_blur = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)

                    if self.classifierCNN == None:
                        img_cnn = []
                    else:
                        img_cnn = img

                    features = self.imageEngine.getImageFeatures(
                        img_blur,
                        imgCNN = img_cnn,
                        img_window = (detection_region[0], detection_region[1]),
                        featureSize = (self.baseWindowSize, self.baseWindowSize),
                        featureScale = float(feature_size) / float(self.baseWindowSize),
                        overlap = detection_region[3],
                        visualise = False)

                    features = np.array(features)
                    x_predict = []
                    x1_predict = []

                    for feature in features:
                        if self.classifierCNN == None:
                            box0, box1, f_vector = feature[0:3]
                        elif len(feature) >= 4:
                            box0, box1, f_vector, cnn_feature = feature[0:4]
                        else:
                            box0, box1, f_vector = feature[0:3]
                            cnn_feature = np.zeros(shape=(self.baseWindowSize, self.baseWindowSize, 3))

                        box0 = (int(box0[0]), int(box0[1]))
                        box1 = (int(box1[0]), int(box1[1]))

                        x_predict += [f_vector]

                        if self.classifierCNN != None:
                            x1_predict += [cnn_feature]

                        if self.visualization:
                            cv2.rectangle(self.visAllBoxesImage, box0[::-1], box1[::-1], self.annotationWindowСolor, self.annotationWindowThickness)

                    x_predict = np.array(x_predict)
                    x1_predict = np.array(x1_predict)
                    features = np.array(features)

                    if len(x_predict) > 0:
                        if self.classifierFast != None:
                            y_predict = self.classifierFast.predict(x_predict)
                            x_predict = x_predict[y_predict == 1]
                            x1_predict = x1_predict[y_predict == 1]
                            features = features[y_predict == 1]

                    if len(x_predict) > 0:
                        if self.classifierCNN != None:
                            y_predict = np.argmax(self.classifierCNN.predict(x1_predict, batch_size = len(x1_predict), verbose = 0), axis = -1)
                            features = features[y_predict == 1]
                        elif self.classifierAccurate != None:
                            y_predict = self.classifierAccurate.predict(x_predict)
                            features = features[y_predict == 1]
                        else:
                            raise Exception("At least one accurate classifier must be specified.")

                        for feature in features:
                            box0, box1 = feature[0:2]
                            box0 = (int(box0[0]), int(box0[1]))
                            box1 = (int(box1[0]), int(box1[1]))

                            # Add region in list of regions and on heat map
                            region_set += [[box0, box1, 1.0]]
                            heatMap[box0[0]:box1[0], box0[1]:box1[1]] += 1.0

                            if self.visualization:
                                cv2.rectangle(self.visVehicleBoxesImage, box0[::-1], box1[::-1], self.annotationWindowСolor, self.annotationWindowThickness)
                                cv2.rectangle(self.visHeatMapImage, box0[::-1], box1[::-1], (127, 127, 0), 1)

        if self.visualization:
            heat_map_img = heatMap.copy().astype(np.float)
            max_heat_map = np.max(heat_map_img)
            if max_heat_map > 0:
                heat_map_img = (heat_map_img / max_heat_map * 128.0) + 128.0
            heat_map_img = np.clip(heat_map_img, 128, 255).astype(np.uint8)

            (self.visHeatMapImage[:,:,2])[heat_map_img > 128] = heat_map_img[heat_map_img > 128]

        self.regionHistory += [region_set]
        if len(self.regionHistory) > self.heatMapFrames:
            self.regionHistory = self.regionHistory[1:]

        if len(self.regionHistory) >= self.heatMapFrames:
            # Combine history regions in one set
            region_set = []
            for cur_reg_set in self.regionHistory:
                region_set += cur_reg_set

            isObjectDetected = True
            detectedObjects_all = []

            if (self.useDarkFlowMode == 2) and (self.classifierDarkFlow != None):
                # Create heat map from hisorical regions set
                heatMaps_total = np.zeros_like(img[:,:,0], dtype = np.float)
                for box0, box1, reg_weight in region_set:
                    heatMaps_total[box0[0]:box1[0], box0[1]:box1[1]] += reg_weight

                for box0, box1, object_center, object_label, confidence in detectedObjects_dark:
                    # Calculate metrix of region (like total heat map sum) and check if region meet all requirements, otherwise it will be rejected
                    def calc_box_parameters(this_box0, this_box1):
                        l_heat_max = int(np.max(heatMaps_total[this_box0[0]:this_box1[0], this_box0[1]:this_box1[1]]))
                        l_heat_sum = int(np.sum(heatMaps_total[this_box0[0]:this_box1[0], this_box0[1]:this_box1[1]]) / 100)
                        l_heat_sum_frame = int(np.sum(heatMap[this_box0[0]:this_box1[0], this_box0[1]:this_box1[1]]) / 100)

                        l_is_valid = (
                            (l_heat_sum >= self.heatMapTotalMin) and
                            (l_heat_sum_frame >= self.heatMapTotalMinFrame) and
                            ((this_box1[0] - this_box0[0]) > 0) and
                            ((this_box1[1] - this_box0[1]) > 0) and
                            (((float(this_box1[0] - this_box0[0]) / float(this_box1[1] - this_box0[1])) <= self.objHeightWidthLimit) or (self.classifierDarkFlow != None)))

                        if l_is_valid:
                            for box0_det, box1_det, object_center, object_metrics in detectedObjects_all:
                                if (this_box0[0] >= box0_det[0]) and (this_box1[0] <= box1_det[0]) and (this_box0[1] >= box0_det[1]) and (this_box1[1] <= box1_det[1]):
                                    l_is_valid = False
                                    break

                        return l_is_valid, l_heat_max, l_heat_sum, l_heat_sum_frame

                    is_valid, heat_max, heat_sum, heat_sum_frame = calc_box_parameters(box0, box1)
                    if is_valid:
                        detectedObjects_all += [[box0, box1, object_center, (heat_max, heat_sum, heat_sum_frame)]]
            else:
                # Repeat object detection in cycle to detect overlapped objects
                while isObjectDetected:
                    isObjectDetected = False
                    detectedObjects = []

                    # Create heat map from hisorical regions set
                    heatMaps_total = np.zeros_like(img[:,:,0], dtype = np.float)
                    for box0, box1, reg_weight in region_set:
                        heatMaps_total[box0[0]:box1[0], box0[1]:box1[1]] += reg_weight

                    # Filter heat map
                    heatMaps_total[(heatMaps_total < self.heatMapThreshold) | (heatMap <= 0)] = 0

                    if self.classifierDarkFlow != None:
                        heatMaps_total[img_dark_mask <= 0] = 0
            
                    # Detect separate regions based on heat map (labeling)
                    labels_matrix, labels_num = scipy.ndimage.measurements.label(heatMaps_total)
                    if labels_num > 0:
                        for label_idx in range(1, labels_num + 1):
                            labels_matrix_filter = labels_matrix == label_idx
                            label_coord = np.array(labels_matrix_filter.nonzero())
                            box0 = (np.min(label_coord[0]), np.min(label_coord[1]))
                            box1 = (np.max(label_coord[0]) + 1, np.max(label_coord[1]) + 1)

                            # Algorithm to find maximal inner region - biggest possible sub-region of labeled region
                            def get_best_fit_box(edges, this_box0, this_box1):
                                if len(edges) <= 0:
                                    return this_box0, this_box1, np.sum(heatMaps_total[this_box0[0]:this_box1[0], this_box0[1]:this_box1[1]])

                                best_sz = -1
                                for idx in range(len(edges)):
                                    cur_edge = edges[idx]
                                    labels_matrix_filter_box = labels_matrix_filter[this_box0[0]:this_box1[0], this_box0[1]:this_box1[1]]
                                    new_box0, new_box1 = this_box0, this_box1

                                    if cur_edge == 0:
                                        filter_arr = np.argmax(labels_matrix_filter_box, axis=0)
                                        filter_arr.sort()
                                        new_box0 = (new_box0[0] + filter_arr[int(len(filter_arr) * self.heatMapEdgeThereshold)], new_box0[1])
                                    elif cur_edge == 1:
                                        filter_arr = np.argmax(labels_matrix_filter_box[::-1, :], axis=0)
                                        filter_arr.sort()
                                        new_box1 = (new_box1[0] - filter_arr[int(len(filter_arr) * self.heatMapEdgeThereshold)], new_box1[1])
                                    elif cur_edge == 2:
                                        filter_arr = np.argmax(labels_matrix_filter_box, axis=1)
                                        filter_arr.sort()
                                        new_box0 = (new_box0[0], new_box0[1] + filter_arr[int(len(filter_arr) * self.heatMapEdgeThereshold)])
                                    else:
                                        filter_arr = np.argmax(labels_matrix_filter_box[::-1, :], axis=1)
                                        filter_arr.sort()
                                        new_box1 = (new_box1[0], new_box1[1] - filter_arr[int(len(filter_arr) * self.heatMapEdgeThereshold)])

                                    cur_box0, cur_box1, cur_sz = get_best_fit_box(edges[:idx] + edges[(idx + 1):], new_box0, new_box1)
                                    if cur_sz > best_sz:
                                        best_box0, best_box1, best_sz = cur_box0, cur_box1, cur_sz

                                return best_box0, best_box1, best_sz

                            # Calculate metrix of region (like total heat map sum) and check if region meet all requirements, otherwise it will be rejected
                            def calc_box_parameters(this_box0, this_box1):
                                l_heat_max = int(np.max(heatMaps_total[this_box0[0]:this_box1[0], this_box0[1]:this_box1[1]]))
                                l_heat_sum = int(np.sum(heatMaps_total[this_box0[0]:this_box1[0], this_box0[1]:this_box1[1]]) / 100)
                                l_heat_sum_frame = int(np.sum(heatMap[this_box0[0]:this_box1[0], this_box0[1]:this_box1[1]]) / 100)

                                l_is_valid = (
                                    (l_heat_sum >= self.heatMapTotalMin) and
                                    (l_heat_sum_frame >= self.heatMapTotalMinFrame) and
                                    ((this_box1[0] - this_box0[0]) > 0) and
                                    ((this_box1[1] - this_box0[1]) > 0) and
                                    (((float(this_box1[0] - this_box0[0]) / float(this_box1[1] - this_box0[1])) <= self.objHeightWidthLimit) or (self.classifierDarkFlow != None)))

                                if l_is_valid:
                                    for box0_det, box1_det, object_center, object_metrics in detectedObjects_all:
                                        if (this_box0[0] >= box0_det[0]) and (this_box1[0] <= box1_det[0]) and (this_box0[1] >= box0_det[1]) and (this_box1[1] <= box1_det[1]):
                                            l_is_valid = False
                                            break

                                return l_is_valid, l_heat_max, l_heat_sum, l_heat_sum_frame

                            box0, box1, box_sz = get_best_fit_box([0,1,2,3], box0, box1)

                            if box_sz > 0:
                                is_valid, heat_max, heat_sum, heat_sum_frame = calc_box_parameters(box0, box1)
                                if is_valid:
                                    is_box_splitted = True
                                    while is_box_splitted:
                                        is_box_splitted = False
                                        
                                        box_center_x = (box1[1] + box0[1]) // 2
                                        if (box1[1] - box0[1]) > self.heatMapConvolutionWindowSize:
                                            # Detect object horizontal center
                                            vert_sum = np.sum(heatMaps_total[box0[0]:box1[0], box0[1]:box1[1]], axis=0)
                                            conv_matrix = np.convolve(vert_sum, conv_win, mode = 'valid')
                                            argmax_left = np.argmax(conv_matrix)
                                            object_center_x = argmax_left + box0[1] + conv_win_half

                                            # Object center is displaced, it must be several objects combined in one region, need split
                                            if object_center_x < ((box1[1] + 2 * box0[1]) // 3):
                                                argmax_right = np.argmax(conv_matrix[(len(conv_matrix) // 2):]) + (len(conv_matrix) // 2)
                                                argmin_left = np.argmin(conv_matrix[argmax_left + 1:argmax_right]) + argmax_left + 1
                                                object_split_x = argmin_left + box0[1] + conv_win_half
                                                if ((object_split_x - box0[1]) >= conv_win_half) and ((box1[1] - object_split_x) >= conv_win_half):
                                                    box0_new = (box0[0], box0[1])
                                                    box1_new = (box1[0], object_split_x)

                                                    box0_res = (box0[0], object_split_x)
                                                    box1_res = (box1[0], box1[1])

                                                    is_valid_l, heat_max_l, heat_sum_l, heat_sum_frame_l = calc_box_parameters(box0_new, box1_new)
                                                    is_valid_r, heat_max_r, heat_sum_r, heat_sum_frame_r = calc_box_parameters(box0_res, box1_res)

                                                    if is_valid_l and is_valid_r:
                                                        is_box_splitted = True

                                                        box0 = box0_new
                                                        box1 = box1_new
                                                        heat_max, heat_sum, heat_sum_frame = heat_max_l, heat_sum_l, heat_sum_frame_l
                                            elif object_center_x > ((2 * box1[1] + box0[1]) // 3):
                                                argmax_right = argmax_left
                                                argmax_left = np.argmax(conv_matrix[:(len(conv_matrix) // 2)])
                                                argmin_left = np.argmin(conv_matrix[argmax_left + 1:argmax_right]) + argmax_left + 1
                                                object_split_x = argmin_left + box0[1] + conv_win_half
                                                if ((object_split_x - box0[1]) >= conv_win_half) and ((box1[1] - object_split_x) >= conv_win_half):
                                                    box0_res = (box0[0], box0[1])
                                                    box1_res = (box1[0], object_split_x)

                                                    box0_new = (box0[0], object_split_x)
                                                    box1_new = (box1[0], box1[1])

                                                    is_valid_l, heat_max_l, heat_sum_l, heat_sum_frame_l = calc_box_parameters(box0_new, box1_new)
                                                    is_valid_r, heat_max_r, heat_sum_r, heat_sum_frame_r = calc_box_parameters(box0_res, box1_res)

                                                    if is_valid_l and is_valid_r:
                                                        is_box_splitted = True

                                                        box0 = box0_new
                                                        box1 = box1_new
                                                        heat_max, heat_sum, heat_sum_frame = heat_max_l, heat_sum_l, heat_sum_frame_l
                                        else:
                                            object_center_x = box_center_x

                                        # Detect object vertical center (used on future steps)
                                        box_center_y = (box1[0] + box0[0]) // 2
                                        if (box1[0] - box0[0]) > self.heatMapConvolutionWindowSize:
                                            horiz_sum = np.sum(heatMaps_total[box0[0]:box1[0], box0[1]:box1[1]], axis=1)
                                            object_center_y = np.argmax(np.convolve(horiz_sum, conv_win, mode = 'valid')) + box0[0] + conv_win_half
                                        else:
                                            object_center_y = box_center_y

                                        detectedObjects += [[box0, box1]]
                                        detectedObjects_all += [[box0, box1, (object_center_y, object_center_x), (heat_max, heat_sum, heat_sum_frame)]]

                                        if is_box_splitted:
                                            box0 = box0_res
                                            box1 = box1_res
                                            heat_max, heat_sum, heat_sum_frame = heat_max_r, heat_sum_r, heat_sum_frame_r

                    isObjectDetected = len(detectedObjects) > 0
                    if isObjectDetected:
                        # If any objects was detected, remove regions from historical region set which overlap with this object on aproximately 50%
                        # Remain regions will be used in next cycle
                        region_set_new = []
                        for box0, box1, reg_weight in region_set:
                            is_extra_region = True
                            for box0_det, box1_det in detectedObjects:
                                intersect0 = (max(box0[0], box0_det[0]), max(box0[1], box0_det[1]))
                                intersect1 = (min(box1[0], box1_det[0]), min(box1[1], box1_det[1]))

                                if ((intersect1[0] > intersect0[0]) and
                                    (intersect1[1] > intersect0[1]) and
                                    ((float((intersect1[0] - intersect0[0]) * (intersect1[1] - intersect0[1])) / float((box1[0] - box0[0]) * (box1[1] - box0[1]))) >= self.heatMapRegionThreshold)):
                                    is_extra_region = False
                                    break

                            if is_extra_region:
                                region_set_new += [[box0, box1, reg_weight]]

                        region_set = region_set_new

            if len(detectedObjects_all) >= 2:
                # Merge overlapped objects which overlap much enough (near 50%) - one-to-one validation
                is_obj_merged = True
                while is_obj_merged:
                    is_obj_merged = False
                    detectedObjects_all_new = []
                    for box0_det, box1_det, object_center, object_metrics in detectedObjects_all:
                        is_extra_region = True
                        for idx in range(len(detectedObjects_all_new)):
                            l_box0_det, l_box1_det, l_object_center, l_object_metrics = detectedObjects_all_new[idx]
                            intersect0 = (max(l_box0_det[0], box0_det[0]), max(l_box0_det[1], box0_det[1]))
                            intersect1 = (min(l_box1_det[0], box1_det[0]), min(l_box1_det[1], box1_det[1]))

                            if ((intersect1[0] > intersect0[0]) and
                                (intersect1[1] > intersect0[1]) and
                                (((float((intersect1[0] - intersect0[0]) * (intersect1[1] - intersect0[1])) / float((box1_det[0] - box0_det[0]) * (box1_det[1] - box0_det[1]))) >= self.objMergeThreshold) or
                                 ((float((intersect1[0] - intersect0[0]) * (intersect1[1] - intersect0[1])) / float((l_box1_det[0] - l_box0_det[0]) * (l_box1_det[1] - l_box0_det[1]))) >= self.objMergeThreshold))):
                            
                                is_extra_region = False

                                if ((box1_det[0] - box0_det[0]) * (box1_det[1] - box0_det[1])) > ((l_box1_det[0] - l_box0_det[0]) * (l_box1_det[1] - l_box0_det[1])):
                                    detectedObjects_all_new[idx] = [box0_det, box1_det, object_center, object_metrics]
                                    is_obj_merged = True
                                break

                        if is_extra_region:
                            detectedObjects_all_new += [[box0_det, box1_det, object_center, object_metrics]]

                    detectedObjects_all = detectedObjects_all_new

            if len(detectedObjects_all) >= 2:
                # If object overlap with multiple objects enough (near 50%), remove it - one-to-multiple validation
                detectedObjects_all_new = []
                for idx0 in range(len(detectedObjects_all)):
                    box0_det, box1_det, object_center, object_metrics = detectedObjects_all[idx0]
                    box0_sz = (box1_det[0] - box0_det[0]) * (box1_det[1] - box0_det[1])
                    box0_intersect_sz = 0

                    for idx1 in range(len(detectedObjects_all)):
                        if idx0 != idx1:
                            l_box0_det, l_box1_det, l_object_center, l_object_metrics = detectedObjects_all[idx1]
                            intersect0 = (max(l_box0_det[0], box0_det[0]), max(l_box0_det[1], box0_det[1]))
                            intersect1 = (min(l_box1_det[0], box1_det[0]), min(l_box1_det[1], box1_det[1]))

                            intersect_sz = (intersect1[0] - intersect0[0]) * (intersect1[1] - intersect0[1])
                            if intersect_sz > 0:
                                box0_intersect_sz += intersect_sz

                    if (float(box0_intersect_sz) / float(box0_sz)) < self.objMergeThreshold:
                        detectedObjects_all_new += [[box0_det, box1_det, object_center, object_metrics]]

                detectedObjects_all = detectedObjects_all_new
            
            for idx in range(len(self.objectsHistory)):
                obj_idx, is_assigned, box0, box1, hist_center, hist_metrics, size_history, object_label, object_class = self.objectsHistory[idx]
                self.objectsHistory[idx] = (obj_idx, False, box0, box1, hist_center, hist_metrics, size_history, object_label, object_class)

            # Match objects detected on current frame with objects history (object tracking list) or add new
            for box0_det, box1_det, object_center, object_metrics in detectedObjects_all:
                obj_history_idx = -1
                obj_history_distance = 0

                for idx in range(len(self.objectsHistory)):
                    obj_idx, is_assigned, box0, box1, hist_center, hist_metrics, size_history, object_label, object_class = self.objectsHistory[idx]
                    if ((not is_assigned) and
                        (object_center[0] >= box0[0]) and
                        (object_center[0] < box1[0]) and
                        (object_center[1] >= box0[1]) and
                        (object_center[1] < box1[1])):

                        cur_distance = ((object_center[0] - hist_center[0]) ** 2) + ((object_center[1] - hist_center[1]) ** 2)
                        if (obj_history_idx < 0) or (cur_distance < obj_history_distance):
                            obj_history_idx, obj_history_distance = idx, cur_distance
                
                if obj_history_idx >= 0:
                    obj_idx, is_assigned, box0, box1, hist_center, hist_metrics, size_history, object_label, object_class = self.objectsHistory[obj_history_idx]
                    self.objectsHistory[obj_history_idx] = (obj_idx, True, box0_det, box1_det, object_center, object_metrics, size_history + [(box0_det, box1_det, object_center)], object_label, object_class)
                else:
                    object_label = "Object"
                    if self.classifierDarkFlow != None:
                        is_dark_box_detected = False
                        min_dark_box_dist = 0
                        for box0_dark, box1_dark, center_dark, label_dark, confidence_dark in detectedObjects_dark:
                            if ((object_center[0] >= box0_dark[0]) and
                                (object_center[0] < box1_dark[0]) and
                                (object_center[1] >= box0_dark[1]) and
                                (object_center[1] < box1_dark[1])):

                                cur_distance = ((object_center[0] - center_dark[0]) ** 2) + ((object_center[1] - center_dark[1]) ** 2)
                                if (not is_dark_box_detected) or (cur_distance < min_dark_box_dist):
                                    is_dark_box_detected = True
                                    min_dark_box_dist = cur_distance
                                    object_label = label_dark

                    self.objLastLabel += 1
                    self.objectsHistory += [(self.objLastLabel, True, box0_det, box1_det, object_center, object_metrics, [(box0_det, box1_det, object_center)], 0, object_label)]
                    
                    if isFirstPrint:
                        isFirstPrint = False
                        print()
                    print("Next object #{} detected. Position: ({}, {}), size: ({}, {})".format(self.objLastLabel, object_center[0], object_center[1], box1_det[0] - box0_det[0], box1_det[1] - box0_det[1]))

            # Some objects from history can leave unmatched. Anyway, try to track object center and detect object. Or consider it is vanished (stop tracking)
            objects_history_new = []
            for obj_idx, is_assigned, box0, box1, hist_center, hist_metrics, size_history, object_label, object_class in self.objectsHistory:
                is_object_displayed = False

                if is_assigned:
                    if len(size_history) > self.objsDetFrames:
                        size_history = size_history[1:]

                    avg_size_x = 0
                    avg_size_y = 0
                    avg_cnt = 0
                    total_cnt = 0

                    for cur_box0, cur_box1, cur_obj_center in size_history[::-1]:
                        total_cnt += 1
                        if (avg_cnt > 0) and (total_cnt > self.objsDetAvgFrames):
                            break;

                        if ((cur_box1[0] - cur_box0[0]) > 0) and ((cur_box1[1] - cur_box0[1]) > 0):
                            last_box0, last_box1 = cur_box0, cur_box1
                            avg_cnt += 1
                            avg_size_x += cur_box1[1] - cur_box0[1]
                            avg_size_y += cur_box1[0] - cur_box0[0]
                    
                    if avg_cnt > 0:
                        if avg_cnt > 1:
                            avg_size_x = int(float(avg_size_x) / float(avg_cnt))
                            avg_size_y = int(float(avg_size_y) / float(avg_cnt))
                            box0 = (
                                int(hist_center[0] - (float(hist_center[0] - last_box0[0]) / float(last_box1[0] - last_box0[0]) * float(avg_size_y))),
                                int(hist_center[1] - (float(hist_center[1] - last_box0[1]) / float(last_box1[1] - last_box0[1]) * float(avg_size_x))))

                            box1 = (box0[0] + avg_size_y, box0[1] + avg_size_x)

                        objects_history_new += [(obj_idx, True, box0, box1, hist_center, hist_metrics, size_history, object_label, object_class)]
                        is_object_displayed = True
                else:
                    size_history += [((0,0), (0,0), (0,0))]
                    if len(size_history) > self.objsDetFrames:
                        size_history = size_history[1:]

                    for cur_box0, cur_box1, cur_obj_center in size_history:
                        if ((cur_box1[0] - cur_box0[0]) > 0) and ((cur_box1[1] - cur_box0[1]) > 0):
                            is_object_displayed = True
                            break;

                    if is_object_displayed:
                        det0 = (max(0, hist_center[0] - self.objsDetCrossFrameMaxDist), max(0, hist_center[1] - self.objsDetCrossFrameMaxDist))
                        det1 = (min(heatMap.shape[0], hist_center[0] + self.objsDetCrossFrameMaxDist), min(heatMap.shape[1], hist_center[1] + self.objsDetCrossFrameMaxDist))

                        det_map = heatMap[det0[0]:det1[0], det0[1]:det1[1]]

                        if np.max(det_map) > 0:
                            if (det1[1] - det0[1]) > self.heatMapConvolutionWindowSize:
                                new_center_x = np.argmax(np.convolve(np.sum(det_map, axis=0), conv_win, mode = 'valid')) + det0[1] + conv_win_half
                            else:
                                new_center_x = (det1[1] + det0[1]) // 2

                            if (det1[0] - det0[0]) > self.heatMapConvolutionWindowSize:
                                new_center_y = np.argmax(np.convolve(np.sum(det_map, axis=1), conv_win, mode = 'valid')) + det0[0] + conv_win_half
                            else:
                                new_center_y = (det1[0] + det0[0]) // 2

                            is_object_vanished = False
                            for l_obj_idx, l_is_assigned, l_box0, l_box1, l_hist_center, l_hist_metrics, l_size_history, l_object_label, l_object_class in self.objectsHistory:
                                if l_is_assigned:
                                    if (new_center_y >= l_box0[0]) and (new_center_y < l_box1[0]) and (new_center_x >= l_box0[1]) and (new_center_x < l_box1[1]):
                                        is_object_vanished = True
                                        break

                            if is_object_vanished:
                                is_object_displayed = False
                            else:
                                center_shift = (new_center_y - hist_center[0], new_center_x - hist_center[1])
                                box0 = (min(heatMap.shape[0], max(0, box0[0] + center_shift[0])), min(heatMap.shape[1], max(0, box0[1] + center_shift[1])))
                                box1 = (min(heatMap.shape[0], max(0, box1[0] + center_shift[0])), min(heatMap.shape[1], max(0, box1[1] + center_shift[1])))
                                hist_center = (new_center_y, new_center_x)
                                objects_history_new += [(obj_idx, True, box0, box1, hist_center, hist_metrics, size_history, object_label, object_class)]
                        else:
                            is_object_displayed = False

                if is_object_displayed:
                    if self.visualization:
                        heat_max, heat_sum, heat_sum_frame = hist_metrics

                        cv2.rectangle(self.visHeatMapImage, box0[::-1], box1[::-1], (255, 255, 0), 2)

                        cv2.putText(
                            self.visHeatMapImage,
                            '{}, {}, {}'.format(heat_max, heat_sum, heat_sum_frame),
                            (box0[1], box0[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            thickness = 1,
                            lineType = cv2.LINE_AA)
                else:
                    if isFirstPrint:
                        isFirstPrint = False
                        print()
                    print("Object #{} is vanished".format(obj_idx))

            self.objectsHistory = objects_history_new

            self._labelVehiclesInHistory();

    def _vehicleTrackingDarkFlow(
        self,
        img # Source image in OpenCV BGR format
        ):
        """
        Vehicle tracking algorithm
        """

        # Initialize just for consistency
        if self.visualization:
            self.visAllBoxesImage = img.copy()
            self.visVehicleBoxesImage = img.copy()

            self.visHeatMapImage = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
            self.visHeatMapImage[:,:,0] = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY) // 2
            self.visHeatMapImage[:,:,1] = self.visHeatMapImage[:,:,0]
            self.visHeatMapImage[:,:,2] = self.visHeatMapImage[:,:,0]
        
        img_shape = img.shape
        img_dark = np.zeros(shape=(max(img_shape[0], img_shape[1]), max(img_shape[0], img_shape[1]), img_shape[2]), dtype = np.uint8)
        img_dark[:img_shape[0], :img_shape[1]] = img
        
        isFirstPrint = True
        detectedObjects_all = []
        for detected_obj in self.classifierDarkFlow.return_predict(img_dark):
            object_label = detected_obj["label"]
            confidence = detected_obj["confidence"]
            box0 = (detected_obj["topleft"]["y"], detected_obj["topleft"]["x"])
            box1 = (detected_obj["bottomright"]["y"], detected_obj["bottomright"]["x"])

            if ((object_label in self.useDarkFlowLabels) and
                (box0[0] <= img_shape[0]) and
                (box0[1] <= img_shape[1]) and
                (box1[0] <= img_shape[0]) and
                (box1[1] <= img_shape[1])):

                detectedObjects_all += [[box0, box1, (int((box0[0] + box1[0]) / 2.0), int((box0[1] + box1[1]) / 2.0)), object_label, confidence]]
            
        for idx in range(len(self.objectsHistory)):
            obj_idx, is_assigned, box0, box1, hist_center, hist_metrics, size_history, object_label, object_class = self.objectsHistory[idx]
            self.objectsHistory[idx] = (obj_idx, False, box0, box1, hist_center, hist_metrics, size_history, object_label, object_class)

        # Match objects detected on current frame with objects history (object tracking list) or add new
        for box0_det, box1_det, object_center, object_class_det, object_confidence_det in detectedObjects_all:
            obj_history_idx = -1
            obj_history_distance = 0

            for idx in range(len(self.objectsHistory)):
                obj_idx, is_assigned, box0, box1, hist_center, hist_metrics, size_history, object_label, object_class = self.objectsHistory[idx]
                if ((not is_assigned) and
                    #(object_class_det == object_class) and
                    (object_center[0] >= box0[0]) and
                    (object_center[0] < box1[0]) and
                    (object_center[1] >= box0[1]) and
                    (object_center[1] < box1[1])):

                    cur_distance = ((object_center[0] - hist_center[0]) ** 2) + ((object_center[1] - hist_center[1]) ** 2)
                    if (obj_history_idx < 0) or (cur_distance < obj_history_distance):
                        obj_history_idx, obj_history_distance = idx, cur_distance
                
            if obj_history_idx >= 0:
                obj_idx, is_assigned, box0, box1, hist_center, hist_metrics, size_history, object_label, object_class = self.objectsHistory[obj_history_idx]
                self.objectsHistory[obj_history_idx] = (obj_idx, True, box0_det, box1_det, object_center, None, size_history + [(box0_det, box1_det, object_center)], object_label, object_class)
            else:
                self.objLastLabel += 1
                self.objectsHistory += [(self.objLastLabel, True, box0_det, box1_det, object_center, None, [(box0_det, box1_det, object_center)], 0, object_class_det)]
                    
                if isFirstPrint:
                    isFirstPrint = False
                    print()
                print("Next object #{} detected. Position: ({}, {}), size: ({}, {})".format(self.objLastLabel, object_center[0], object_center[1], box1_det[0] - box0_det[0], box1_det[1] - box0_det[1]))

        # Some objects from history can leave unmatched. Anyway, try to track object center and detect object. Or consider it is vanished (stop tracking)
        objects_history_new = []
        for obj_idx, is_assigned, box0, box1, hist_center, hist_metrics, size_history, object_label, object_class in self.objectsHistory:
            is_object_displayed = False

            if is_assigned:
                if len(size_history) > self.objsDetFrames:
                    size_history = size_history[1:]

                avg_size_x = 0
                avg_size_y = 0
                avg_cnt = 0
                total_cnt = 0

                for cur_box0, cur_box1, cur_obj_center in size_history[::-1]:
                    total_cnt += 1
                    if (avg_cnt > 0) and (total_cnt > self.objsDetAvgFrames):
                        break;

                    if ((cur_box1[0] - cur_box0[0]) > 0) and ((cur_box1[1] - cur_box0[1]) > 0):
                        last_box0, last_box1 = cur_box0, cur_box1
                        avg_cnt += 1
                        avg_size_x += cur_box1[1] - cur_box0[1]
                        avg_size_y += cur_box1[0] - cur_box0[0]
                    
                if avg_cnt > 0:
                    if avg_cnt > 1:
                        avg_size_x = int(float(avg_size_x) / float(avg_cnt))
                        avg_size_y = int(float(avg_size_y) / float(avg_cnt))
                        box0 = (
                            int(hist_center[0] - (float(hist_center[0] - last_box0[0]) / float(last_box1[0] - last_box0[0]) * float(avg_size_y))),
                            int(hist_center[1] - (float(hist_center[1] - last_box0[1]) / float(last_box1[1] - last_box0[1]) * float(avg_size_x))))

                        box1 = (box0[0] + avg_size_y, box0[1] + avg_size_x)

                    objects_history_new += [(obj_idx, True, box0, box1, hist_center, hist_metrics, size_history, object_label, object_class)]
                    is_object_displayed = True
            else:
                size_history += [((0,0), (0,0), (0,0))]
                if len(size_history) > self.objsDetFrames:
                    size_history = size_history[1:]

                for cur_box0, cur_box1, cur_obj_center in size_history:
                    if ((cur_box1[0] - cur_box0[0]) > 0) and ((cur_box1[1] - cur_box0[1]) > 0):
                        is_object_displayed = True
                        break;

                if is_object_displayed:
                    objects_history_new += [(obj_idx, False, box0, box1, hist_center, hist_metrics, size_history, object_label, object_class)]

            if not is_object_displayed:
                if isFirstPrint:
                    isFirstPrint = False
                    print()
                print("Object #{} is vanished".format(obj_idx))

        self.objectsHistory = objects_history_new

        self._labelVehiclesInHistory();

    def processFrame(
        self,
        img # Source image in OpenCV BGR format
        ):
        """
        Frame processing entry point
        """

        if self.visualization:
            self.visOrigImage = img.copy()

        # Do image undistortion
        img_undist = self.camera.undistort(img)

        if self.visualization:
            self.visUndistortImage = img_undist.copy()

        # Detect edges and do image binary
        img_binary = self.camera.makeBinary(img_undist)

        if self.visualization:
            self.visBinaryImage = self.camera.binaryToImg(img_binary)

        # Perform perspective transformation
        top_bin = self.camera.perspectiveTransformBinary(img_binary);

        if self.visualization:
            self.visTopViewBinaryImage = self.camera.binaryToImg(top_bin)

        # Detect lane lines
        self._detectLines(top_bin)

        # Vehicle tracking
        if (self.useDarkFlowMode == 0) and (self.classifierDarkFlow != None):
            self._vehicleTrackingDarkFlow(img_undist)
        else:
            self._vehicleTracking(img_undist)

        # Calculate parameters and annotate image
        self.isImageAnnotated, self.visImageAnnotated = self._calculateParameters(img_undist, top_bin.shape)

        return self.visImageAnnotated



