import numpy as np

class LaneLine():
    """
    Definition of lane line. Is used to track lines separately from frame to frame
    """

    def __init__(
        self,
        avgFrameNum = 15, # Number of frames averaged
        minDistanceToFilterPx = 150 # Minimal distance from pixel to lane to be rejected
        ):

        self.avgFrameNum = avgFrameNum
        self.minDistanceToFilterPx = minDistanceToFilterPx

        self.isLineDetected = False
        self.lineShape = np.zeros(3, dtype = np.float64) # a * y**2 + b * y + c
        self.lineShapeWorld = np.zeros(3, dtype = np.float64) # A * y**2 + B * y + C
        self.histFrames = 0
        self.histFramesNonEmpty = 0
        self.histPoints = []

    def _filterPoints(
        self,
        points, # Set of points to be filtered
        shape # Curve shape
        ):
        """
        Filter out spurious values
        """

        if len(points) > 0:
            histPoints = points.T
            x_points = shape[0] * (histPoints[0] ** 2) + shape[1] * histPoints[0] + shape[2]
            filter = np.absolute(histPoints[1] - x_points) < self.minDistanceToFilterPx
            histPoints = histPoints.T
            histPoints = histPoints[filter]

            return histPoints
        
        return points

    def checkFilterLinePoints(
        self,
        points, # Array of points of new detected line
        checkPoints, # (Y, X1, X2) - check if calculated lane shape is started from expecetd place
        checkCurvaturePoints, # Y coordinate to check curvature
        maxCurvatureChange, # Maximal power of allowed line curvature change
        minPointsNum = 3, # Minumum number of points
        ):
        """
        Validate if points detected from frame can be used for lane line calculation
        """

        if len(points) < minPointsNum:
            return False, points, []

        # Check single line shape
        line_detected, line_shape = self._calcLineShape(points)
        if not line_detected:
            return False, points, []

        # Filter spurious values
        points_new = self._filterPoints(points, line_shape)

        # Check if minimal number of values remain
        if len(points_new) < minPointsNum:
            return False, points, []

        # If points was filtered, recalculate line shape
        if len(points_new) != len(points):
            line_detected, line_shape = self._calcLineShape(points)
            if not line_detected:
                return False, points, []

        # Check if line starts from expected place
        x0 = line_shape[0] * (checkPoints[0] ** 2) + line_shape[1] * checkPoints[0] + line_shape[2]
        if (x0 < checkPoints[1]) | (x0 > checkPoints[2]):
            return False, points, []

        # Check line curvature
        # Average history line and single line must not have very different curvature
        if self.isLineDetected and (line_shape[0] != 0) and (self.lineShape[0] != 0):
            rad1 = np.log10(np.min(((1 + (2 * line_shape[0] * checkCurvaturePoints + line_shape[1]) ** 2) ** 1.5) / np.absolute(2 * line_shape[0])))
            rad2 = np.log10(np.min(((1 + (2 * self.lineShape[0] * checkCurvaturePoints + self.lineShape[1]) ** 2) ** 1.5) / np.absolute(2 * self.lineShape[0])))

            if (rad1 != 0) and (rad2 != 0):
                change_power = 100.0**(max(rad1, rad2) / min(rad1, rad2)) / 100.0
                if change_power > maxCurvatureChange:
                    return False, points, []

        # Combine line shape with history points and calculate average shape
        line_detected, line_shape = self.calcLineShape(extraPoints = points_new)
        if not line_detected:
            return False, points, []

        # Check if line starts from expected place
        x0 = line_shape[0] * (checkPoints[0] ** 2) + line_shape[1] * checkPoints[0] + line_shape[2]
        if (x0 < checkPoints[1]) | (x0 > checkPoints[2]):
            return False, points, []

        return True, points_new, line_shape

    def addLinePoints(
        self,
        points, # Array of points of new detected line to be added
        ):
        """
        Add point from new frame to history array
        """

        if len(points) > 0:
            self.histFramesNonEmpty += 1

        self.histPoints += [points]
        self.histFrames += 1

        if self.histFrames > self.avgFrameNum:
            if len(self.histPoints[0]) > 0:
                self.histFramesNonEmpty -= 1

            self.histPoints = self.histPoints[1:]
            self.histFrames -= 1

    def _calcLineShape(
        self,
        points, # Array of points
        transformation = (1.0, 1.0) # Transformation of pixels to other system of coordinates
        ):
        """
        Calculate lane shape from set of points
        """
        
        if len(points) < 3: # Need minimum 3 points
            return False, np.zeros(3, dtype = np.float64)

        histPointsMerged = points.T

        if histPointsMerged.shape[0] >= 3:
            # With weigths
            line_shape = np.polyfit(histPointsMerged[0] * transformation[0], histPointsMerged[1] * transformation[1], 2, w = histPointsMerged[2])
        else:
            # Without weights
            line_shape = np.polyfit(histPointsMerged[0] * transformation[0], histPointsMerged[1] * transformation[1], 2)
        
        return True, line_shape

    def calcLineShape(
        self,
        transformation = (1.0, 1.0), # Transformation of pixels to other system of coordinates
        extraPoints = [] # Extra points to be added to history points
        ):
        """
        Calculate lane shape from set of points collected in several previous frames
        """

        histPointsMerged = []
        for points in self.histPoints:
            if len(points) > 0:
                if len(histPointsMerged) > 0:
                    histPointsMerged = np.append(histPointsMerged, points, axis=0)
                else:
                    histPointsMerged = points.copy()

        if len(extraPoints) > 0:
            if len(histPointsMerged) > 0:
                histPointsMerged = np.append(histPointsMerged, extraPoints, axis=0)
            else:
                histPointsMerged = extraPoints.copy()

        return self._calcLineShape(histPointsMerged, transformation)

    def updateLineShape(
        self,
        transformation = (1.0, 1.0)
        ):
        """
        Claculate line shape for both systems of coordinates.
        """

        self.isLineDetected, self.lineShape = self.calcLineShape()
        isLineDetectedWorld, self.lineShapeWorld = self.calcLineShape(transformation)

    def mutualLaneShapeCorrection(
        laneLine1, # Lane line 1
        laneLine2, # Lane line 2
        fixed_point_y, # Point not changed after lane shape correction
        fixed_point_world_y, # Point not changed after lane shape correction (second system of coordinates)
        ):
        """
        In case 2 lines are detected we can correct lane shape to do it average.
        Static method.
        """

        if laneLine1.isLineDetected and laneLine2.isLineDetected:
            a1 = (laneLine1.lineShape[0] + laneLine2.lineShape[0]) / 2.0
            b1 = (laneLine1.lineShape[1] + laneLine2.lineShape[1]) / 2.0
            c1 = (laneLine1.lineShape[0] - a1) * (fixed_point_y ** 2) + (laneLine1.lineShape[1] - b1) * fixed_point_y + laneLine1.lineShape[2]
            c2 = (laneLine2.lineShape[0] - a1) * (fixed_point_y ** 2) + (laneLine2.lineShape[1] - b1) * fixed_point_y + laneLine2.lineShape[2]

            laneLine1.lineShape = [a1, b1, c1]
            laneLine2.lineShape = [a1, b1, c2]

            a1 = (laneLine1.lineShapeWorld[0] + laneLine2.lineShapeWorld[0]) / 2.0
            b1 = (laneLine1.lineShapeWorld[1] + laneLine2.lineShapeWorld[1]) / 2.0
            c1 = (laneLine1.lineShapeWorld[0] - a1) * (fixed_point_world_y ** 2) + (laneLine1.lineShapeWorld[1] - b1) * fixed_point_world_y + laneLine1.lineShapeWorld[2]
            c2 = (laneLine2.lineShapeWorld[0] - a1) * (fixed_point_world_y ** 2) + (laneLine2.lineShapeWorld[1] - b1) * fixed_point_world_y + laneLine2.lineShapeWorld[2]

            laneLine1.lineShapeWorld = [a1, b1, c1]
            laneLine2.lineShapeWorld = [a1, b1, c2]
