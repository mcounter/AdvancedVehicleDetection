import numpy as np
import cv2
import pickle
import csv
import os
import glob
import matplotlib.image as mpimg

from DeepDataEngine import DeepDataEngine
from ImageEngine import ImageEngine

class DeepDataEngineFeatures(DeepDataEngine):
    """
    Data storage for image feature vectors.
    """

    def __init__(
        self,
        set_name, # Name of data set, like 'train' or 'valid'
        storage_dir = './deep_storage_features', # Folder where data files will be stored
        mem_size = 512 * 1024 * 1024, # Desired maximum size of each file in data storage
        batch_size = 256 # Batch size (used in training and validation process)
        ):
        """
        Initialize class instance
        """
        DeepDataEngine.__init__(self, set_name, storage_dir, mem_size, batch_size)

        self.storage_measure_filter = -1

    def createGenerationPlan(
        filePathVehicles, # Path to vehicles samples
        filePathNonVehicles, # Path to non-vehicles samples
        testSplit = None, # Split test set (0..1)
        groupSplitThreshold = 10000, # Metric threshold to split between groups
        augmentData = False, # Augment data
        normalizeDataGroups = False, # Split data per groups and do normalization
        ):
        """
        Static method, creates generation plan from set of images.
        """

        # Add image in generation plan and calculate it metric for normalization
        def processImageSet(images, img_label, imgEng, img_shape = None, formatFilter = ['.JPG', '.JPEG', '.PNG']):
            local_gen_plan = []

            for image_path in images:
                if os.path.isfile(image_path) and (str(os.path.splitext(image_path)[1]).upper() in formatFilter):
                    img = cv2.imread(image_path)

                    if img_shape is None:
                        img_shape = img.shape

                    cur_img_shape = img.shape
                    if (img_shape[0] != cur_img_shape[0]) or (img_shape[1] != cur_img_shape[1]):
                        img = cv2.resize(img, (img_shape[1], img_shape[0]), interpolation = cv2.INTER_CUBIC)

                    features = imgEng.getImageFeatures(img, visualise = False)
                    img_metric = features[0][2]

                    local_gen_plan += [[(image_path, 0), img_label, img_metric]]
                    if augmentData:
                        local_gen_plan += [[(image_path, 1), img_label, img_metric]]
                    
                    #if len(local_gen_plan) >= np.random.randint(100, 110):
                    #    break

            return local_gen_plan, img_shape

        # Comparator function
        def compare_metric(x1, x2):
            metric1 = x1[2]
            metric2 = x2[2]

            sz1 = len(metric1)
            sz2 = len(metric2)
            sz = min(sz1, sz2)

            for idx in range(sz):
                if metric1[idx] < metric2[idx]:
                    return -1
                elif metric1[idx] > metric2[idx]:
                    return 1

            if sz1 < sz2:
                return -1

            if sz1 > sz2:
                return 1

            return 0

        # Comparator wrapper class
        def cmp_to_key(mycmp):
            class K(object):
                def __init__(self, obj, *args):
                    self.obj = obj
                def __lt__(self, other):
                    return mycmp(self.obj, other.obj) < 0
                def __gt__(self, other):
                    return mycmp(self.obj, other.obj) > 0
                def __eq__(self, other):
                    return mycmp(self.obj, other.obj) == 0
                def __le__(self, other):
                    return mycmp(self.obj, other.obj) <= 0
                def __ge__(self, other):
                    return mycmp(self.obj, other.obj) >= 0
                def __ne__(self, other):
                    return mycmp(self.obj, other.obj) != 0
            return K

        # Split data on groups
        def get_data_groups(gen_plan):
            train_plan = []
            test_plan = []

            if len(gen_plan) > 0:
                groups = []

                gen_plan = sorted(gen_plan, key=cmp_to_key(compare_metric))
                last_group = [gen_plan[0]]
                last_metric = np.array(gen_plan[0][2])

                for idx in range(1, len(gen_plan)):
                    cur_metric = np.array(gen_plan[idx][2])

                    diff = np.mean(np.square(last_metric - cur_metric))
                    if diff >= groupSplitThreshold:
                        groups += [last_group]
                        last_group = []
                    
                    last_group += [gen_plan[idx]]
                    last_metric = cur_metric

                groups += [last_group]

            return groups, len(gen_plan)

        # Normalaze groups
        def normalize_group(data_groups, group_size):
            data_groups_new = []
            for cur_group in data_groups:
                if len(cur_group) > group_size:
                    np.random.shuffle(cur_group)
                    cur_group = cur_group[:group_size]
                else:
                    while len(cur_group) < group_size:
                        diff = group_size - len(cur_group)
                        np.random.shuffle(cur_group)
                        cur_group += cur_group[:diff]

                data_groups_new += [cur_group]
            return data_groups_new

        # Merge groups to one plan
        def groups_to_plan(data_groups):
            gen_plan = []
            for cur_group in data_groups:
                gen_plan += cur_group

            return gen_plan

        # Split test data equally from all groups
        def split_test_data(data_groups, split_num):
            gen_plan_train = []
            gen_plan_test = []

            cur_num = split_num
            np.random.shuffle(data_groups)

            for cur_group in data_groups:
                if len(cur_group) <= cur_num:
                    cur_num -= len(cur_group)
                    gen_plan_test += cur_group
                else:
                    gen_plan_train += cur_group

            return gen_plan_train, gen_plan_test

        img_shape = None

        # Use image engine to retrieve image metric - short form of feature vector
        imgEng = ImageEngine(
            load_setup = False,
            color_space = 'YUV',
            hog_features = True,
            hog_channels = [0],
            hog_orientations = 9,
            hog_pix_per_cell = 16,
            hog_cells_per_block = 1,
            hog_block_norm = 'L2-Hys',
            hog_transform_sqrt = False,
            spatial_features = False,
            histogram_features = True,
            histogram_channels = [1,2],
            histogram_bins = 16)

        vehicle_plan, img_shape = processImageSet(glob.glob(filePathVehicles + '/**/*.*', recursive = True), 1, imgEng, img_shape = img_shape)
        nonvehicle_plan, img_shape = processImageSet(glob.glob(filePathNonVehicles + '/**/*.*', recursive = True), 0, imgEng, img_shape = img_shape)

        vehicle_groups, vehicle_num = get_data_groups(vehicle_plan)
        if normalizeDataGroups:
            vehicle_groups = normalize_group(vehicle_groups, vehicle_num // len(vehicle_groups))

        nonvehicle_groups, nonvehicle_num = get_data_groups(nonvehicle_plan)
        if normalizeDataGroups:
            nonvehicle_groups = normalize_group(nonvehicle_groups, nonvehicle_num // len(nonvehicle_groups))

        if testSplit is None:
            return groups_to_plan(vehicle_groups) + groups_to_plan(nonvehicle_groups)

        vehicle_train, vehicle_test = split_test_data(vehicle_groups, int(vehicle_num * testSplit))
        nonvehicle_train, nonvehicle_test = split_test_data(nonvehicle_groups, int(vehicle_num * testSplit))

        return vehicle_train + nonvehicle_train, vehicle_test + nonvehicle_test

    def createStorage(
        self,
        generation_plan, # Generation plan - python list
        imgEng = None, # Image engine
        override = True): # Indicates that old storage must be deleted if exists. Otherwise it will be augmented with new files.
        """
        Create data storage from generation plan.
        """

        if len(generation_plan) <= 0:
            return

        self._loadStorage()

        if override:
            self._delete_storage()

        # In case storage already has some data, find index of next file.
        file_idx = -1
        for file_name in self.storage_files:
            cur_idx = int(file_name[-10:-4])
            file_idx = max(file_idx, cur_idx)

        file_idx += 1

        # Read first image to determine shape
        image = cv2.imread(generation_plan[0][0][0])
        image_shape = image.shape

        # Create empty X and Y buffers of fixed size. These buffers will be populated with inbound and outbound data and pickled to disk.
        features = imgEng.getImageFeatures(image, visualise = False)
        features_size = features[0][2].shape[0]
        features_bytes = features[0][2].nbytes
        buf_size = int(self.mem_size / features_bytes)

        x_buf = np.zeros((buf_size, features_size), dtype = np.float64)
        y_buf = np.zeros((buf_size, 1), dtype = np.uint8)

        # Shuffle generation plan to have random distribution across all data files.
        np.random.shuffle(generation_plan)
        
        buf_pos = 0

        for plan_line in generation_plan:
            img_path, img_flip= plan_line[0]
            y_label = plan_line[1]
            
            # Load images from disk
            image = cv2.imread(img_path)
            cur_shape = image.shape
            if (image_shape[0] != cur_shape[0]) or (image_shape[1] != cur_shape[1]):
                image = cv2.resize(image, (image_shape[1], image_shape[0]), interpolation = cv2.INTER_CUBIC)

            if img_flip:
                image = cv2.flip(image, 1)

            features = imgEng.getImageFeatures(image, visualise = False)
            x_buf[buf_pos] = features[0][2]
            y_buf[buf_pos, 0] = y_label
                        
            buf_pos += 1

            if buf_pos >= buf_size:
                # Pickle buffer to file
                self._pickleToFile('{}/{}_{:0>6}.dat'.format(self.storage_dir, self.set_name, file_idx), x_buf, y_buf)
                self.storage_size += buf_size
                self._pickleStorageSize(self.storage_size)
                file_idx += 1
                buf_pos = 0

        if buf_pos > 0:
            # Pickle non-full last buffer to file
            x_buf = x_buf[:buf_pos]
            y_buf = y_buf[:buf_pos]
            self._pickleToFile('{}/{}_{:0>6}.dat'.format(self.storage_dir, self.set_name, file_idx), x_buf, y_buf)
            self.storage_size += buf_pos
            self._pickleStorageSize(self.storage_size)

        # Initialize storage for reading
        self._loadStorage()