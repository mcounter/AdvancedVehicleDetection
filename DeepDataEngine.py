import numpy as np
import cv2
import pickle
import csv
import os
import glob
import matplotlib.image as mpimg

class DeepDataEngine:
    """
    Base data storage management functions.
    """

    def __init__(
        self,
        set_name, # Name of data set, like 'train' or 'valid'
        storage_dir = './deep_storage', # Folder where data files will be stored
        mem_size = 512 * 1024 * 1024, # Desired maximum size of each file in data storage
        batch_size = 256 # Batch size (used in training and validation process)
        ):
        """
        Initialize class instance
        """

        self.set_name = set_name
        self.storage_dir = storage_dir
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.storage_files = []
        self.storage_file_active = -1
        self.storage_buf_x = None
        self.storage_buf_y = None
        self.storage_measure_filter = None
        self.storage_size = -1

    def _unpickleFromFile(self, file_path):
        """
        Unpickle file with data.
        """

        with open(file_path, mode='rb') as f:
            data_set = pickle.load(f)
    
        X_data, y_data = data_set['features'], data_set['labels']

        assert(len(X_data) == len(y_data))

        return X_data, y_data

    def _pickleToFile(self, file_path, X_data, y_data):
        """
        Pickle file with data.
        """

        with open(file_path, mode='wb') as f:
            data_set = {'features' : X_data, 'labels' : y_data}
            pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)

    def _unpickleStorageSize(self):
        """
        Unpickle file with storage size (cached to avoid reload all files to calculate it).
        """

        storage_size = 0

        try:
            with open('{}/{}_ext.ext'.format(self.storage_dir, self.set_name), mode='rb') as f:
                data_set = pickle.load(f)
    
            storage_size = data_set['storage_size']
        except:
            pass

        return storage_size

    def _pickleStorageSize(self, storage_size):
        """
        Unpickle file with storage size (cached to avoid reload all files to calculate it).
        """

        with open('{}/{}_ext.ext'.format(self.storage_dir, self.set_name), mode='wb') as f:
            data_set = {'storage_size' : storage_size}
            pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)

    def _loadStorage(self):
        """
        Load information about data storage - size and files with data.
        In this way storage is initialized for reading.
        """

        self.storage_files = []
        self.storage_file_active = -1

        set_file_base_name = self.set_name + '_';

        try:
            os.makedirs(self.storage_dir)
        except:
            pass

        try:
            for file_name in os.listdir(self.storage_dir):
                file_path = self.storage_dir + '/' + file_name
                if (os.path.exists(file_path) and
                    os.path.isfile(file_path) and
                    (str(os.path.splitext(file_path)[1]).upper() in ('.DAT')) and
                    (str(file_name[:len(set_file_base_name)]).upper() == str(set_file_base_name).upper())):
                    
                    self.storage_files += [file_path]

        except:
            pass

        self.storage_size = self._unpickleStorageSize()

    def _delete_storage(self):
        """
        Delete data storage.
        """

        for file_name in self.storage_files:
            try:
                os.remove(file_name)
            except:
                pass

        self.storage_files = []
        self.storage_size = 0
        self._pickleStorageSize(self.storage_size)

    def initStorage(self):
        """
        Initialize storage for reading, call _loadStorage().
        """

        self._loadStorage()

    def _readNextStorageFile(self):
        """
        Read next storage file from disk.
        """

        self.storage_buf_x, self.storage_buf_y = self._unpickleFromFile(self.storage_files[self.storage_file_active])

        if self.storage_measure_filter != None:
            if self.storage_measure_filter == -1:
                self.storage_buf_y = self.storage_buf_y.reshape(-1)
            else:
                self.storage_buf_y = self.storage_buf_y[:, self.storage_measure_filter]

        permutation = np.random.permutation(len(self.storage_buf_x))
        self.storage_buf_x = self.storage_buf_x[permutation]
        self.storage_buf_y = self.storage_buf_y[permutation]

    def initRead(self):
        """
        Initialize data reading - shuffle file list and read next non-empty file.
        """

        np.random.shuffle(self.storage_files)
        self.storage_file_active = 0
        self._readNextStorageFile()

        while len(self.storage_buf_x) <= 0:
            if (self.storage_file_active + 1) < len(self.storage_files):
                self.storage_file_active += 1
                self._readNextStorageFile()
            else:
                break

    def canReadMore(self):
        """
        Determine that data storage is fully read and to read more need be initialized with initRead() function.
        """

        return len(self.storage_buf_x) > 0

    def readNext(self):
        """
        Read next batch for training or validation.
        If end of current file is reached, next file is automatically read from disk and append to current buffer.
        Only one last buffer per epoch can have size less that batch_size.
        """

        x_data = np.array(self.storage_buf_x[:self.batch_size])
        y_data = np.array(self.storage_buf_y[:self.batch_size])

        batch_buf_size = len(x_data)
        self.storage_buf_x = self.storage_buf_x[batch_buf_size:]
        self.storage_buf_y = self.storage_buf_y[batch_buf_size:]

        try_read_next = True

        while try_read_next:
            try_read_next = False

            if len(self.storage_buf_x) <= 0:
                if (self.storage_file_active + 1) < len(self.storage_files):
                    self.storage_file_active += 1
                    self._readNextStorageFile()

                    if len(self.storage_buf_x) > 0:
                        if len(x_data) <= 0:
                            x_data = np.array(self.storage_buf_x[:self.batch_size])
                            y_data = np.array(self.storage_buf_y[:self.batch_size])

                            batch_buf_size = len(x_data)
                            self.storage_buf_x = self.storage_buf_x[batch_buf_size:]
                            self.storage_buf_y = self.storage_buf_y[batch_buf_size:]
                        elif len(x_data) < self.batch_size:
                            size_orig = len(x_data)
                            batch_remain = self.batch_size - size_orig
                            x_data = np.append(x_data, np.array(self.storage_buf_x[:batch_remain]), axis = 0)
                            y_data = np.append(y_data, np.array(self.storage_buf_y[:batch_remain]), axis = 0)

                            batch_buf_size = len(x_data) - size_orig
                            self.storage_buf_x = self.storage_buf_x[batch_buf_size:]
                            self.storage_buf_y = self.storage_buf_y[batch_buf_size:]

                    if len(self.storage_buf_x) <= 0:
                        try_read_next = True

        return x_data, y_data

    def _generator(self):
        """
        Infinite generator compatible with Keras
        """

        while True:
            self.initRead()
            while self.canReadMore():
                yield self.readNext()

    def getGenerator(self):
        """
        Return number of unique batches can be read per epoch and generator instance. Compatible with Keras.
        """

        gen_step_max = self.storage_size // self.batch_size
        if (self.storage_size % self.batch_size) > 0:
            gen_step_max += 1

        return gen_step_max, self._generator()

    def getInOutShape(self):
        """
        Get shape of input and output data.
        """

        self.initRead()
        if self.canReadMore():
            x_data, y_data = self.readNext()
            return x_data.shape[1:], y_data.shape[1:]

        return (), ()

    def readAllData(self):
        """
        Read whole data amount.
        """

        self.initRead()

        x_data_all, y_data_all = self.readNext()

        while self.canReadMore():
            x_data, y_data = self.readNext()

            x_data_all = np.append(x_data_all, x_data, axis = 0)
            y_data_all = np.append(y_data_all, y_data, axis = 0)

        return x_data_all, y_data_all


