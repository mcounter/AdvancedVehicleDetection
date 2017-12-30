import os
import pickle
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree

from DeepDataEngineFeatures import DeepDataEngineFeatures

class ImageClassifier:
    """
    Image classifier
    """

    def __init__(
        self,
        instanceName, # Classifier instance name
        storageDir = './config', # Location of camera parameters
        ):

        self.instanceName = instanceName
        self.storageDir = storageDir
        
        self.data_scaler = None
        self.data_classifier = None

        self._unpickle_parameters()

    def _unpickle_parameters(self):
        """
        Load parameters.
        """

        is_loaded = False

        try:
            with open('{}/{}.dat'.format(self.storageDir, self.instanceName), mode='rb') as f:
                data_set = pickle.load(f)
    
            self.data_scaler = data_set['data_scaler']
            self.data_classifier = data_set['data_classifier']

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
            os.makedirs(self.storageDir)
        except:
            pass

        try:
            with open('{}/{}.dat'.format(self.storageDir, self.instanceName), mode='wb') as f:
                data_set = {
                    'data_scaler' : self.data_scaler,
                    'data_classifier' : self.data_classifier}

                pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)

            is_saved = True;
        except:
            is_saved = False

        return is_saved

    def fit(
        self,
        train_data, # Train data DeepDataEngineFeatures object
        classifierType, # Type of classifier: SVC, LinearSVC, DTree
        C = 1.0, # Classifier penalty factor C
        kernel = 'rbf', # Classifier kernel
        gamma = 'auto', # Gamma parameter for some kernels like RBF
        degree = 3, # Degree parameter for Polynomial kernel
        coef0 = 0.0, # Coef0 parameter for Polynomial kernel
        min_samples_split = 2, # Parameter for Decision trees
        cache_size = 1024, # Cache size
        ):
        """
        Train classifier and save on disk
        """

        if isinstance(train_data, DeepDataEngineFeatures):
            x_data, y_data = train_data.readAllData()
        else:
            x_data, y_data = train_data

        x_scaler = sklearn.preprocessing.StandardScaler().fit(x_data)
        x_data = x_scaler.transform(x_data)
        
        if classifierType == 'LinearSVC':
            clf = sklearn.svm.LinearSVC(C = C)
        elif classifierType == 'SVC':
            if gamma != 'auto':
                gamma = gamma * (1.0 / len(x_data))

            clf = sklearn.svm.SVC(kernel = kernel, C = C, gamma = gamma, degree = degree, coef0 = coef0, cache_size = cache_size)
        elif classifierType == 'DTree':
            clf = sklearn.tree.DecisionTreeClassifier(min_samples_split = min_samples_split)
        else:
            raise Exception("Unsupported classifier type {}.".format(classifierType))

        clf.fit(x_data, y_data)

        self.data_scaler = x_scaler
        self.data_classifier = clf

        self._pickle_parameters()

    def score(
        self,
        valid_data # Evaluate data set acuracy
        ):
        """
        Evaluate classifier
        """

        if isinstance(valid_data, DeepDataEngineFeatures):
            x_data, y_data = valid_data.readAllData()
        else:
            x_data, y_data = valid_data

        x_data = self.data_scaler.transform(x_data)

        return self.data_classifier.score(x_data, y_data)

    def predict(
        self,
        x_data # Feature matrix
        ):
        """
        Predict label value with trained classifier
        """

        if isinstance(x_data, DeepDataEngineFeatures):
            x_data, y_data = valid_data.readAllData()

        x_data = self.data_scaler.transform(x_data)

        return self.data_classifier.predict(x_data)


