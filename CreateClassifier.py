import time
import numpy as np

from DeepDataEngineFeatures import DeepDataEngineFeatures
from ImageClassifier import ImageClassifier
from ImageEngine import ImageEngine

# Set this parameter True to recreate feature storage first
recreateStorage = False
# Set this parameter True to train on full data set.
# When optimal parameters found, use this option
full_train = True

train_data = DeepDataEngineFeatures('train')
valid_data = DeepDataEngineFeatures('valid')

if recreateStorage:
    # Initialize image engine. Use default setup or saved in file
    imgEng = ImageEngine(load_setup = True)

    train_plan, test_plan = DeepDataEngineFeatures.createGenerationPlan('./img_data/vehicles', './img_data/non_vehicles', testSplit = 0.2)

    train_data.createStorage(train_plan, imgEng = imgEng, override = True)
    valid_data.createStorage(test_plan, imgEng = imgEng, override = True)
else:
    train_data.initStorage()
    valid_data.initStorage()

x_data_train, y_data_train = train_data.readAllData()
x_data_valid, y_data_valid = valid_data.readAllData()

if full_train:
    x_data_train = np.vstack((x_data_train, x_data_valid))
    y_data_train = np.append(y_data_train, y_data_valid)

# Train LinearSVC classifier
clfLinearSVC = ImageClassifier('clf_linear')

t=time.time()
clfLinearSVC.fit((x_data_train, y_data_train), classifierType = 'LinearSVC', C=0.01)
print(round(time.time()-t, 2), 'seconds to train classifier...')

t=time.time()
score = clfLinearSVC.score((x_data_valid, y_data_valid))
print(round(time.time()-t, 2), 'seconds to evaluate accuracy...')
print('Test accuracy of classifier = ', round(score, 4))

# Train SVC with RBF kernel classifier
clfSVC = ImageClassifier('clf_rbf')

t=time.time()
clfSVC.fit((x_data_train, y_data_train), classifierType = 'SVC', kernel = 'rbf', C=100)
print(round(time.time()-t, 2), 'seconds to train classifier...')

t=time.time()
score = clfSVC.score((x_data_valid, y_data_valid))
print(round(time.time()-t, 2), 'seconds to evaluate accuracy...')
print('Test accuracy of classifier = ', round(score, 4))
