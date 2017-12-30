import numpy as np
from DeepDataEngineImages import DeepDataEngineImages

np.random.seed() # Randomize

# Set this parameter True to recreate feature storage first
recreateStorage = False

train_data = DeepDataEngineImages('train')
valid_data = DeepDataEngineImages('valid')

if recreateStorage:
    train_plan, test_plan = DeepDataEngineImages.createGenerationPlan('./img_data/vehicles', './img_data/non_vehicles', testSplit = 0.2)

    train_data.createStorage(train_plan, override = True)
    valid_data.createStorage(test_plan, override = True)
else:
    train_data.initStorage()
    valid_data.initStorage()

in_shape, out_shape = train_data.getInOutShape() # Get model input and output size
train_steps, train_gen = train_data.getGenerator() # Load generator for training
valid_steps, valid_gen = valid_data.getGenerator() # Load generator for validation

print('Input shape: {}'.format(in_shape))
print('Train batches: {}'.format(train_steps))
print('Validation batches: {}'.format(valid_steps))

# Import Keras (just topmost namespace)
import keras

continue_learning = False # Set True to continue learning
continue_learning_new_set = False # Set True to train pre-trained model on new data set

if continue_learning:
    # If continue learning, load existing model
    model = keras.models.load_model('./config/clf_cnn.dat')

    if continue_learning_new_set:
        best_val_acc = 0
        best_val_loss = 0
        isFirst = True
    else:
        # If continue learning on same data set, evaluate on validation set to know best loss value already achieved
        best_val_loss, best_val_acc = model.evaluate_generator(valid_gen, valid_steps)
        print("VALIDATION LOSS: {}".format(best_val_loss))
        print("VALIDATION ACCURACY: {}".format(best_val_acc))
        print()
        print()

        isFirst = False
else:
    # Model definition - Based on Keras Sequental model
    model = keras.models.Sequential()
    model.add(keras.layers.Lambda(lambda x: ((x / 255.0) - 0.5) * 2.0, input_shape = in_shape)) # 64 x 64, normalization

    model.add(keras.layers.Conv2D(24, (5, 5), padding='valid')) # 60 x 60, 1st convolutional layer
    #model.add(keras.layers.BatchNormalization(axis=-1, scale=False))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.AvgPool2D(pool_size = (2, 2), strides = (2, 2))) # 30 x 30

    model.add(keras.layers.Conv2D(48, (3, 3), padding='valid')) # 28 x 28, 2rd convolutional layer
    #model.add(keras.layers.BatchNormalization(axis=-1, scale=False))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.AvgPool2D(pool_size = (2, 2), strides = (2, 2))) # 14 x 14

    model.add(keras.layers.Conv2D(64, (3, 3), padding='valid')) # 12 x 12, 3th convolutional layer
    #model.add(keras.layers.BatchNormalization(axis=-1, scale=False))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.AvgPool2D(pool_size = (2, 2), strides = (2, 2))) # 6 x 6

    model.add(keras.layers.Conv2D(128, (3, 3), padding='valid')) # 4 x 4, 4th convolutional layer
    #model.add(keras.layers.BatchNormalization(axis=-1, scale=False))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Flatten()), # 2048, Flattening to 1-dimension array

    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(200)) # 1st fully-connected layer
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dropout(0.35))
    model.add(keras.layers.Dense(150)) # 2nd fully-connected layer
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(50)) # 3rd fully-connected layer
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dropout(0.10))
    model.add(keras.layers.Dense(out_shape[0])) # Output layer
    model.add(keras.layers.Activation('softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['acc']) # Compile model to use Adam optimizer and cross-entropy as loss factor

    best_val_loss = 0
    best_val_acc = 0
    isFirst = True

#for epoch in range(12): # Used to train with fixed number of epochs
epoch = 0
while True: # Infinit loop, must be manually terminated
    print("EPOCH: {}".format(epoch + 1)) # External epoch management is used to be possible save successfull models

    history = model.fit_generator(train_gen, train_steps, validation_data = valid_gen, validation_steps = valid_steps, epochs=1, verbose=1) # Train and validate model on full training set for single epoch

    val_loss = float(history.history['val_loss'][0]) # Get validation losses
    print("VALIDATION LOSS: {}".format(val_loss))
    val_acc = float(history.history['val_acc'][0]) # Get validation accuracy
    print("VALIDATION ACCURACY: {}".format(val_acc))
    
    if isFirst or ((val_acc >= best_val_acc) and (val_loss <= best_val_loss)):
        # Save model if loss factor is decreased and accuracy is increased on validation set.
        best_val_loss = val_loss
        best_val_acc = val_acc
        isFirst = False

        print("MODEL SAVING...")

        model.save('./config/clf_cnn.dat')

        print("MODEL IS SAVED!")
    else:
        print("MODEL SAVING IS SKIPPED.")

    print()
    print()

    epoch += 1