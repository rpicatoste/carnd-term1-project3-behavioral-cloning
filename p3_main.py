
# TODO: Generate data recording driving.



#%% Get data

# Transfer data to or from ec2 instance
# scp ~/Desktop/data.zip carnd@35.160.30.216:.
# From the ec2 instance to my computer
# scp carnd@35.160.30.216:model.h5 .
import csv
import cv2 
import numpy as np


lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    if line[3] == 'steering':
        print(line)
        continue
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    

X_train = np.array(images)
y_train = np.array(measurements)

#%% Train
import matplotlib.pyplot as plt
# Initial Setup for Keras
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add( Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)) )

model.add( Lambda( lambda x: x / 255.0 - 0.5 )) # Regularization

model.add( Convolution2D(16, 5, 5) )
model.add( MaxPooling2D((3, 3)))
model.add( Activation('relu'))
#
model.add( Convolution2D(32, 3, 3) )
model.add( MaxPooling2D((3, 3)))
model.add( Activation('relu'))
#
model.add( Flatten() )
model.add( Dense(128) )
model.add( Activation('relu') )
model.add( Dropout(0.5) )
#
model.add( Dense(64) )
model.add( Dropout(0.5) )

#model.add( Dense(128) )
# Output
model.add( Dense(1) )

model.summary()

model.compile( loss = 'mse', optimizer = 'adam' )

try:
    model  = load_model( 'model.h5' )
    print('Previous model loaded')
    
except:
    print('No existing mode, training a new one')

history_object = model.fit(  X_train, y_train, validation_split = 0.2,
                             shuffle = True, nb_epoch = 3, batch_size= 16*2**0, verbose = 1 )

#metrics = model.evaluate(X_normalized_test, y_one_hot_test, batch_size=32, verbose=1)
#model.compile('adam', 'categorical_crossentropy', ['accuracy'])
#history = model.fit(X_normalized, y_one_hot, nb_epoch=10, validation_split=0.2)

model.save( 'model.h5' )
print('Model saved')


#history_object = model.fit_generator(   train_generator, 
#                                        samples_per_epoch = len(train_samples), 
#                                        validation_data = validation_generator,
#                                        nb_val_samples = len(validation_samples), 
#                                        nb_epoch = 5, verbose = 1 )

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# TODO: Use dropout
# Augment data: for example the mirrored images.


# Validate: train 80%-70%, validation 20%-30%. Shuffle before sppliting


# Test with the simulator
# python drive.py model.h5
# Then launch the simulator in autonomous mode.
#in drive.py:
#set_speed = 9 # change this to value you like and which works :) 
#controller.set_desired(set_speed)