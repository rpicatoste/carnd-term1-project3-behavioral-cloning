
# Parameters of the training
epochs = 3
batch_size = 64
validation_proportion = 0.3

#%% Get data

import csv
import os.path
from sklearn.utils import shuffle     
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt   
import cv2
import numpy as np

samples = []
#datasets = ['recorded_sample','recorded_1','recorded_2','recorded_3','recorded_4','recorded_corrections']
#datasets = ['recorded_sample','recorded_1','recorded_2','recorded_corrections']
#
datasets = ['recorded_sample','recorded_1','recorded_2','recorded_corrections']
#datasets = ['recorded_1']
#datasets = ['recorded_sample']

for dataset in datasets:
    with open('./data/'+ dataset + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if line[3] == 'steering':
                print(line)
                continue
            samples.append(line)
            
print('List generated')

train_samples, validation_samples = train_test_split( shuffle(samples), test_size = validation_proportion)
 
print('Using generator')

def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
    
        shuffle(samples)
        zero_counter = 0
        
        for offset in range(0, num_samples, batch_size):
        
            batch_samples = samples[ offset : offset + batch_size ]

            images = []
            measurements = []
            
            for batch_sample in batch_samples:
  
                image_path = batch_sample[0]    
                image_left_path = batch_sample[1]
                image_right_path = batch_sample[2]
                measurement = float(batch_sample[3])
                
                dataset_ud = 'recorded_sample'
                if image_path[0:3] == 'IMG': # the naming in the udacity data come from linux and this is needed.
                    image_path       = './data/'+ dataset_ud + '/IMG/' + image_path.split('/')[-1]
                    image_left_path  = './data/'+ dataset_ud + '/IMG/' + image_left_path.split('/')[-1]
                    image_right_path = './data/'+ dataset_ud + '/IMG/' + image_right_path.split('/')[-1]
                
                if not os.path.isfile(image_path):
                    import sys;sys.exit('\nFILE DOESN''T EXIST!!\n '+ image_path)
                    
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
                
                # Treatment for low steering images
                if abs(measurement) < 0.01:
                    zero_counter += 1
                    # Remove 1 out of 20 captures where the steering is 0.
                    if zero_counter%20 != 0:
                        pass
                    else:
                        images.append(image)
                        measurements.append(measurement)
                    
                        # Augment data using lateral cameras as "being closer to the side"  
                        # I will do it only for low steering images
                        extra_steering_for_lateral_cameras = 0.2
                        # Left camera (add positive steering)
                        image = cv2.imread(image_left_path)
                        images.append(image)
                        measurements.append(measurement + extra_steering_for_lateral_cameras)
                        # Right camera (add negative steering)
                        image = cv2.imread(image_right_path)
                        images.append(image)
                        measurements.append(measurement - extra_steering_for_lateral_cameras)
                    
                # Treatment for above low steering images
                else:    
            
                    images.append(image)
                    measurements.append(measurement)
                
                    # Augment flipped data
                    images.append(np.fliplr(image))
                    measurements.append(-measurement)
               
            # Since we are augmenting data, we need to drop some samples. Since we are shuffling in the 
            # generation, the samples dropped are random and will appear eventually.
            X_train = np.array(images[:batch_size])
            y_train = np.array(measurements[:batch_size])
                        
            yield X_train, y_train
       

#%% Create the network
# Initial Setup for Keras
from keras.models import Sequential, load_model, Model
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D, Input, PReLU
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import regularizers


model = Sequential()
model.add( Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)) )
# normalization
model.add(Lambda(lambda x: x / 127.5 - 1.0))

# convolutional and maxpooling layers
model.add( Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), activation='relu') )
model.add( BatchNormalization() )

model.add( Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), activation='relu') )
model.add( BatchNormalization() )

model.add( Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), activation='relu') )
model.add( BatchNormalization() )

model.add( Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1), activation='relu') )
model.add( BatchNormalization() )

model.add( Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1), activation='relu') )
model.add( BatchNormalization() )

model.add( Flatten() )

# fully connected layers 1164
model.add( Dense(100, activation='relu') )
model.add( BatchNormalization() )
model.add( Dropout(0.5) )

model.add( Dense(100) )
model.add( BatchNormalization() )
model.add( Dropout(0.5) )

model.add( Dense(50) )
model.add( BatchNormalization() )
model.add( Dropout(0.5) )

model.add( Dense(10) )
model.add( BatchNormalization() )

model.add( Dense(1, activation='tanh') ) # Tanh limits the output between -1 and 1

model.summary()

model.compile( loss = 'mse', optimizer = 'adam' )


#%% Train
try:
    model  = load_model( 'model.h5' )
    print('Previous model loaded')
    
except:
    print('No existing model, training a new one')

model.compile( loss = 'mse', optimizer = 'adam' )
#history = model.fit(X_train, y_train, validation_split = validation_proportion,
#                    shuffle = True, nb_epoch = epochs, batch_size= batch_size, verbose = 1 )


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
#
history = model.fit_generator(train_generator, 
                              steps_per_epoch = len(train_samples)/batch_size, 
                              validation_data = validation_generator, 
                              validation_steps = len(validation_samples)/batch_size, 
                              nb_epoch = epochs)


model.save( 'model.h5' )
print('Model saved')

### print the keys contained in the history object
print(history.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

