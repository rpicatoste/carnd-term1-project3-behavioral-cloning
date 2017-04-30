
# TODO: Generate data recording driving.
epochs = 6


#%% Get data

# Transfer data to or from ec2 instance
# scp ~/Desktop/data.zip carnd@35.160.30.216:.
# From the ec2 instance to my computer
# scp carnd@35.160.30.216:model.h5 .
import csv
import os.path
from sklearn.utils import shuffle     
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt   
import cv2
import numpy as np
import sklearn
import random
import numpy as np

samples = []
#datasets = ['recorded_sample','recorded_1','recorded_2','recorded_3','recorded_4','recorded_corrections']
#datasets = ['recorded_sample','recorded_1','recorded_2','recorded_corrections']
#datasets = ['recorded_1','recorded_2','recorded_corrections']
#datasets = ['recorded_1']
datasets = ['recorded_sample']

for dataset in datasets:
    with open('./data/'+ dataset + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if line[3] == 'steering':
                print(line)
                continue
            samples.append(line)
print('List generated')

train_samples, validation_samples = train_test_split(samples, test_size=0.3)
 

#def generator(samples, batch_size = 32):
#    num_samples = len(samples)
#    while 1: # Loop forever so the generator never terminates
#        shuffle(samples)
#        for offset in range(0, num_samples, batch_size):
#            batch_samples = samples[offset:offset+batch_size]
#
#            images = []
#            angles = []
#            for batch_sample in batch_samples:
#                name = './data/IMG/'+batch_sample[0].split('/')[-1]
#                if not os.path.isfile(name):
#                    print('\nFILE DOESN''T EXIST!!', name)
#                center_image = cv2.imread(name)
#                if batch_sample[3] == 'steering':
#                    print(batch_sample)
#                    continue
#                center_angle = float(batch_sample[3])
#                images.append(center_image)
#                angles.append(center_angle)
#
#            # trim image to only see section with road
#            X_train = np.array(images)
#            y_train = np.array(angles)
#            yield shuffle(X_train, y_train)
            
print('Loading data')
images = []
measurements = []
zero_counter = 0



for line in samples:
        
    image_path = line[0]    
    image_left_path = line[1]
    image_right_path = line[2]
    measurement = float(line[3])
    
    dataset_ud = 'recorded_sample'
    if image_path[0:3] == 'IMG':
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
    
    
X_train = np.array(images)
y_train = np.array(measurements)

print('Data loaded')
    
#print(filename)
plt.figure()
plt.imshow(image)
plt.show()
plt.figure()
plt.hist(y_train, 200, alpha=0.75)
plt.show()
    




#%% Create the network
# Initial Setup for Keras
from keras.models import Sequential, load_model, Model
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D, Input, PReLU
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras import optimizers
from keras import regularizers
from keras.layers.normalization import BatchNormalization


model = Sequential()
model.add( Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)) )
# normalization
model.add(Lambda(lambda x: x / 127.5 - 1.0))

# convolutional and maxpooling layers
model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), activation='relu'))
model.add(BatchNormalization())

model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), activation='relu'))
model.add(BatchNormalization())

model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), activation='relu'))
model.add(BatchNormalization())

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1), activation='relu'))
model.add(BatchNormalization())

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1), activation='relu'))
model.add(BatchNormalization())

model.add(Flatten())

# fully connected layers 1164
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(50))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(BatchNormalization())

#model.add(Dense(1, activation='tanh')) # Tanh limits the output between -1 and 1
model.add(Dense(1)) 


#inp = Input(shape = (160,320,3))
#
#x = Cropping2D( cropping=( (50,20), (0,0) ))(inp)
#x = Conv2D(24, 5, 5, border_mode="valid", subsample=(2, 2), activation='relu')(x)
#x = Conv2D(36, 5, 5, border_mode="valid", subsample=(2, 2), activation='relu')(x)
#x = Conv2D(48, 5, 5, border_mode="valid", subsample=(2, 2), activation='relu')(x)
#x = Conv2D(64, 3, 3, border_mode="valid", subsample=(1, 1), activation='relu')(x)
#x = Conv2D(64, 3, 3, border_mode="valid", subsample=(1, 1), activation='relu')(x)
#x = Flatten()(x)
### using dropout tends to make all-zero predictions! 
#x = Dense(100, activation='relu')(x)
##x = Dropout(0.8)(x)
#x = Dense(100, activation='relu')(x)
##x = Dropout(0.8)(x)
#x = Dense(50, activation='relu')(x)
##x = Dropout(0.8)(x)
#x = Dense(10, activation='relu')(x)
##x = Dropout(0.8)(x)
#x = Dense(1, activation="linear")(x)
#
#model = Model(input=inp, output=x)

#model = Sequential()
#model.add( Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)) )
##model.add(AveragePooling2D(pool_size=(2, 2), strides=2))
#model.add( Lambda( lambda x: x / 255.0 - 0.5 )) # Regularization
##model.add( Lambda( lambda x: x / 255.0 - 0.5, input_shape = (160,320,3) ) ) # Regularization
#
#model.add( Convolution2D(24, 5, 5, subsample = (2,2), activation = 'relu'))
#model.add( Convolution2D(36, 5, 5, subsample = (2,2), activation = 'relu'))
#model.add( Convolution2D(48, 5, 5, subsample = (2,2), activation = 'relu'))
##
#model.add( Convolution2D(64, 3, 3, activation = 'relu'))
##model.add( Dropout(0.5) )
#model.add( Convolution2D(64, 3, 3, activation = 'relu') )
##model.add( Dropout(0.5) )
##model.add( Conv2D(64, (3, 3), padding='same', activation = 'relu') )
##
#model.add( Flatten() )
##
#model.add( Dense(500, activation = 'relu') )
##model.add( Activation('relu') )
##model.add( Dropout(0.5) )
##
##model.add( Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(0.01)) )
#model.add( Dense(100, activation = 'relu') )
##model.add( Dropout(0.5) )
##
##model.add( Dense(50, kernel_regularizer = regularizers.l2(0.01)) )
#model.add( Dense(50) )
##model.add( Dropout(0.5) )
##
##model.add( Dense(10, kernel_regularizer = regularizers.l2(0.01)) )
#model.add( Dense(10) )
##model.add( Dropout(0.5) )
## Output
##model.add( Dense(1, kernel_regularizer = regularizers.l2(0.01)) )
#model.add( Dense(1) )

model.summary()

model.compile( loss = 'mse', optimizer = 'adam' )


#import sys;sys.exit("Cricho exit")

#%%
if 0:
    #%%
#    plt.figure()
#    plt.plot(bins)

    for ii in np.random.randint(0,len(images),10):
        plt.figure()
        plt.imshow(X_train[ii])
        plt.title('Image:'+str(ii)+' - steering: '+str(y_train[ii]))
        plt.show()



#%% Train
try:
    model  = load_model( 'model.h5' )
    print('Previous model loaded')
    
except:
    print('No existing model, training a new one')

learn_rate = 0.001
learn_rate = 0.0001
learn_rate = 0.01
#model.compile( loss = 'mse', optimizer = optimizers.Adam(lr = learn_rate, decay = 0.1*learn_rate) )
model.compile( loss = 'mse', optimizer = 'adam' )
history_object = model.fit(  X_train, y_train, validation_split = 0.3,
                             shuffle = True, nb_epoch = epochs, batch_size= 16*2**2, verbose = 1 )

#metrics = model.evaluate(X_normalized_test, y_one_hot_test, batch_size=32, verbose=1)
#model.compile('adam', 'categorical_crossentropy', ['accuracy'])
#history = model.fit(X_normalized, y_one_hot, nb_epoch=10, validation_split=0.2)

# compile and train the model using the generator function
#train_generator = generator(train_samples, batch_size=32)
#validation_generator = generator(validation_samples, batch_size=32)
#
#history_object = model.fit_generator(train_generator, 
#                                     samples_per_epoch = len(train_samples), 
#                                     validation_data = validation_generator, 
#                                     nb_val_samples = len(validation_samples), 
#                                     nb_epoch=3)


model.save( 'model.h5' )
print('Model saved')

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


# Various notes
# TODO: Use dropout
# Augment data: for example the mirrored images.
# Validate: train 80%-70%, validation 20%-30%. Shuffle before sppliting


# Test with the simulator
# python drive.py model.h5
# Then launch the simulator in autonomous mode.
#in drive.py:
#set_speed = 9 # change this to value you like and which works :) 
#controller.set_desired(set_speed)

# FROM THE FORUM
#1. The single layer "check out your work flow" model in the video, is, as advertised, utter rubbish.
#2. The model is seriously sensitive to over training. Like one too many epochs will send it from useless (but humorous) with both positive and negative steering angles into a mode where it produces stuck outputs. In my case, if I trained the model on sample data as long as it is trained in the video, it gets stuck.