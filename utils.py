
# To use the plots in this file, run the cells corresponding to them (after 
# running the model.py). In spyder this is done with ctrl + enter being over
# the cell to run.

#%%
# Function to get n random images from the samples.
def get_n_random_images( n_images ):

    measurements = []
    images_to_plot = []
    
    for ii in np.random.randint(0, num_samples, n_images):
  
        image_path = samples[ii][0]
        measurement = float(samples[ii][3])
        dataset_ud = 'recorded_sample'
        if image_path[0:3] == 'IMG':
            image_path       = './data/'+ dataset_ud + '/IMG/' + image_path.split('/')[-1]
        
        if not os.path.isfile(image_path):
            import sys;sys.exit('\nFILE DOESN''T EXIST!!\n '+ image_path)
 
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
        images_to_plot.append( image )
        measurements.append( measurement )
        
    return images_to_plot, measurements
    
#%%
if 0:
    #%%
    import matplotlib.pyplot as plt   

    n_samples = 5
    images_to_plot, angles = get_n_random_images( n_samples )
    
    for ii in range( n_samples ):
                                          
        plt.figure() 
        plt.imshow(images_to_plot[ii])
        plt.title('Image - steering: '+str(angles[ii]))
        plt.show()
        
        
#%%
if 0:
    #%% What the network can see.
    
    images_to_plot, _ = get_n_random_images( 2 )
    X_to_plot = np.array( images_to_plot )
    
    # Build a model with just the input layers and pass the sample images.
    model_crop = Sequential()
    model_crop.add( Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)) )
    
    model_crop_norm = Sequential()
    model_crop_norm.add( Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)) )
    model_crop_norm.add( Lambda( lambda x: x / 255.0 - 0.5 ))

    cropped_im = model_crop.predict(X_to_plot, batch_size=128)
    cropped_im_norm = model_crop_norm.predict(X_to_plot, batch_size=128)
 
    f, axarr = plt.subplots(3, sharex=True, figsize=(7, 10))
    ii = 0
    axarr[0].imshow(X_to_plot[ii])
    axarr[0].set_title('Original')
    axarr[1].imshow(cropped_im[ii].astype(np.uint8))
    axarr[1].set_title('Cropped')
    axarr[2].imshow(cropped_im_norm[ii,:,:,0], cmap = 'gray')
    axarr[2].set_title('Normalized (one layer, gray)')
    
    
    
    
    #%%
if 0:
    #%% Understanding color content     

    images_to_plot, _ = get_n_random_images( 2 )
    X_to_plot = np.array( images_to_plot )
    
    f, axarr = plt.subplots(2, 2, figsize=(8, 7))
    ii = 0
    axarr[0][0].imshow(X_to_plot[ii])
    axarr[0][0].set_title('Original')
    axarr[0][1].hist(X_to_plot[ii,:,:,0].ravel(), 100, alpha=0.75)
    axarr[0][1].set_title('Histogram layer 1')
    axarr[1][0].hist(X_to_plot[ii,:,:,1].ravel(), 100, alpha=0.75)
    axarr[1][0].set_title('Histogram layer 2')
    axarr[1][1].hist(X_to_plot[ii,:,:,2].ravel(), 100, alpha=0.75)
    axarr[1][1].set_title('Histogram layer 3')
    
    
    