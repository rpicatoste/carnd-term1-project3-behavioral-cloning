

#%%
if 0:
    #%% What the network can see.
    X_to_plot = X_train[0:1,:,:,:]
    
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
    axarr[1].imshow(cropped_im[ii])
    axarr[1].set_title('Cropped')
    axarr[2].imshow(cropped_im_norm[ii,:,:,0], cmap = 'gray')
    axarr[2].set_title('Normalized (one layer, gray)')
    
    
    
    
    #%%
if 0:
    #%%       
    ii = 0
    
    plt.figure()
    plt.imshow(X_train[ii])
    plt.show()
    
    plt.figure()
    plt.hist(X_train[ii,:,:,2].ravel(), 100, alpha=0.75)
    plt.show()
    
    
    