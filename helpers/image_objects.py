### This is the Image Object 
# this is how to set up each style and content image as an object 
# that tracks the various elements that need to be tracked


import tensorflow as tf
import numpy as np

from helpers.viz_funcs import *

import IPython.display as display

'''Each image should be initialized with 
1) the image file path
2) the type of this image object
3) the feature layers utilized
4) the vector of layer weights
'''
class image_obj:
    def __init__ (self, file_path, img_type,
                  feature_layers, feature_vector,
                  color_adj):
        self.file_path = file_path
        self.img_type = img_type
        self.color_adj= color_adj
        self.original_img = load_img(file_path)
        self.img = self.original_img
        assert len(feature_layers) == len(feature_vector), "should be same number weights as layers" 
        self.feature_layers = feature_layers
        self.feature_vector = feature_vector
        if self.color_adj:
            self.img = self.img - color_mean
    
    
    def get_color_adj_img(self):
        return self.img + color_mean
    
    
    def set_feature_extractor(self):
        # load a pretrained vgg19 model
        ## include_top is set to false, because we do not need the classification layer
        ## avg pooling is selected because Gatys et al. said that's what they did
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', pooling = 'avg')
        # set trainable to false because we're just using the pretrained system
        vgg.trainable = False

        # extract given layers
        outputs = []
        for layer in self.feature_layers:
            outputs.append(vgg.get_layer(layer).output)
        # setting the feature extractor
        self.feature_extractor = tf.keras.Model([vgg.input], outputs)
        print('feature_extractor is now set...') 
        # establishing the targets on the original image 
        # targets for this process do not move and should be stationary on the og img
        self.targets = self.feature_extractor(self.preprocessed_img(self.original_img))    
    
    
    # simple preprocessing of the image for use in vgg19
    def preprocessed_img(self, img):
        pp_img = tf.keras.applications.vgg19.preprocess_input(img*255)
        return pp_img
    