### Creating the architecture object
# this is the object that actually does the transformation on the images
# and it handles all of the loss computation and gradient computation & application

import tensorflow as tf
import numpy as np
import time

from helpers.loss_functions import *
from helpers.viz_funcs import * 

import IPython.display as display

# Architecture object is primarily based on the optimizer type, learning rate, and number of epochs/training cycles
class arch_obj:
    def __init__ (self, optimizer = tf.optimizers.Adam, learning_rate = 0.005,
                 n_epochs=5, n_steps_per_epoch=25, decay = 0):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_steps_per_epoch = n_steps_per_epoch
        self.decay = decay
        self.timer = 0
        # accommodating a research loop of different optimizer
        if optimizer == tf.optimizers.RMSprop:
            self.optim = self.optimizer(learning_rate= self.learning_rate, rho=0.9, momentum=0.001, epsilon=1e-07)
        else: 
            self.optim = self.optimizer(learning_rate= self.learning_rate, beta_1=0.99, epsilon=1e-1, decay=self.decay)
    
    
    # This is the kickoff of the main meat of the system
    # This takes as argument the content image object and the style image object 
    # these are the objects that supply the baselines for calculating the losses and will give 
    # the functionality for finding the feature representation via their feature extractors 
    def run_network(self, content_object, style_object):
        # make tensor 
        # holy shit... preprocessing this image ahead of time makes the whole thing fall apart in bizarre ways
        img = tf.Variable(content_object.original_img)
        
        # set the feature extractors for both images
        style_object.set_feature_extractor()
        content_object.set_feature_extractor()
        
        # train loop
        start = time.time()
        for i in range(self.n_epochs):
            for j in range(self.n_steps_per_epoch):
                # compute loss
                with tf.GradientTape() as tape:
                    # computing the total loss of the images
                    total_loss, c, s = calc_total_loss(img, content_object,style_object)
                grad = tape.gradient(total_loss, img)
                self.optim.apply_gradients([(grad, img)])
                # clipping the image to only be in range 0 - 1
                img.assign(tf.clip_by_value(img, clip_value_min= 0.0, clip_value_max=1.0))

#             print('Style Loss : {:4f} Content Loss: {:4f}'.format(s, c))
            # visualize img at each epoch
            display.display(viz_tensor(img))
            print("Epoch: {}".format(i))
        # TIMING THIS PROCESS
        end = time.time()
        self.timer = end-start
        print("Total time: {:.1f}".format(end-start))
        
        # saving the final image in multiple places for redundancy/safety
        self.final_img = img
        content_object.img = img
        # can continue running the training with the final image saved in content object
        
        if content_object.color_adj:
            self.final_img = content_object.get_color_adj_img()
        
        return(self.final_img)
        
        
        