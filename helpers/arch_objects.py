import tensorflow as tf
import numpy as np
import time

from helpers.loss_functions import *
from helpers.viz_funcs import * 

import IPython.display as display


class arch_obj:
    def __init__ (self, optimizer = tf.optimizers.Adam, learning_rate = 0.005,
                 n_epochs=5, n_steps_per_epoch=25, decay = 0):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_steps_per_epoch = n_steps_per_epoch
        self.decay = decay
        self.timer = 0
        if optimizer == tf.optimizers.RMSprop:
            self.optim = self.optimizer(learning_rate= self.learning_rate, rho=0.9, momentum=0.001, epsilon=1e-07)
        else: 
            self.optim = self.optimizer(learning_rate= self.learning_rate, beta_1=0.99, epsilon=1e-1, decay=self.decay)
        
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
                    total_loss, c, s = calc_total_loss(img, content_object,style_object)
                grad = tape.gradient(total_loss, img)
                self.optim.apply_gradients([(grad, img)])
                img.assign(tf.clip_by_value(img, clip_value_min= 0.0, clip_value_max=1.0))

#             print('Style Loss : {:4f} Content Loss: {:4f}'.format(s, c))
#             print()
            # visualize img at each epoch
            display.display(viz_tensor(img))
            print("Epoch: {}".format(i))
        end = time.time()
        self.timer = end-start
        print("Total time: {:.1f}".format(end-start))
        self.final_img = img
        content_object.img = img
        
        if content_object.color_adj:
            self.final_img = content_object.get_color_adj_img()
        
        return(self.final_img)
        
        
        