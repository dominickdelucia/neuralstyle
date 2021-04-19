import tensorflow as tf
import numpy as np
import time

from helpers.loss_functions import *
from helpers.feature_extraction import *
from helpers.architecture import * 
from helpers.viz_funcs import *

import IPython.display as display


def run_style_transfer(content_image, style_image,
                       content_layers = base_content_layers,
                       style_layers = base_style_layers,
                       n_epochs=10, n_steps_per_epoch=100, 
                       lr = 0.02, decay = 1e-6,
                       alpha=1, beta=100):
    """ 
    Params:
    - n_epochs: number of epochs
    - n_steps: number of steps
    - alpha: content_loss weight (note alpha/beta ratio should be 1*10^-3 or 1*10^-4 according to Gatys et al.)
    - beta: style_loss weight
    - content_image: the image whose content we wish to match
    - style_image: the image whose style we wish to match
    Returns:
    - None
    """
    # define targets
    pp_content_image = tf.keras.applications.vgg19.preprocess_input(content_image*255)
    pp_style_image = tf.keras.applications.vgg19.preprocess_input(style_image*255)
    
    style_feature_extractor = create_feature_extractor(style_layers)
    content_feature_extractor = create_feature_extractor(content_layers)
    
    style_targets = style_feature_extractor(pp_style_image)
    content_targets = content_feature_extractor(pp_content_image)
    
    # define optimizer
    optim = tf.optimizers.Adam(learning_rate= lr, beta_1=0.99, epsilon=1e-1, decay=decay)

    # initialize input image
    img = tf.Variable(content_image)

    # train loop
    start = time.time()
    for i in range(n_epochs):
        for j in range(n_steps_per_epoch):
            # compute loss
            with tf.GradientTape() as tape:
                total_loss, c, s = total_loss_vanilla(img,
                                                      style_feature_extractor, 
                                                      content_feature_extractor,
                                                      style_targets, content_targets,
                                                      alpha, beta)
            grad = tape.gradient(total_loss, img)
            print(grad)
            optim.apply_gradients([(grad, img)])
            # todo: use something else for this
            img.assign(clip_0_1(img))
            display.display(viz_tensor(img))

#         print(total_loss)
        print('Style Loss : {:4f} Content Loss: {:4f}'.format(s, c))
        print()
        # visualize img at each epoch
        display.display(viz_tensor(img))
        print("Epoch: {}".format(i))
        
    end = time.time()
    print("Total time: {:.1f}".format(end-start))
    
    return(img)
    
def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


