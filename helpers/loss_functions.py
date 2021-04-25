import tensorflow as tf
import numpy as np



def compute_content_loss(content_outputs, content_targets):
    content_loss = 0

    # iterate over content features
    for i in range(len(content_outputs)):
        content_loss += tf.reduce_mean((content_targets[i]-content_outputs[i])**2)
        
    content_loss /= len(content_outputs)

    return content_loss

def compute_content_loss_vector(img, content_object):
    content_loss = 0
    content_targets = content_object.targets
    content_outputs = content_object.feature_extractor(img)
    
    # iterate over content features
    for i in range(len(content_outputs)):
        content_loss += tf.reduce_mean((content_targets[i]-content_outputs[i])**2)*content_object.feature_vector[i]
        
    content_loss /= len(content_outputs)

    return content_loss


def compute_content_loss_l1(content_outputs, content_targets):
    content_loss = 0

    # iterate over content features
    for i in range(len(content_outputs)):
        content_loss += tf.reduce_mean(np.abs(content_targets[i]-content_outputs[i]))
        
    content_loss /= len(content_outputs)

    return content_loss



# todo: I think the normalization is wrong (style loss is gigantic)
def compute_style_loss(style_outputs, style_targets):
    style_loss = 0
    # iterate over style features
    for i in range(len(style_outputs)):
        # todo: might have to do the long version
        # compute gram matrix for each style tensor and compute the mse between them
        G = gram_matrix(style_outputs[i])
        A = gram_matrix(style_targets[i])
        loss_i = tf.reduce_mean((G-A)**2)
        style_loss += loss_i
        # todo normalize? 1/(4NM)
        

    style_loss = style_loss / len(style_outputs)
    return style_loss


def compute_style_loss_vector(img, style_object):
    style_outputs = style_object.feature_extractor(img)
    style_targets = style_object.targets
                                                   
    style_loss = 0
    # iterate over style features
    for i in range(len(style_outputs)):
        # compute gram matrix for each style tensor and compute the mse between them
        G = gram_matrix(style_outputs[i])
        A = gram_matrix(style_targets[i])
        # According to equation 4 in the paper 
        loss_i = tf.reduce_sum((G-A)**2) /(4*G.shape[0]**2 * A.shape[0]**2)
        # implementing the different weights for each layer here 
        style_loss += (loss_i*style_object.feature_vector[i])/len(style_outputs) 

    style_loss = style_loss 
    return style_loss



def gram_matrix(output):
    height = tf.shape(output)[1]
    width = tf.shape(output)[2]
    num_channels = tf.shape(output)[3]
    gram_matrix = tf.transpose(output, [0, 3, 1, 2])
    gram_matrix = tf.reshape(gram_matrix, [num_channels, width * height])
    gram_matrix = tf.matmul(gram_matrix, gram_matrix, transpose_b=True)
    return gram_matrix


def total_loss_vanilla(img, 
                       style_feature_extractor, content_feature_extractor,
                       style_targets, content_targets,
                       alpha, beta):
    """ 
    Params:
    - img: input image
    - style_targets: outputs from style_feature extractor on style image
    - content_targets: outputs from content_feature extractor on content image
    - alpha: content_loss weight
    - beta: style_loss weight
    Returns:
    - style transfer loss
    """
    # preprocess img
    pp_img = tf.keras.applications.vgg19.preprocess_input(img*255)
    # style loss
    style_outputs = style_feature_extractor(pp_img)
    style_loss = compute_style_loss(style_outputs, style_targets)
    # content loss
    content_outputs = content_feature_extractor(pp_img)
    # CAN MAKE THIS L1???
    content_loss = compute_content_loss(content_outputs, content_targets)

    # total loss
    total_loss = alpha*content_loss + beta*style_loss
    return total_loss, content_loss, style_loss


def calc_total_loss(img, content_object, style_object):    
    pp_img = tf.keras.applications.vgg19.preprocess_input(img*255)
    # choosing to use the vectorized style and content weights
    content_loss = compute_content_loss_vector(pp_img, content_object)
    style_loss = compute_style_loss_vector(pp_img, style_object)
    total_loss = content_loss + style_loss*100
    return total_loss, content_loss, style_loss
    


