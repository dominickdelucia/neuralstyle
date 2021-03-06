### These are extremely simple visualization helpers
# Chose to house these functions in a separate file because they were used often
# and used all over the place when building this process
#
# 




import tensorflow as tf
import numpy as np



def viz_tensor(x):
    return tf.keras.preprocessing.image.array_to_img(x.numpy()[0])



def load_img(path_to_img):
    # capping the max image size at 512 for speed and efficacy 
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img