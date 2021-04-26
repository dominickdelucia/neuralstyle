##### This file is never actually run... just a clone of the process for kicking off many style transfers
# Copied code from the jupyter notebook into here just in case I was messing around with the code too much
# and something broke or got out of whack.. I had a reference to go back to
#
#


from PIL import Image
from helpers.arch_objects import * 
from helpers.image_objects import *


style_stem = "imgs/style"
content_stem = "imgs/content"
output_stem = "imgs/outputs"

style_imgs = os.listdir("imgs/style")
content_imgs = os.listdir("imgs/content")


for style_img in style_imgs:
    for content_img in content_imgs:
        content_path = content_stem + content_img
        style_path = style_stem + style_img
        output_path = output_stem + '_' + style_img + '_' + content_img
        arch = arch = arch_obj(optimizer = tf.optimizers.Adam, learning_rate = 0.005,
                 n_epochs=5, n_steps_per_epoch=25, decay = 1e-5) 
        content_object = image_obj(content_path, img_type = 'content',
                           feature_layers = exp_content_layers,
                           feature_vector = exp_content_weights, color_adj = False)
        style_object = image_obj(style_path, img_type = 'style',
                         feature_layers = exp_style_layers,
                         feature_vector = exp_style_weights, color_adj = False) 
        numpy_image = arch.final_img.numpy()
        PIL_image = Image.fromarray(np.uint8(numpy_image[0]*255)).convert('RGB')
        PIL_image.save(output_path)