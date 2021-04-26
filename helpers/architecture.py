### In this file you can find the various settings for running different style transfers
#
# When researching for different results, it is easiest to change these constructions and have it flow
# Thru to the main ipython notebook for running and getting results
# 
# things labeled "base" are compatible with the original paper's recommendations on style/content layers
# things labeled exp are experimental and we changed these a lot to see their results

base_content_layers= ["block4_conv2"]
base_content_weights = [1e6]
base_content_weightsL = [1e4]
base_content_weightsflip = [1e-1]
base_content_weightsS = [1e3]

paper_content = [1e-3]
paper_content2 = [1e-4]

base_style_layers = [
                "block1_conv1",
                "block2_conv1",
                "block3_conv1",
                "block4_conv1",
                "block5_conv1"
]
base_style_weights = [1e-2,1e-2,1e-2,1e-2,1e-2]

base_style_weightsL = [1e-2,1e-2,1e-1,1e0,1e0]

base_style_weightsS = [1e-1,1e-1,1e-2,1e-3,1e-3]

base_style_weightsflip = [100, 100, 100, 100, 100]

paper_style = [1,1,1,1,1]




exp_content_layers = ["block4_conv2", "block5_conv2"] 
exp_content_weights = [1e-1,1e-1]


exp_low_layers = ["block3_conv2", "block4_conv2"]


exp_style_layers = ["block1_conv1","block2_conv1","block3_conv1","block4_conv1"] 
exp_style_weights = [100,100,100,100]

def get_candidate_layers():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    layer_list = [layer.name for layer in vgg.layers]
    conv_layers = []
    for layer in layer_list:
        if 'conv' in layer:
            conv_layers.append(layer)
            
    return conv_layers