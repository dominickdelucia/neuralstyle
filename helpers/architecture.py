

base_content_layers= ["block4_conv2"]
base_content_weights = [1]

base_style_layers = [
                "block1_conv1",
                "block2_conv1",
                "block3_conv1",
                "block4_conv1",
                "block5_conv1"
]
base_style_weights = [100,100,100,100,100]

base_style_largefield = [1000,1000,1000,1000,1000]


experimental_content_layers = ["block4_conv1","block5_conv1"] 
experimental_style_layers = ["block1_conv1","block2_conv1","block3_conv1"] 


exp_content_layers = ["block4_conv2", "block5_conv2"] 
exp_content_weights = [1,1]
exp_style_layers = ["block5_conv1","block4_conv1"] 
exp_style_weights = [100,100]

def get_candidate_layers():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    layer_list = [layer.name for layer in vgg.layers]
    conv_layers = []
    for layer in layer_list:
        if 'conv' in layer:
            conv_layers.append(layer)
            
    return conv_layers