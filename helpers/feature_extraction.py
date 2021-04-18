import tensorflow as tf


def create_feature_extractor(layers):
    # load a pretrained vgg19 model
    ## include_top is set to false, because we do not need the classification layer
    
    #TODO: Are we including avg pooling??
    ## avg pooling is selected because Gatys et al. said that's what they did
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    # set trainable to false
    vgg.trainable = False

    # extract given layers
    outputs = []
    for layer in layers:
        outputs.append(vgg.get_layer(layer).output)

    return tf.keras.Model([vgg.input], outputs)

# # create style feature extractor
# style_feature_extractor = create_feature_extractor(style_layers)

# # create content feature extractor
# content_feature_extractor = create_feature_extractor(content_layers)