import tensorflow as tf

#Functions of CNN:

# 1. Convolution: apply filters to generate feature maps
tf.keras.layers.Conv2D

# 2. Non-linearity: often ReLU to deal with non-linear data
tf.keras.activations.something  #choose activation

# 3. Pooling: downsampling operation on each feature map to scale down size
tf.keras.layers.MaxPool2D


#Those functions ^^^^ in action:


# 1. CNNs: Spatial Arrangement of Output Volume
d, h, w, s = "depth", "height", "width", "filter step size"
tf.keras.layers.Conv2D(filters = d, kernel_size = (h, w), strides = s)

# 2. Introducing Non-Linearity
#applpy agter every conv. layer
tf.keras.layers.ReLU


# 3. Pooling
tf.keras.layers.MaxPool2D(
    pool_size = (2,2),
    strides = 2
)





#PUTTING IT ALL TOGETHER

def generate_model():
    model = tf.keras.Sequential([
        #first convolutional layer
        tf.keras.layers.Conv2D(32, filter_size=3, activation='relu'), #32 feature maps, filter size: 3x3
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        #feed these 32 feature maps into next set of conv/pooling layers:

        #second convolutional layer
        tf.keras.layers.Conv2D(64, filter_size=3, activation='relu'),  #increasing number of features being detected
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),   #but still downscaling image

        #fully connected classifier
        tf.keras.layers.Flatten(), #flatten down info into a single vector to be fed into dense layer
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax') #softmax to make sure outputs are categorical probability distribution
    ])
    return model




#Semantic Segmentation: Fully Convolutional Networks

tf.keras.layers.Conv2DTranspose