import os 
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.models import *


# Simple Convolution model on pytorch
class PyModel(nn.Module):
    def __init__(self):
        super(PyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (3,3), 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, (3,3), 1,1),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out
    
# Simple Keras model
def KModel(inp_shape=[200, 200, 3]):
    x = kl.Input(inp_shape)
    net = kl.ZeroPadding2D((1,1))(x)
    net = kl.Conv2D(64, 3, strides=(1,1), padding='valid', name='conv1')(net)
    net = kl.ZeroPadding2D((1,1))(net)
    net = kl.Conv2D(64, 3, strides=(1,1), padding='valid', name='conv2')(net)
    net = kl.LeakyReLU(0.2)(net)
    model = Model(inputs=x, outputs=net)
    return model
    
def transfer_weights(pymodel, kmodel):
    # Create a weights dictionary
    weights = {}
    
    # Loop over the pytorch model and store in dict
    for name, module in pymodel._modules.items():
        if isinstance(module, nn.Conv2d):
            # Take weight and bias of convolutional model
            # For the purpose of testing, we are setting the weight to be constant 1 and 0
            w = module.weight.data.numpy()
            b = module.bias.data.numpy()
            # Store weight in bias in the dict
            weights[name] = [w,b]
        # Handle Sequential model
        if isinstance(module, nn.Sequential):
            for m in module:
                if isinstance(m, nn.Conv2d):
                    w = m.weight.data.numpy()
                    b = m.bias.data.numpy()
                    weights[name] = [w,b]

    # Loop over the keras model
    for lyr in kmodel.layers:
        # Check if it contains trainable weights
        if len(lyr.get_weights()) > 0:
            if lyr.name in weights.keys():
                # Set weights from dict and transpose the 'w' from NCHW - NHWC
                [w,b] = weights[lyr.name]
                lyr.set_weights([w.transpose(2,3,1,0), b])
            else:
                print("Missing", lyr.name, "key in dictionary")
            
def read_img(x):
    im = cv2.imread(x)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

def img_to_tensor(x):
    tsr = torch.from_numpy(x.transpose(2,0,1))
    tsr = tsr.unsqueeze(0)
    tsr = tsr.type(torch.FloatTensor)
    return tsr
    

keras_model_path = os.path.join('../models/kmodel.h5')
pytorch_model_path = os.path.join('../models/pymodel.pt')
tflite_model_path = os.path.join('../models/tfmodel.tflite')

# Init pytorch and keras model
pymodel = PyModel()
kmodel = KModel()

# Load available pytorch model, if exist
# pymodel.load_state_dict(torch.load(pytorch_model_path))

# Transfer weights to keras model
transfer_weights(pymodel, kmodel)
print('Finished transferring model')

# Save the model to h5
tf.keras.models.save_model(kmodel, keras_model_path)

# Convert the keras model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_model_path)
tfmodel = converter.convert()
open(tflite_model_path, "wb").write(tfmodel)
# Done ... 

# Test loading the saved keras model
kmodel_ = tf.keras.models.load_model(keras_model_path)

# Test the model
img_path = '../img/test_img.jpg'
inp_img = read_img(img_path)
inp_tsr = img_to_tensor(inp_img)
inp_img = np.expand_dims(inp_img, 0)

out_pytorch = pymodel(inp_tsr)
out_pytorch = out_pytorch.detach().numpy().transpose(0,2,3,1)
out_keras = kmodel_.predict(inp_img, batch_size=1)

print(out_pytorch.shape, out_keras.shape)
print(np.mean(np.abs(out_keras - out_pytorch)))
