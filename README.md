# PytorchToTFLite
This repository contains tutorial for converting model from Pytorch to TFLite. 
The model converted via weights transfer from Pytorch to Keras, then converted to TFLite.
This is not automatic conversion tools, The model must be rewrite from scratch in the Keras model format.

There are others method to convert the Pytorch model to Keras/TF model directly, but I haven't got it worked for TFLite conversion.
The main issue is the NCHW - NHWC format between these tools. 
Also they use `Keras` instead of `tf.Keras` which causes failure during the TFLite conversion.

The function used in this tutorial is to provide the general concept how to transfer the weights between these two models.
After reading the tutorial you should get general idea how it works and can be extended accordingly to different variations.

---
The tutorial section consists of 2 parts
* [Pytorch to TFLite (Keras) conversion](./src/pytorch_to_tflite.ipynb)
  * This tutorial describes how to transfer weights from Pytorch - Keras - TFLite
* [Using TFLite in Android](./src/Android_TFLite_tutorial.md)
  * This is a simple tutorial with more details on how to use the TFLite in android, as the official doc is lacking detail.
---

For this tutorial a simple convolutional model is used. 
For my project I used it for Image synthesis.  
However, It should be straight forward to implement it for other tasks like classification, recognition, and etc.

In the future, I will try to implement automatic conversion and provide my learning based image synthesis project on android.
