# UAV-image-multilabeling


## Overview
This is the implementation for the UAV image multilabeling model described in this paper: <a href="https://www.mdpi.com/2076-3417/11/9/3974/xml"> "UAV Image Multi-Labeling with Data-Efficient Transformers"</a> by Laila Bashmal, Yakoub Bazi, Mohamad Mahmoud Al Rahhal, Haikel AlHichri, and Naif Alajlan.


In this paper, we present an approach for the multi-label classification of remote sensing images based on data-efficient transformers. During the training phase, we generated a second view for each image from the training set using data augmentation. Then, both the image and its augmented version were reshaped into a sequence of flattened patches and then fed to the transformer encoder. The latter extracts a compact feature representation from each image with the help of a self-attention mechanism, which can handle the global dependencies between different regions of the high-resolution aerial image. On the top of the encoder, we mounted two classifiers, a token and a distiller classifier. During training, we minimized a global loss consisting of two terms, each corresponding to one of the two classifiers. In the test phase, we considered the average of the two classifiers as the final class labels. Experiments on two datasets acquired over the cities of Trento and Civezzano with a ground resolution of two-centimeter demonstrated the effectiveness of the proposed model.

## Model Architecture
![Model Architecture](model_arch.png)
