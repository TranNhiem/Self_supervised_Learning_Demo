# Self-supervised-Learning Demo

 This repository deploys the pre-trained neural networks from self-supervised pretraining for the variety downstream applications.
 
--------------------------------------------------------------------------------------------------------------
## Section 1 Image Retrieval (Text-to-Image retrieval, Image-to-Image Retrieval) 

1. Text-to-Image Retrieval 

- Transformer model to extract the text-queries embedding representation 

- ViTs model to extract Image embedding representation 

+ Retrieving Top-K similarity between the text-queries & data images-embedding


2. Image-to-Image Retrieval 

- ViTs model extract Image-Queries embedding representation 

- ViTs model extract all other images embedding representation 

+ Retrieving Top-K similarity between the Image-queries & data images-embedding

--------------------------------------------------------------------------------------------------------------
## Section 2 Patch-Level Retrieval 

- ViTs model extract Patch-Queries embedding representation 

- ViTs model extract all other images Patches embedding representation 

+ Retrieving Top-K similarity between the Patch-queries & other Patches-embedding
--------------------------------------------------------------------------------------------------------------
## Section 3 Image Segmentation 

1. CoCo dataset segmentation

- ResNet50 pretraining with Heuristic Attention Representation Learning for Self-Supervised Pretraining
- Fine-tune MaskRCNN with ResNet50 backbone using coco dataset
+ Segment objects in an image

2. Model feature attention part

- ResNet50 pretraining with Heuristic Attention Representation Learning for Self-Supervised Pretraining
+ Get the attention map from ResNet50.