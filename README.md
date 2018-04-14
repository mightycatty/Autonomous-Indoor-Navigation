# Autonomous Indoor Navigation
Addresses indoor navigation for autonomous systems (eg. AGV) via ConvNets

Experimentation is performed with 3 model flavours (VGG16, VGG19, and ResNet50)

```
bottle_neck_feat.py
``` extracts the bottleneck features using the pretrained model as a fixed feature extractor
train_topmodel.py trains the top model using the extracted bottleneck features saved on disk
