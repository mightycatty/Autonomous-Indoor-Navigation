# Autonomous Indoor Navigation
Addresses indoor navigation for autonomous systems (eg. AGV) via ConvNets

Experimentation is performed with 3 model flavours (VGG16, VGG19, and ResNet50)

For each flavour, the model is first initialized with IMAGENET pretraining weights.

`bottle_neck_feat.py` extracts the bottleneck features using the pretrained model as a fixed feature extractor.

`train_topmodel.py` trains the top model (`fc` and `softmax` layers) using the extracted bottleneck features saved on disk.

`finetune.py` unfreezes the last convolutions layers of the pretrained model and fine-tunes the weights together with the top model trained in the previous step.

`infer.py` performs inference using the fine-tuned model on novel images.
