# Autonomous Indoor Navigation

This repository contains the source code for the paper [Autonomous Indoor Robot Navigation via Siamese Deep Convolutional Neural Network](https://dl.acm.org/doi/abs/10.1145/3268866.3268886).

Addresses indoor navigation for autonomous systems (eg. AGV) via ConvNets. Experimentation is performed with 3 model flavours (VGG16, VGG19, and ResNet50).For each flavour, the model is first initialized with IMAGENET pretraining weights.

## Usage
```bash
# extracts the bottleneck features using the pretrained model as a fixed feature extractor.
python bottle_neck_feat.py
# trains the top model (`fc` and `softmax` layers) using the extracted bottleneck features saved on disk.
python train_topmodel.py
# unfreezes the last convolutional layers of the pretrained model and fine-tunes the weights together with the top model trained in the previous step.
python finetune.py
# performs inference using the fine-tuned model on novel images.
python infer.py
```