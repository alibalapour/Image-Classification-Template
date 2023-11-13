# Image Classification Template

The project contains a simple resnet model with different options for data augmentations, optimizers, loss functions, etc. The simple resnet model implemented in this repository was aimed to be trained and tested on cifar10. The model got 96.1% accuracy without any pre-training.


## To Do
The main steps are that to train, evaluate and deploy a simple deep model as an image classification. 
A list of to do has been prepared for the guidance of the procedure.
- [x] Preparing data pipelines
- [x] Preparing model
- [x] Preparing training loop
- [x] Considering the config.yaml file to change the training settings
- [x] Training and reach the best model (96% accuracy) on the valid set
- [x] Deploying the best model by creating and testing an API

## Training
Set the config file as you want and run the following command:
`python training/train.py -c [curl_config_path]`

## Config File
Using curl config.yaml file you can change many settings without going into the codes. In fact, 90 percent of the time you 
only need to change its values and start training. In the following, every option has been explained. 

### General Settings
**pretrained_weights**: Using pretrained weights (True or False). \
**fix_seed**: Setting fix seed for training (e.g., 2 or False). \
**result_path**: Path for saving the results including tensorboard, logs and checkpoints.\
**checkpoints_every**: Saving checkpoints every epoch. \
**tensorboard_log**: Saving log in tensorboard (True or False).

### Training Settings
**train_path**: The path of training images. \
**train_batch_size**: Batch size of training step. \
**num_epochs**: Number of epochs for training. \
**shuffle**: Shuffling image pairs during training.\
**h_input**: Height of images during training (e.g., 1000). \
**w_input**: Width of images during training (e.g., 1000). \
**mixed_precision**: Training using mixed precision for faster training with less gpu memory footprint (True or False). \
**device**: Device name for training step (cuda or cpu) 

### Validation Settings
**valid_path**: The path of validation images. \
**valid_batch_Size**: Batch size of validation step. \
**do_every**: Evaluate every n epochs (e.g., 5) \
**device**: Device name for doing validation step (cuda or cpu) \

### Optimizer
Adam, AdamW and [AdaBelief](https://www.google.com/search?q=adabelief+pytorch&oq=adabelief&aqs=chrome.1.69i57j0i512l5j0i390l3j69i59.3407j0j7&sourceid=chrome&ie=UTF-8)
are the only optimizers we want to have here. \

**name**: name of optimizers (Adam, AdamW, AdaBelief). \
**lr**: Learning rate. \
**sam**: Using SAM on the optimizer (True or False). \
**weight_decouple**: Decoupling weight decay (True or False). \
**weight_decay**: Weight decay (e.g., 0.01). \
**eps**: Epsilon value in optimizers (e.g., 1e-8). \
**grad_clip_norm**: Gradient normalization value (e.g., 5). 
 
### Learning Rate Decay
Here, the learning rate refers to a cosine learning rate decay with warmup in the beginning of training.

**warmup**: Number of steps for warmup. Must be an int number (e.g., 800). \
**min_lr**: Minimum learning rate for learning rate scheduler. \
**gamma**: Gamma vale for cosine weight decay parameter (e.g., 1)

### Augmentation
Implement following augmentations in your code:

**mix_up**: Using mixup transformation (True or False). \
**rand_aug**: Using the randaugment policy with parameters of 2 and 10 (True or False).

## Tensorboard
During training:
1. Step loss
2. Epoch loss
3. Accuracy
4. Learning rate step

Learning rate step and step loss should add to the training tensorboard every 5 steps.

During validation:
1. Accuracy
2. Precision
3. Recall
4. F1
5. Loss


## Tips
* Use train.py to write the training and validation loop.
* Use model.py to create and test model.
* Use dataset to prepare and build dataset.
* Use utils.py to add every function that can not fit in mentioned files.
