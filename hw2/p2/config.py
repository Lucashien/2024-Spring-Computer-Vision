################################################################
# NOTE:                                                        #
# You can modify these values to train with different settings #
# p.s. this file is only for training                          #
################################################################

# Experiment Settings
exp_name   = 'sgd_pre_da' # name of experiment

# Model Options
model_type = 'resnet18' # 'mynet' or 'resnet18'
# model_type = 'resnet18' # 'mynet' or 'resnet18'

# Learning Options
epochs     = 20           # train how many epochs
batch_size = 128           # batch size for dataloader 
use_adam   = True        # Adam or SGD optimizer
lr         = 0.0008       # learning rate
milestones = [8, 16, 32] # reduce learning rate at 'milestones' epochs
