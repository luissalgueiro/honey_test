# Honey Test

The honey test dataset is composed of 12 classes of different microspcope images. The images are related to the molecular structure of the different species of Honey.


### Training

For training the models, use the worker scripts:

> sbatch worker_train.sh

### Hyper-parameter Sweep using Wandb

Modify the config_honey_sweep.yml file for declaring the hyperparameters to be swept.

Then launch the agent: 

> wandb sweep config_honey_sweep.yml

Copy the agent configuration and modify the worker_sweep.sh file, then:

> sbatch worker_sweep.sh


### Test

Use the test notebook for testing the models, there are several metrics that shows the performance of the test set.

>> Test.ipynb