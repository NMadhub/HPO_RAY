# HPO_Ray

To run these examples, you will need to install the following:
```
pip3 install "ray[tune]" torch torchvision pytorch-lightning
```

For training using PytorchLightning on single GPU , run:
```
python3 train_sgpu.py
```


For training on multiple GPUs, run:
```
python3 train_mgpu.py
```

Usage of ```The RayPlugin``` provides Distributed Data Parallel training on a Ray cluster. PyTorch DDP is used as the distributed training protocol, and Ray is used to launch and manage the training worker processes.
Here is a simple example:
```
python3 train_ray.py
```


For knowing the best Hyperparameters using ray-tune, run these:
```
##For single GPU and single CPU usage run the following:
python3 hpo_tune.py

##For multi GPU usage run the following:
python3 hpo_tune_multi.py
```
Note that the important bit in resources_per_trial is per trial. If e.g. you have 4 GPUs and your grid search has 4 combinations, you must set 1 GPU per trial if you want the 4 of them to run in parallel.

If you set it to 4, each trial will require 4 GPUs, i.e. only 1 trial can run at the same time.

This is explained in the ray tune docs, with the following code sample:

If you have 8 GPUs, this will run 8 trials at once.
```tune.run(trainable, num_samples=10, resources_per_trial={"gpu": 1})```

If you have 4 CPUs on your machine and 1 GPU, this will run 1 trial at a time.
```tune.run(trainable, num_samples=10, resources_per_trial={"cpu": 2, "gpu": 1})```


More resources:
1.) Local on Premise Cluster setup using [RayCluster](https://docs.ray.io/en/master/cluster/cloud.html#local-on-premise-cluster-list-of-nodes)

2.) Deploying on [Slurm](https://docs.ray.io/en/latest/cluster/slurm.html) 

Note:SLURM support is still a work in progress. 
SLURM users should be aware of current limitations regarding networking.

3.) To profile your model training loop, wrap the code in the [PyTorch profiler](https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/#:~:text=PyTorch%20Profiler%20is%20the%20next,kernel%20events%20with%20high%20fidelity)

4.) [TensorFlow Profiler](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras): Profile model performance
