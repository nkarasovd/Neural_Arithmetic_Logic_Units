# Neural Arithmetic Logic Units

A PyTorch implementation of [Neural Arithmetic Logic Units](https://arxiv.org/abs/1808.00508) by *Andrew Trask, Felix Hill, Scott Reed, Jack Rae, Chris Dyer and Phil Blunsom*.

## NAC and NALU architectures 
<p align="center">
 <img src="./src/images/paper/models.png" alt="Drawing", width=75%, height="100%">
</p>

# Experiments

## Experiment 1: Numerical Extrapolation Failures in Neural Networks

<figure align="center">
 <img src="./src/images/paper/experiment_01.png" alt="Drawing", width=75%, height="100%", title="Results from paper"> <img src="./src/images/experiments/extrapolation_failure.png" alt="Drawing", width=75%, height="100%", title="Actual results">
 <figcaption>MLPs learn the identity function only for the range of values they are trained on. The mean error ramps up severely both below and above the range of numbers seen during training.</figcaption>
</figure>
