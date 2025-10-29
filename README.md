# GRPO_CUDA
This repository contains the official implementation of the paper **"Group Relative Policy Optimization for Robust Blind Interference Alignment with Fluid Antennas"**.This project is built upon the repository QianrenLi/rt_grpo. We have made significant modifications including a completely redesigned simulation environment for our problem and flexibility, CUDA acceleration to migrate the GRPO algorithm to GPU for substantially improved training speed, and comprehensive code optimization for enhanced maintainability and extensibility.
### Key Features and Improvements

Redesigned Simulation Environment
A fully customized and flexible simulation framework tailored for robust blind interference alignment problems with fluid antennas.

CUDA Acceleration
The Group Relative Policy Optimization (GRPO) algorithm has been migrated to GPU, achieving a significant improvement in training speed and computational efficiency.

Comprehensive Code Optimization
The codebase has been extensively refactored to ensure better maintainability, scalability, and clarity, facilitating further research and extension.
## Setup
To set up the environment, you can follow the instructions in the [`rl-baselines3-zoo`](https://github.com/DLR-RM/rl-baselines3-zoo/tree/506bb7aa40e9d90e997580a369f2e9bf64abe594) repository.
After setting up the environment, you can apply backbone setup to setup necessary parameters and config.
```bash
python backbone_setup.py
```
> Note: it is recommended to directly clone the repo with  `git clone --recursive https://github.com/QianrenLi/rt_grpo.git`.

## Training
To run the GRPO algorithm, you need to first use the `backbone_setup.py` script to set up the backbone correctly.
After that, the `train.py` script can be used to train the baseline model (e.g. PPO A2C) as a initial point for GRPO.
With the trained model, you can then run the `GRPO.py` script to train the GRPO model.
Take the initial policy PPO, 2 antenna and power constriant 1 as an example, the command is as follows:
```bash
python train.py --algo ppo --env MISOEnv-custom --tensorboard-log ./logs/ --n-timesteps 3000000
```
and then you can run the GRPO algorithm with the following command:
```bash
python GRPO.py  --env MISOEnv-custom --exp-id 1 --algo ppo --folder ./logs/
```
where `--env` specifies the environment, `--exp-id` specifies the experiment ID, `--algo` specifies the algorithm in the initial point  (e.g. PPO), and `--folder` specifies the folder where the logs will be saved.


```bash
python3 GRPO.py --env MISOEnv-custom --exp-id 1 --algo ppo --folder ./logs/ --eval
```
where `--env` specifies the environment, `--exp-id` specifies the experiment ID, `--algo` specifies the algorithm in the initial point (e.g. PPO), and `--folder` specifies the folder where the logs will be saved. The `--eval` flag indicates that you want to evaluate the model.

## TODO

- [ ] Add more details to the README file.
- [ ] Code refactoring for better usage.
- [ ] Add more examples for different environments.

## Acknowledgements
This code use the [`rl-baselines3-zoo`](https://github.com/DLR-RM/rl-baselines3-zoo/tree/506bb7aa40e9d90e997580a369f2e9bf64abe594) repository as the backbone. 
Thanks to the authors for their great work.
