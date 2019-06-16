This repository contains an implementation of the data generation and model from "Exploration is key for neural
theorem proving". 

The model and much of the code is based on the work from Rocktäschel and Riedel (2017) (Arxiv
https://arxiv.org/abs/1705.11040, Github https://github.com/uclmr/ntp).

The main script to run is ntp/scripts/algo_experiment. The experiments in the paper are
based on the latter. Both scripts should run out of the box. To configure experiment, either edit conf file or edit
the conf dict in algo_experiment. 