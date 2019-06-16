This repository contains an implementation of the data generation and model from "Exploration is key for neural theorem proving". 

The base model and much of the code is based on the work from Rocktäschel and Riedel (2017) (Arxiv
https://arxiv.org/abs/1705.11040, Github https://github.com/uclmr/ntp).

The main experimental script to run is ntp/scripts/algo_experiment. Run from ntp-release/ntp folder. Sample command: python ntp/scripts/algo_experiment -conf_path conf_synth/algexp.conf

To configure experiment, edit or create new conf file. Test conf file (should take <20s to run) is conf_synth/algtest.conf. 
