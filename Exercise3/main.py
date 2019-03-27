#!/usr/bin/env python3
# encoding utf-8
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from Networks import ValueNetwork
from SharedAdam import SharedAdam
from Worker import train

# Training settings
parser = argparse.ArgumentParser(description='RL Ex3 Training !')

parser.add_argument('--numProcesses', type=int, default=8)
#parser.add_argument('--numEpisode', type=int, default=8000)
parser.add_argument('--timeStep', type=int, default=1e6)
parser.add_argument('--learningRate', type=float, default=1e-3)
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--discountFactor', type=float, default=0.99)
parser.add_argument('--updateIntV', type=int, default=100)
parser.add_argument('--updateIntT', type=int, default=1000)
parser.add_argument('--seed', type=int, default=1)


# Use this script to handle arguments and 
# initialize important components of your experiment.
# These might include important parameters for your experiment,
# your models, torch's multiprocessing methods, etc.
if __name__ == "__main__" :
	args = parser.parse_args()

	# Setting Zone
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	torch.manual_seed(args.seed)
	mp.set_start_method('spawn')

	value_network = ValueNetwork(15,[16,16,4],4).to(device)
	target_network = ValueNetwork(15,[16,16,4],4).to(device)
	value_network.share_memory()
	target_network.share_memory()
	optimizer = SharedAdam(value_network.parameters(), lr=args.learningRate)

	counter = mp.Value('i', 0)
	lock = mp.Lock()

	processes = []
	for rank in range(0, args.numProcesses):
		trainingArgs = (rank, args, value_network, target_network, optimizer, device, lock, counter)
		p = mp.Process(target=train, args=trainingArgs)
		p.start()
		processes.append(p)
	for p in processes:
		p.join()
