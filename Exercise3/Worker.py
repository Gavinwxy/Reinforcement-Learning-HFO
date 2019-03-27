import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
from SharedAdam import SharedAdam
import random


def hard_copy(targetValueNetwork, valueNetwork):
	for target_param, param in zip(targetValueNetwork.parameters(), valueNetwork.parameters()):
					target_param.data.copy_(param.data)


def train(rank, args, value_network, target_network, optimizer, device, lock, counter):

	# Seed initialize session
	torch.manual_seed(args.seed+rank)

	port = rank*100+15000
	env_seed = rank+123
	random_seed = rank+100
	

	# Hyperparameters
	#num_episode = args.numEpisode
	epsilon = args.epsilon
	discountFactor = args.discountFactor
	vnet_update = args.updateIntV
	tnet_update = args.updateIntT

	# Env configuration
	hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=env_seed)
	hfoEnv.startEnv()
	hfoEnv.connectToServer()


	hard_copy(target_network, value_network)

	loss_func = nn.MSELoss()
	random.seed(random_seed)
	model_dir = './saved_model'	

	# Counter initialization
	t = 0 # thread step counter (for network updating)
	actions = ['MOVE', 'SHOOT', 'DRIBBLE', 'GO_TO_BALL']
	episode = 0
	total_reward = 0
	saved_cnt = 0
	batch_loss = 0

	# Start training through episodes
	while counter.value <= args.timeStep:

		done = False
		curState = hfoEnv.reset()
		curState = torch.Tensor(curState)
		episode += 1

		# Through time steps
		while not done:
			
			# Correct version of value computing	
			action_value = value_network(curState.to(device))
			act_val_collect = list(action_value.detach().cpu().numpy())
			optAct = [i for i, x in enumerate(act_val_collect) if x==max(act_val_collect)]

			if random.random() < epsilon:
				act = random.randint(0,3)
			else:
				act = random.choice(optAct)


			pred_val = action_value[act]
			pred_val = pred_val.to(device)
			

			# Obtain reward and next state
			nextState, reward, done, _, _ = hfoEnv.step(actions[act])
			nextState = torch.Tensor(nextState)
			reward = torch.Tensor([reward])
			discountFactor = torch.Tensor([discountFactor])

			# Compute target value
			target_val = computeTargets(reward.to(device), nextState.to(device), discountFactor.to(device), done, target_network)	
			
			# Update state
			curState = nextState

			# Compute step loss 
			loss = loss_func(pred_val, target_val)
			batch_loss += loss		
			

			with lock:
				# Update global counter
				counter.value = counter.value + 1

			#  Update thread counter
			t += 1

			# Update target network parameter
			if counter.value % tnet_update == 0:
				hard_copy(target_network, value_network)

			# Update value network parameter
			if t % vnet_update == 0 or done:
				optimizer.zero_grad()
				batch_loss.backward(retain_graph=True)
				optimizer.step()
				batch_loss = 0
			
			if counter.value > 0 and counter.value % 1e6 == 0:
				saved_cnt += 1
				if counter.value == args.timeStep:
					saved_cnt = 'last'
				saveModelNetwork(value_network, os.path.join(model_dir, 'params_' + str(saved_cnt) + '.pth'))

			total_reward += reward

		if episode > 0 and episode % 1000 == 0:
			with open('out.txt', 'w') as f:
				print('Reward Per 1000 Episode: ', total_reward, file=f)
			total_reward = 0

		print(counter.value)

def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
	'''
		Apply greedy policy to get max possible action value
	'''
	if done: # Next state terminates
		target_value = reward
	else:
		act_vals = targetNetwork(nextObservation)
		# Obtain the max possible action value	

		vals = act_vals.detach()
		val = torch.max(vals)

		target_value = reward + discountFactor * val

	return target_value 

def computePrediction(state, action, valueNetwork):
	'''
		Apply epsilon greedy on possible action value
	'''
	act_vals = valueNetwork(state)
	act_val = act_vals[0][action]

	return act_val

# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
	torch.save(model.state_dict(), f=strDirectory)
	




