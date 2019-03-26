import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#from Networks import ValueNetwork
from torch.autograd import Variable
#from Environment import HFOEnv
#from SharedAdam import SharedAdam
import random

random.seed(0)
def hard_copy(targetValueNetwork, valueNetwork):
	for target_param, param in zip(targetValueNetwork.parameters(), valueNetwork.parameters()):
					target_param.data.copy_(param.data)


def train(valueNetwork):

	# Hyperparameters
	num_episode = 8000
	epsilon = 0.1
	discountFactor = 0.99
	learning_rate = 0.0001
	vnet_update = 100
	tnet_update = 1000

	# Env configuration
	port = 000
	seed = 123
	hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
	hfoEnv.startEnv()
	hfoEnv.connectToServer()

	value_network = ValueNetwork(77,[16,16,4],4)
	target_value_network = ValueNetwork(77,[16,16,4],4)
	hard_copy(target_value_network, value_network)
	optimizer = sharedAdam(value_network.parameters(), lr=learning_rate)
	loss_func = nn.MSELoss()
	random.seed()

	# Start training through episodes
	for episode in range(num_episode):

		# Counter initialization
		T = 0 # Target counter
		t = 0 # thread step counter (for network updating)
		done = False
		curState = hfoEnv.reset()
		curState = torch.Tensor(curState)
		total_reward = 0
		batch_loss = 0
		# Through time steps
		while not done:
			
			# Correct version of value computing	
			action_value = value_network(curState)
			act_val_collect = action_value.numpy()
			optAct = [i for i, x in enumerate(act_val_collect) if x == max(act_val_collect)]

			if random.random() < epsilon:
				act = random.randint(0,3)
			else:
				act = random.choice(optAct)

			pred_val = action_value[act]

			# Obtain reward and next state
			nextState, reward, done, _, _ = hfoEnv.step(act)
			total_reward += reward
			nextState = torch.Tensor(nextState)

			# Compute target value
			target_value = computeTargets(reward, nextState, discountFactor, done, target_value_network)
			target_val = target_value.detach()		
			
			# Update state
			curState = nextState

			# Compute step loss and com
			loss = loss_func(pred_val, target_val)
			batch_loss += loss
			T += 1
			t += 1
			
			# Update target network parameter
			if 	T % tnet_update == 0:
				hard_copy(target_value_network, value_network)
			# Update value network parameter
			if 	t % vnet_update == 0:
				optimizer.zero_grad()
				batch_loss.backward()
				optimizer.step()

	model_dir = './saved_model'		
	saveModelNetwork(value_network, model_dir)
	print(episode, total_reward)

def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
	'''
		Apply greedy policy to get max possible action value
	'''
	if done: # Next state terminates
		target_value = torch.Tensor([reward])
	else:
		act_vals = targetNetwork(nextObservation)
		# Obtain the max possible action value	

		vals = act_vals.detach()
		val = torch.max(vals).item()

		target_value = torch.Tensor([reward + discountFactor * val])

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
	torch.save(model.state_dict(), strDirectory)
	




