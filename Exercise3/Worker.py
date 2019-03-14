import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
from SharedAdam import SharedAdam
import random

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

	value_network = ValueNetwork(4,[16,16,4],2)
	target_value_network = ValueNetwork(4,[16,16,4],2)
	hard_copy(target_value_network, value_network)
	optimizer = sharedAdam(value_network.parameters(), lr=learning_rate)
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
			# Transverse all possible actions for current state
			action_values = []
			actions = []
			for action in hfoEnv.possibleActions:
				action_value = computePrediction(curState, action, value_network)
				actions.append(action)
				action_values.append(action_value[(curState, action)])

			optAct = [actions[i] for i, x in enumerate(action_values) if x == max(action_values)]

			# Apply epsilon greedy for behaviour policy
			probs = [1-epsilon, epsilon]
			choice = random.choices([0, 1], weights=probs, k=1)
			indices = [i for i in range(len(actions))]

			if choice == 0:
				action = random.choice(optAct)
				act_value = max(action_values)
			else:
				idx = random.choice(indices)
				action = actions[idx]
				act_value = action_values[idx]

			# Obtain reward and next state
			nextState, reward, done, status, info = hfoEnv.step(action)
			total_reward += reward

			# Compute target value
			target_value = computeTargets(reward, nextState, discountFactor, done, target_value_network)		
			
			# Update state
			curState = nextState

			# Compute step loss and com
			loss = 0.5 * (target_value - act_value)**2
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
	# Obtain all possible actions
	possibelActions = HFOEnv.possibleActions
	target_value = {} # in the form of {nextState: target_value}
	if done: # Next state terminates
		target_value[nextObservation] = reward
	else:
		target_values = []
		for action in possibelActions:
			target_values.append(targetNetwork(nextObservation, action))
		# Obtain the max possible action value
		target_value[nextObservation] = reward + discountFactor*max(target_values)
		
	return target_value

def computePrediction(state, action, valueNetwork):
	'''
		Apply epsilon greedy on possible action value
	'''
	action_value = {} # in the form of {(state, action): action_value}
	action_value[(state,action)] = valueNetwork(state, action)
	return action_value

# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
	torch.save(model.state_dict(), strDirectory)
	




