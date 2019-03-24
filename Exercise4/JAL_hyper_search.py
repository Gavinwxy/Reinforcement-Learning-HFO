#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import itertools
import argparse
import numpy as np 
from tqdm import tqdm

np.random.seed(0)

class JointQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, numTeammates, initVals=0.0):
		super(JointQLearningAgent, self).__init__()
		self.learningRate = learningRate
		self.discountFactor = discountFactor
		self.epsilon = epsilon
		self.initVals = initVals
		self.qTable = {}
		self.episodeRecord = []
		self.qInit = {}
		# Initialize qTable element
		for action1 in self.possibleActions:
			# Action count for teammate
			self.qInit[action1] = 0
			for action2 in self.possibleActions:
				# Joint action table
				self.qInit[(action1, action2)] = self.initVals	
		self.qInit['state_count'] = 0

	def setExperience(self, state, action, oppoActions, reward, status, nextState):
		self.episodeRecord = [(state, action, oppoActions, reward), nextState]
		
	def learn(self):
		nextState = self.episodeRecord[-1]
		# Check if next sate is in q table
		if not nextState in self.qTable:
			self.qTable[nextState] = deepcopy(self.qInit)

		currentState = self.episodeRecord[0][0]
		action = self.episodeRecord[0][1]
		action_oppo = self.episodeRecord[0][2][0]
		reward = self.episodeRecord[0][3]

		table_curState = self.qTable[currentState]
		table_nextState = self.qTable[nextState]
		currentValue = self.qTable[currentState][(action, action_oppo)]

		# Obtain target value according to target policy
		values = []
		state_cnt = table_nextState['state_count']
		if state_cnt == 0:
			targetValue = self.initVals
		else:
			for act_agent in self.possibleActions:
				agent_act_value = 0
				for act_oppo in self.possibleActions:
					act_value = table_nextState[(act_agent, act_oppo)]
					ratio = table_nextState[act_oppo] / state_cnt 
					agent_act_value += act_value*ratio		
				values.append(agent_act_value)

			targetValue = max(values)

		error = self.learningRate*(reward+self.discountFactor*targetValue-currentValue)
		table_curState[(action, action_oppo)] += error
		table_curState['state_count'] += 1
		table_curState[action_oppo] += 1

		return error

	def act(self):
		table = self.qTable[self.state]
		state_cnt = table['state_count']
		# Check if this state is a new state
		if state_cnt == 0:
			action = np.random.choice(self.possibleActions)
		else:
			values = []

			for action_agent in self.possibleActions:
				agent_act_value = 0
				for action_oppo in self.possibleActions:
					action_value = table[(action_agent, action_oppo)]
					ratio = table[action_oppo] / state_cnt 
					agent_act_value += action_value*ratio

				# Collect expected value for each action
				values.append(agent_act_value)

			opt_act = [self.possibleActions[i] for i, x in enumerate(values) if x == max(values)]

			# Add epsilon greedy here
			probs = [1-self.epsilon, self.epsilon]
			indicator = np.random.choice([0,1], p=probs)
			# Choose optimal action with tie breaking
			if indicator == 0: 
				action = np.random.choice(opt_act)
			# Randomly choose one action
			else:
				action = np.random.choice(self.possibleActions)

		return action
	
	def act_base(self):
		action = np.random.choice(self.possibleActions)
		return action

	def setEpsilon(self, epsilon) :
		self.epsilon = epsilon
		
	def setLearningRate(self, learningRate) :
		self.learningRate = learningRate

	def setState(self, state):
		# Initialize new state table
		if not state in self.qTable:
			qInit = deepcopy(self.qInit)
			self.qTable[state] = qInit
		self.state = state

	def toStateRepresentation(self, rawState):
		# Q = [[loc1],[loc2]]
		state_tmp = deepcopy(rawState)
		agent_locs = []	
		for loc in state_tmp[0]:
			agent_locs.append(tuple(loc))
		return tuple(agent_locs)
		
	def computeHyperparameters_cosAn(self, args, episodeIdx, episodeTotal):
		lr_max = args[0]
		lr_min = args[1]
		ep_max = args[2]
		ep_min = args[3]

		lr = lr_min + 1/2*(lr_max-lr_min)*(1+np.cos((episodeIdx/episodeTotal)*np.pi))
		ep = ep_min + 1/2*(ep_max-ep_min)*(1+np.cos((episodeIdx/episodeTotal)*np.pi))

		return lr, ep

	def computeHyperparameters_Exp(self, args, episodeIdx):
		lr_initial = args[0]
		k1 = args[1]
		ep_initial = args[2]
		k2 = args[3]
	
		lr = lr_initial * np.exp(-k1*episodeIdx)
		ep = ep_initial * np.exp(-k2*episodeIdx)
		return lr, ep

if __name__ == '__main__':

	# For cosineAnnealling
	lr_max = [1e-1, 5e-2, 1e-2]
	lr_min = [5e-3, 1e-3, 5e-4]
	ep_max = [1e-1, 5e-2, 1e-2]
	ep_min = [5e-3, 1e-3, 5e-4]

    # For exponential 
	lr = [1e-1, 5e-2, 1e-2]
	k1 = [0, 1e-5, 1e-4]
	ep = [5e-2, 1e-1, 2e-1]
	k2 = [0, 1e-5, 1e-4]


	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
	
	best_setting = {}
	best_avg_reward = 0

	itlen = (len(lr_max)*len(lr_min)*len(ep_max)*len(ep_min))
	#itlen = (len(lr)*len(k1)*len(ep)*len(k2))
	t = tqdm(total=itlen)
	for arg in itertools.product(lr_max, lr_min, ep_max, ep_min):
		
		agents = []
		numAgents = args.numAgents
		numEpisodes = args.numEpisodes
		for i in range(numAgents):
			agent = JointQLearningAgent(learningRate = 0.1, discountFactor = 0.99, epsilon = 0.1, numTeammates=args.numAgents-1)
			agents.append(agent)

		numEpisodes = numEpisodes
		numTakenActions = 0
		totalReward = 0
		reward_collect = []

		for episode in range(numEpisodes):	
			status = ["IN_GAME","IN_GAME","IN_GAME"]
			observation = MARLEnv.reset()
				
			while status[0]=="IN_GAME":
				for agent in agents:
					#learningRate, epsilon = agent.computeHyperparameters_Exp(arg, episode)
					learningRate, epsilon = agent.computeHyperparameters_cosAn(arg, episode, numEpisodes)
					agent.setEpsilon(epsilon)
					agent.setLearningRate(learningRate)
				actions = []
				stateCopies = []
				for agentIdx in range(args.numAgents):
					obsCopy = deepcopy(observation[agentIdx])
					stateCopies.append(obsCopy)
					agents[agentIdx].setState(agents[agentIdx].toStateRepresentation(obsCopy))
					actions.append(agents[agentIdx].act())

				nextObservation, reward, done, status = MARLEnv.step(actions)
				numTakenActions += 1

				for agentIdx in range(args.numAgents):
					oppoActions = actions.copy()
					del oppoActions[agentIdx]
					agents[agentIdx].setExperience(agents[agentIdx].toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], oppoActions, 
						reward[agentIdx], status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
					agents[agentIdx].learn()
					
				observation = nextObservation
				totalReward += reward[0]
			
			if episode % 1000 == 0:
				#print(totalReward)
				reward_collect.append(totalReward)
				totalReward = 0.0

		avg_reward = sum(reward_collect)/len(reward_collect)
		if best_avg_reward < avg_reward:
			best_avg_reward = avg_reward
			best_setting['params'] = arg
			best_setting['avg_reward'] = best_avg_reward
		t.update(1)
		t.set_description('Current Best Avg Reward: {:.4f}'.format(best_avg_reward))
	t.close()
	print(best_setting)

	np.save('JAL_best_cosAn.npy', best_setting)
	#np.save('JAL_best_Exp.npy', best_setting)