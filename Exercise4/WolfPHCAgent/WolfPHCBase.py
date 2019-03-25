#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import numpy as np

np.random.seed(0)	
class WolfPHCAgent(Agent):
	def __init__(self, learningRate, discountFactor, winDelta=0.01, loseDelta=0.1, initVals=0.0):
		super(WolfPHCAgent, self).__init__()
		self.learningRate = learningRate
		self.discountFactor = discountFactor
		self.initVals = initVals
		self.qTable = {} 
		self.episodeRecord = []
		self.qInit = {}
		self.winDelta = winDelta
		self.loseDelta = loseDelta
		# Initialize action value and policy table in the form {action: (value, policy_prob, avg_policy_prob)}
		for action in self.possibleActions:
				self.qInit[action] = [self.initVals, 1/len(self.possibleActions), 1/len(self.possibleActions)]
		self.qInit['state_count'] = 0
		
		
	def setExperience(self, state, action, reward, status, nextState):
		self.episodeRecord = [(state, action, reward), nextState]

	def learn(self):
		# Check if next state is encountered 
		nextState = self.episodeRecord[-1]
		if nextState not in self.qTable:
			self.qTable[nextState] = deepcopy(self.qInit)
		
		currentState = self.episodeRecord[0][0]
		action = self.episodeRecord[0][1]
		reward = self.episodeRecord[0][2]
		
		currentTable = self.qTable[currentState]
		nextTable = self.qTable[nextState]

		currentValue = currentTable[action][0]
		targetValues = []
		for act in self.possibleActions:
			targetValues.append(nextTable[act][0])
		targetValue = max(targetValues)

		error = self.learningRate*(reward+self.discountFactor*targetValue-currentValue)
		currentTable[action][0] += error
		currentTable['state_count'] += 1

		return error


	def act(self):
		# Act according to policy
		currentTable = self.qTable[self.state]
		actions = []
		probs = []
		for key, value in currentTable.items():
			if key != 'state_count':
				actions.append(key)
				probs.append(value[1])		
		action = np.random.choice(actions, p=probs)

		return action

	def act_base(self):
		action = np.random.choice(self.possibleActions)
		return action

	def calculateAveragePolicyUpdate(self):
		currentTable = self.qTable[self.state]
		state_cnt = currentTable['state_count']
		avg_policy_collect = []

		for action, policy_set in currentTable.items():
			if action != 'state_count':
				policy = policy_set[1]
				avg_policy = policy_set[2]
				error = 1/state_cnt * (policy - avg_policy)
				# Update avg policy
				currentTable[action][2] += error
				avg_policy_collect.append(currentTable[action][2])

		return avg_policy_collect


	def calculatePolicyUpdate(self):
		currentTable = self.qTable[self.state]
		values = []
		actions = []
		exp_policy = 0
		exp_avg_policy = 0
		policy_collect = []

		for act, val in currentTable.items():
			if act != 'state_count':
				actions.append(act)
				values.append(val[0])
				exp_policy += val[0]*val[1]
				exp_avg_policy += val[0]*val[2]

		# Obtain optimal action(s)
		optAct = [actions[i] for i, x in enumerate(values) if x == max(values)]
		# Obtain suboptimal actions 
		subOptAct = [act for act in actions if act not in optAct]

		# Decide which learning rate to use
		if exp_policy >= exp_avg_policy:
			delta = self.winDelta
		else:
			delta = self.loseDelta
		
		# Update probs for suboptimal actions
		p = 0
		for act_sub in subOptAct:
			p = p + min(delta/len(subOptAct), currentTable[act_sub][1])
			currentTable[act_sub][1] -= min(delta/len(subOptAct), currentTable[act_sub][1])
		
		# Update probs for optimal actions
		for act_opt in optAct:
			currentTable[act_opt][1] += p/(len(optAct))


		for action, policy_set in currentTable.items():
			if action != 'state_count':
				policy_collect.append(currentTable[action][1])

		return policy_collect
	
	def toStateRepresentation(self, rawState):
		locs = []
		for loc in rawState[0]:
			locs.append(tuple(loc))
		return tuple(locs)

	def setState(self, state):
		if state not in self.qTable:
			self.qTable[state] = deepcopy(self.qInit)
		self.state = state

	def setLearningRate(self,lr):
		self.learningRate = lr
		
	def setWinDelta(self, winDelta):
		self.winDelta = winDelta
		
	def setLoseDelta(self, loseDelta):
		self.loseDelta = loseDelta
	
	def computeHyperparameters(self, episodeIdx, episodeTotal):
		lr_max = 0.05
		lr_min = 0.001

		lr = lr_min + 1/2*(lr_max-lr_min)*(1+np.cos((episodeIdx/episodeTotal)*np.pi))

		return self.loseDelta, self.winDelta, lr

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	numOpponents = args.numOpponents
	numAgents = args.numAgents
	MARLEnv = DiscreteMARLEnvironment(numOpponents = numOpponents, numAgents = numAgents)

	agents = []
	for i in range(args.numAgents):
		agent = WolfPHCAgent(learningRate = 0.2, discountFactor = 0.99, winDelta=0.01, loseDelta=0.1)
		agents.append(agent)

	numEpisodes = args.numEpisodes
	numTakenActions = 0
	totalRewards = 0
	reward_collect = []
	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
		
		while status[0]=="IN_GAME":
			for agent in agents:
				loseDelta, winDelta, learningRate = agent.computeHyperparameters(episode, numEpisodes)
				agent.setLoseDelta(loseDelta)
				agent.setWinDelta(winDelta)
				agent.setLearningRate(learningRate)
			actions = []
			perAgentObs = []
			agentIdx = 0
			for agent in agents:
				obsCopy = deepcopy(observation[agentIdx])
				perAgentObs.append(obsCopy)
				agent.setState(agent.toStateRepresentation(obsCopy))
				actions.append(agent.act())
				agentIdx += 1
			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1

			agentIdx = 0
			for agent in agents:
				agent.setExperience(agent.toStateRepresentation(perAgentObs[agentIdx]), actions[agentIdx], reward[agentIdx], 
					status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agent.learn()
				agent.calculateAveragePolicyUpdate()
				agent.calculatePolicyUpdate()
				agentIdx += 1
			
			observation = nextObservation
			totalRewards += reward[0]
		
		if episode % 1000 == 0:
			print(totalRewards)
			reward_collect.append(totalRewards)
			totalRewards = 0.0

	print('Final average reward: ', sum(reward_collect)/len(reward_collect))