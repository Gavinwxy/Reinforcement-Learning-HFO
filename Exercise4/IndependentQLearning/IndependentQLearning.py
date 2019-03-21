#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import argparse
import numpy as np
		
class IndependentQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(IndependentQLearningAgent, self).__init__()
		self.learningRate = learningRate
		self.discountFactor = discountFactor
		self.epsilon = epsilon
		self.initVals = initVals
		self.qTable = {}
		self.episodeRecord = []

	def setExperience(self, state, action, reward, status, nextState):
		self.episodeRecord = [(state, action, reward), nextState]
	
	
	def learn(self):
		nextState = self.episodeRecord[-1]
		# Check if next sate is in q table
		if not nextState in self.qTable:
			self.qTable[nextState] = self.stateInit(nextState)
		
		currentState = self.episodeRecord[0][0]
		action = self.episodeRecord[0][1]
		reward = self.episodeRecord[0][2]
		currentValue = self.qTable[currentState][(currentState, action)]

		# Obtain target value according to target policy
		targetValue = max([value for value in self.qTable[nextState].values()])

		error = self.learningRate*(reward+self.discountFactor*targetValue-currentValue)
		self.qTable[currentState][(currentState, action)] += error

		return error

	def act(self):
		# Epsilon greedy
		values = []
		actions = []
		# Select actions according to probs
		allPossibleActions = self.qTable[self.state]
		for action, actionValue in allPossibleActions.items():
			actions.append(action[1])
			values.append(actionValue)
		# Looking for all possible optimal actions
		optAct = [actions[i] for i, x in enumerate(values) if x == max(values)]
		# Setting epsilon greedy algorithm
		probs = [1-self.epsilon, self.epsilon]
		indicator = np.random.choice([0,1], p=probs)
		# Choose optimal action with tie breaking
		if indicator == 0: 
			action = np.random.choice(optAct)
		# Randomly choose one action
		else:
			action = np.random.choice(actions)

		return action

	def act_base(self):
		action = np.random.choice(self.possibleActions)
		return action

	def toStateRepresentation(self, state):
		# Q = [[loc1],[loc2]]
		state_tmp = deepcopy(state)
		agent_locs = []	
		for loc in state_tmp[0]:
			agent_locs.append(tuple(loc))
		return tuple(agent_locs)

	def stateInit(self, state):
		qInit = {}
		for action in self.possibleActions:
			qInit[(state, action)] = self.initVals
		
		return qInit

	def setState(self, state):
		# Q = [[loc1],[loc2]]
		if not state in self.qTable:
			self.qTable[state] = self.stateInit(state)
		self.state = state

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon
		
	def setLearningRate(self, learningRate):
		self.learningRate = learningRate
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		return self.learningRate, self.epsilon

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
	agents = []
	for i in range(args.numAgents):
		agent = IndependentQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 0.1)
		agents.append(agent)

	numEpisodes = args.numEpisodes
	numTakenActions = 0
	totalReward = 0.0
	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
		timeSteps = 0
			
		while status[0]=="IN_GAME":
			for agent in agents:
				learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
				agent.setEpsilon(epsilon)
				agent.setLearningRate(learningRate)
			actions = []
			stateCopies = []
			for agentIdx in range(args.numAgents):
				obsCopy = deepcopy(observation[agentIdx])
				stateCopies.append(obsCopy)
				agents[agentIdx].setState(agent.toStateRepresentation(obsCopy))
				actions.append(agents[agentIdx].act_base())
			numTakenActions += 1
			nextObservation, reward, done, status = MARLEnv.step(actions)

			for agentIdx in range(args.numAgents):
				agents[agentIdx].setExperience(agent.toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], reward[agentIdx], 
					status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()
				
			observation = nextObservation
			totalReward += reward[1]

		if episode % 1000 == 0:
			print(totalReward)
			totalReward = 0.0