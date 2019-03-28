#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np

class QLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(QLearningAgent, self).__init__()
		self.learningRate = learningRate
		self.discountFactor = discountFactor
		self.initVals = initVals
		self.qTable = {} # Contains all state-action pairs: {(S):{(S,A):value,...},...}
		self.epsilon = epsilon
		self.episodeRecord = []
		

	def learn(self):
		nextState = self.episodeRecord[-1][0]
		# Check if next sate is in q table
		if not nextState in self.qTable:
			self.qTable[nextState] = self.stateInit(nextState)
		
		currentState = self.episodeRecord[-2][0]
		action = self.episodeRecord[-2][1]
		reward = self.episodeRecord[-2][2]
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

	def stateInit(self, state):
		qInit = {}
		for action in self.possibleActions:
			qInit[(state, action)] = self.initVals
		
		return qInit

	def toStateRepresentation(self, state):
		return state[0]

	def setState(self, state):
		if not state in self.qTable:
			self.qTable[state] = self.stateInit(state)
		self.state = state

	def setExperience(self, state, action, reward, status, nextState):
		self.episodeRecord = [(state, action, reward), nextState]

	def setLearningRate(self, learningRate):
		self.learningRate = learningRate

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon	

	#def reset(self):
	#	raise NotImplementedError
		
	def computeHyperparameters(self, episodeIdx):
		lr = 0.1
		ep_initial = 0.2
		k = 1e-4
	
		ep = ep_initial * np.exp(-k*episodeIdx)
		return lr, ep	

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=5000)

	args=parser.parse_args()

	# Initialize connection with the HFO server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Q-Learning Agent
	agent = QLearningAgent(learningRate = 0.1, discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes

	# Run training using Q-Learning
	numTakenActions = 0 
	for episode in range(numEpisodes):
		status = 0
		observation = hfoEnv.reset()
		
		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			update = agent.learn()
			
			observation = nextObservation
	