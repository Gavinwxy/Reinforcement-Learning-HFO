#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np 

class SARSAAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon=0.1, initVals=0.0):
		super(SARSAAgent, self).__init__()
		self.learningRate = learningRate
		self.discountFactor = discountFactor
		self.inivals = initVals
		self.qTable = {} # Contains all state-action pairs: {(S):{(S,A):value,...},...}
		self.epsilon = epsilon
		self.episodeRecord = []

	def learn(self):
		lastState = self.episodeRecord[-1][0]
		# Check if last sate is terminate state
		if not lastState in self.qTable:
			self.qTable[lastState] = self.stateInit(lastState)

		# The current state
		state1 = self.episodeRecord[-2][0]
		action1 = self.episodeRecord[-2][1]
		reward1 = self.episodeRecord[-2][2]
		value1 = self.qTable[state1][(state1,action1)]

		# The next state
		state2 = lastState
		action2 = self.episodeRecord[-1][1]
		value2 = self.qTable[state2][(state2, action2)]

		# Update q table
		self.qTable[state1][(state1,action1)] += self.learningRate*(reward1 + self.discountFactor*value2 - value1)
		return value2 - value1

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
			qInit[(state, action)] = 0
		
		return qInit


	def setState(self, state):
		if not state in self.qTable:
			self.qTable[state] = self.stateInit(state)

		self.state = state # This state should be (1,1) form

	def setExperience(self, state, action, reward, status, nextState):
		if action == None:
			action = np.random.choice(self.possibleActions)
		self.episodeRecord.append((state, action, reward))


	def computeHyperparameters(self, numTakenActions, episodeNumber):
		return self.learningRate, self.epsilon

	def toStateRepresentation(self, state):
		return state[0]

	def reset(self):
		self.episodeRecord = []

	def setLearningRate(self, learningRate):
		self.learningRate = learningRate

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()
	
	numEpisodes = args.numEpisodes
	# Initialize connection to the HFO environment using HFOAttackingPlayer
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a SARSA Agent
	agent = SARSAAgent(0.1, 0.99)

	# Run training using SARSA
	numTakenActions = 0 
	for episode in range(numEpisodes):	
		agent.reset()
		status = 0

		observation = hfoEnv.reset()
		nextObservation = None
		epsStart = True

		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1

			nextObservation, reward, done, status = hfoEnv.step(action)
			#print(obsCopy, action, reward, nextObservation)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			
			if not epsStart :
				agent.learn()
			else:
				epsStart = False
			
			observation = nextObservation

		agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
		agent.learn()