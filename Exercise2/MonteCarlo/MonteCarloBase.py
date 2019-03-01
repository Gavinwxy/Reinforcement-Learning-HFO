#!/usr/bin/env python3
# encoding utf-8

import numpy as np
import collections
import argparse
from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent


class MonteCarloAgent(Agent):
	def __init__(self, discountFactor, epsilon, initVals=0.0):
		super(MonteCarloAgent, self).__init__()
		self.discountFactor = discountFactor
		self.inivals = initVals
		self.qTable = {} # Contains all state-action pairs: {(S):{(S,A):[value, qcnt],...},...}
		self.epsilon = epsilon

	def learn(self):
		updateRecord = []
		# Update state action pair
		for stateActionPairs in self.experience.keys():
			state = stateActionPairs[0]
			self.qTable[state][stateActionPairs][1] += 1

			# Update value
			qCnt = self.qTable[state][stateActionPairs][1]
			qValue = self.qTable[state][stateActionPairs][0]
			self.qTable[state][stateActionPairs][0] += 1/qCnt * (self.episodeReturn - qValue)
			updateRecord.append(self.qTable[state][stateActionPairs][0])

		return updateRecord

	def toStateRepresentation(self, state):
		# Only take the state of the attacking agent
		return state[0]

	def setExperience(self, state, action, reward, status, nextState):
		# Store state action pair appears in episode
		qEpisode = (state, action)
		self.experience[qEpisode] = [reward, status, nextState] # Ordered dict 
		# Accumulate returns
		#self.episodeReturn = self.discountFactor*self.episodeReturn + reward
		self.episodeReturn += np.power(self.discountFactor, self.discountPower)*reward
		self.discountPower += 1

	def stateInit(self, state):
		qInit = {}
		for action in self.possibleActions:
			qInit[(state, action)] = [0, 0]
		
		return qInit


	def setState(self, state):
		if not state in self.qTable:
			self.qTable[state] = self.stateInit(state)

		self.state = state # This state should be (1,1) form
		

	def reset(self):
		self.experience = collections.OrderedDict()
		self.discountPower = 0
		self.episodeReturn = 0

	def act(self):
		actions = []
		values = []
		# Select actions according to probs
		allPossibleActions = self.qTable[self.state]
		for action, actionInf in allPossibleActions.items():
			actions.append(action[1])
			values.append(actionInf[0])

		actionIndex = [i for i, x in enumerate(values) if x == max(values)]

		# All zero probs for actions
		actNum = len(actions)
		optActNum = len(actionIndex)
		probs = np.zeros(actNum)
		
		# Only one optimal 
		if optActNum == 1:
			probs[actionIndex[0]] = 1-self.epsilon + self.epsilon/actNum
			for idx in range(actNum):
				if idx != actionIndex[0]:
					probs[idx] = self.epsilon/actNum

		# Multiple best values
		else:
			selectedIdx = np.random.choice(actionIndex)
			probs[selectedIdx] = 1-self.epsilon + self.epsilon/actNum
			for idx in range(actNum):
				if idx != selectedIdx:
					probs[idx] = self.epsilon/actNum			
			
		action = np.random.choice(actions, p=probs)
		return action

	def act_baseline(self):
		action = np.random.choice(self.possibleActions)
		return action


	def setEpsilon(self, epsilon):
		self.epsilon = epsilon


	def computeHyperparameters(self, numTakenActions, episodeNumber):
		return self.epsilon
	
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()

	#Init Connections to HFO Server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Monte-Carlo Agent
	agent = MonteCarloAgent(discountFactor = 0.99, epsilon = 0.1)
	numEpisodes = args.numEpisodes
	numTakenActions = 0
	# Run training Monte Carlo Method
	for episode in range(numEpisodes):	
		agent.reset()
		observation = hfoEnv.reset()
		status = 0

		while status==0:
			epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			observation = nextObservation

		agent.learn()
