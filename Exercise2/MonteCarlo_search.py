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
		self.discountFactor = discountFactor

	def learn(self):

		updateRecord = []
		all_states = list((val[0], val[1]) for val in self.experience)
		for state in self.stateRecord.keys():
			G = 0

			idx = [i for i, x in enumerate(all_states) if x == state]
			idx = min(idx)
			power = 0

			for reward in self.experience[idx:]:
				#G = self.discountFactor*G + reward[2]
				G += np.power(self.discountFactor, power)*reward[2]
				power += 1

			self.qTable[state[0]][state][1] += 1
			qcnt = self.qTable[state[0]][state][1]
			self.qTable[state[0]][state][0] += 1/qcnt * (G - self.qTable[state[0]][state][0])

			updateRecord.append(self.qTable[state[0]][state][0])

		return self.qTable, updateRecord

	def toStateRepresentation(self, state):
		# Only take the state of the attacking agent
		return state[0]

	def setExperience(self, state, action, reward, status, nextState):

		# Record State with reward
		self.experience.append((state, action, reward))
		self.stateRecord[(state, action)] = 0

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
		self.experience = []
		self.stateRecord = collections.OrderedDict()

	def act(self):
		actions = []
		values = []
		# Select actions according to probs
		allPossibleActions = self.qTable[self.state]
		for action, actionInf in allPossibleActions.items():
			actions.append(action[1])
			values.append(actionInf[0])

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


	def act_baseline(self):
		action = np.random.choice(self.possibleActions)
		return action


	def setEpsilon(self, epsilon):
		self.epsilon = epsilon


	def computeHyperparameters(self, numTakenActions, episodeNumber):
		return self.epsilon

	def computeHyperparameters(self, episodeIdx):
		ep = 0.1
		
		return ep

	def computeHyperparameters_exp(self, episodeIdx):
		ep_initial = 0.2
		k = 3e-4
		ep = ep_initial * np.exp(-k*episodeIdx)
		return ep, ep_initial, k			
	
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
	goal_cnt = 0
	# Run training Monte Carlo Method
	for episode in range(numEpisodes):	
		agent.reset()
		observation = hfoEnv.reset()
		status = 0

		while status==0:
			epsilon, ep_int, k = agent.computeHyperparameters_exp(episode)
			agent.setEpsilon(epsilon)
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			observation = nextObservation

			if reward == 1:
				goal_cnt += 1

		agent.learn()


		if episode > 0 and episode % 1000 == 0:
			goal_collect.append(goal_cnt)
			goal_cnt = 0


	print(goal_collect)
	with open('Monte-Carlo-Log.txt', 'a+') as f:
		f.write('Epsilon: {}	Decay rate: {}	result: {} \n'.format(ep_int, k, str(goal_collect)))
