from MDP import MDP

class BellmanDPSolver(object):
	def __init__(self, discount):
		self.MDP = MDP()
		self.states = self.MDP.S
		self.values = {}
		self.policy = {}
		self.discount = discount
		self.initVs()

	def initVs(self):
		# Initialize state values and policy
		for state in self.states:
			self.values[state] = 0
			self.policy[state] = self.MDP.A
		
	def BellmanUpdate(self):
		currentValues = self.values
		for state in self.states:
			actions = self.MDP.A
			possibleValue = {}
			for action in actions:
				nextProb = self.MDP.probNextStates(state, action)
				actionValue = 0
				for nextState, prob in nextProb.items():
					reward = self.MDP.getRewards(state, action, nextState)
					actionValue += prob*(reward+self.discount*currentValues[nextState])
				possibleValue[action] = actionValue

			# Find all maximum values and corresponding actions

			maxValue = max(possibleValue.values())
			policyTemp = []
			for key in possibleValue.keys(): 
				if possibleValue[key]==maxValue:
					policyTemp.append(key)
			self.values[state] = maxValue
			self.policy[state] = policyTemp

		return self.values, self.policy


		
if __name__ == '__main__':
	solution = BellmanDPSolver(1)
	for i in range(20000):
		values, policy = solution.BellmanUpdate()
	print("Values : ", values)
	print("Policy : ", policy)

