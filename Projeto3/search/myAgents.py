from game import Directions
from game import Agent
from game import Actions
import util
import time
import search


n = Directions.NORTH
s = Directions.SOUTH
e = Directions.EAST
w = Directions.WEST
actions = [w,w,w,w,s,w]

class MyGoWestAgent(Agent):
	"An agent that goes West until it can't."
	

	def getAction(self, state):
		"The agent receives a GameState (defined in pacman.py)."
		print(state.getLegalPacmanActions())
		print(state.getPacmanState())

		print(state.getPacmanPosition())
		print(state.getScore())
		action = s
		if len(actions) > 0:
			action = actions.pop(0)
		if action in state.getLegalPacmanActions():
			print(state.generateSuccessor(0,action).getGhostPositions())
			return action
		else:
			return Directions.STOP