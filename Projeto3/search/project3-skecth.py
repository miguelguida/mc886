from game import *
from util import *
from pacman import *

import util, layout
import sys, types, time, random, os

args = readCommand(['--layout', 'smallClassic', '--pacman', 'MyGoWestAgent']) # Get game components based on input
runGames( **args )