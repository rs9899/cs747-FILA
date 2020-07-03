import sys
import argparse
import math
import random
import numpy as np

## Parse the arguments

from math import log

parser = argparse.ArgumentParser()
parser.add_argument("--instance" , required=True, help="path to the instance file")
parser.add_argument("--algorithm" , required=True, help="one of round-robin, epsilon-greedy,\
											 ucb, kl-ucb, and thompson-sampling")
parser.add_argument("--randomSeed" , required=True, help="Non-negative Integer", type=int)
parser.add_argument("--epsilon" , required=True, help="a number in [0, 1]", type=float)
parser.add_argument("--horizon" , required=True, help="non-negative integer", type=int)

args = vars(parser.parse_args())

args['instance'] = open(args['instance'] , 'r')
random.seed(args['randomSeed'])
np.random.seed(args['randomSeed'])

## FOR PRINTING INTERMEDIATE VALUES
printImd = 0
if printImd == 1:
	args['horizon'] = 204800
listOfHorizon = [50, 200, 800, 3200, 12800, 51200, 204800]
accum = []

## Read file
bandits = []
for line in args['instance']:
	bandits.append((float(line)))

# print(bandits)

def randomGen(p):
	while True:
		num = random.random()
		if num < p:
			yield 1
		else:
			yield 0

banditInst = [randomGen(p) for p in bandits]
numBandit = len(bandits)

## Round Robin
# start from 0 and move in cycle
def RR():
	if printImd != 1:
		return sum([next(banditInst[i%numBandit]) for i in range(args['horizon'])])
	else:
		regret = 0
		for x in range(args['horizon']):
			regret += next(banditInst[x%numBandit])
			if x + 1 in listOfHorizon:
				accum.append(regret)
		return regret

## Epsilon Greedy
# when multiple bests, pick at random
# while exploration, pick at random
def EG():
	succes = [0 for i in range(numBandit)]
	trial = [0 for i in range(numBandit)]
	def best():
		p_emperical = np.array([0.0 for i in range(numBandit)])
		for i in range(numBandit):
			if trial[i] == 0:
				p_emperical[i] = 0
			else:
				p_emperical[i] = succes[i] * 1.0 / trial[i]
		maxx = np.argmax(p_emperical)
		return maxx
	reward = 0
	for x in range(args['horizon']):
		num = random.random()
		if num < args['epsilon']:
			bndt = random.randint(0,numBandit - 1)
			trial[bndt] += 1
			cal = next(banditInst[bndt])
			succes[bndt] += cal
			reward += cal
		else:
			bndt = best()
			trial[bndt] += 1
			cal = next(banditInst[bndt])
			succes[bndt] += cal
			reward += cal
		if printImd == 1 and x + 1 in listOfHorizon :
			accum.append(reward)
	return reward

## UCB
def UCB():
	# take each action once
	oneCall = np.array([next(i) for i in banditInst])
	if args['horizon'] <= numBandit:
		return sum(oneCall[:args['horizon']])
	succes = oneCall
	trial = np.array([1 for i in range(numBandit)])
	pulls = numBandit
	def ucb_help(pls):
		p_ucb = (succes * 1.0 / trial) + (math.log(pls) * 2.0 / trial)**(0.5)
		maxx = np.argmax(p_ucb)
		return maxx
	reward = sum(succes)
	for x in range(args['horizon'] - numBandit):
		bndt = ucb_help(pulls)
		trial[bndt] += 1
		cal = next(banditInst[bndt])
		succes[bndt] += cal
		reward += cal
		pulls += 1
		if printImd == 1 and x + 1 + numBandit in listOfHorizon :
			accum.append(reward)
	return reward	


## KL-UCB
def KU():
	# take each action once
	oneCall = np.array([next(i) for i in banditInst])
	if args['horizon'] <= numBandit:
		return sum(oneCall[:args['horizon']])
	succes = oneCall
	trial = np.array([1 for i in range(numBandit)])
	pulls = numBandit
	def ucb_help(pls):
		## success , trial --  best
		p_emp = (succes * 1.0 / trial)
		def klF(p , q , U):
			# print( p , q)
			if p == 0:
				l = - log(1-q)
			else :
				l = p * (log(p)- log(q)) + (1-p) * (log(1-p) -log(1-q))

			return l - U
		def BinSearch(a , b , p , U):
			r = (a +b) / 2
			z = klF(p , r , U)
			if abs(z) <= 1e-4 or abs(a - b) <= 1e-2:
				return r
			elif z > 0:
				# print("1" , a , r)
				return BinSearch(a , r , p , U)
			else :
				# print("2" , r , b)
				return BinSearch(r , b , p , U)

		U = (log(pls) + 3*log(log(pls)))/trial
		for i in range(numBandit):
			if p_emp[i] < 0.9999:
				p_emp[i] = BinSearch(p_emp[i] , 0.9999 , p_emp[i], U[i])
		return np.argmax(p_emp)
	reward = sum(succes)
	for x in range(args['horizon'] - numBandit):
		bndt = ucb_help(pulls)
		trial[bndt] += 1
		cal = next(banditInst[bndt])
		succes[bndt] += cal
		reward += cal
		pulls += 1
		if printImd == 1 and x + 1 + numBandit in listOfHorizon :
			accum.append(reward)
	return reward	

## Thompson Sampling
def TS():
	succes = [0 for i in range(numBandit)]
	trial = [0 for i in range(numBandit)]
	def best():
		p_emperical = np.array([0.0 for i in range(numBandit)])
		for i in range(numBandit):
			k = np.random.beta(succes[i] + 1 , trial[i] - succes[i] + 1)
			p_emperical[i] = k
		maxx = np.argmax(p_emperical)
		return maxx
	reward = 0
	for x in range(args['horizon']):
		bndt = best()
		trial[bndt] += 1
		cal = next(banditInst[bndt])
		succes[bndt] += cal
		reward += cal
		if printImd == 1 and x + 1 in listOfHorizon :
			accum.append(reward)
	return reward


## FINAL REGRET CALLER

MaxVal = max(bandits) * args['horizon']
if args['algorithm'] == "round-robin":
	reward = RR()
elif args['algorithm'] == 'epsilon-greedy':
	reward = EG()
elif args['algorithm'] == "ucb":
	reward = UCB()
elif args['algorithm'] == "kl-ucb":
	reward = KU()
elif args['algorithm'] == "thompson-sampling":
	reward = TS()
else:
	reward = 0
# print reward , MaxVal

regret = MaxVal - reward

## PRINTING OUTPUT
if printImd == 0:
	print(', '.join([str(x) for x in [args['instance'].name, args['algorithm'], args['randomSeed'],
 			args['epsilon'], args['horizon'], regret]]))
else:
	for z in range(len(listOfHorizon)):
		print(', '.join([str(x) for x in [args['instance'].name, args['algorithm'], args['randomSeed'],
	 			args['epsilon'], listOfHorizon[z], max(bandits)*listOfHorizon[z] - accum[z]]]))

