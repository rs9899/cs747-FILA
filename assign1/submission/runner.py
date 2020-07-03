import os
seedLen = 50

for i in range(3):
	## algorithm from round-robin; epsilon-greedy with epsilon set to 0.002, 0.02, 0.2; ucb, 
	for sed in range(seedLen):
		os.system( "./bandit.sh --instance ../instances/i-"+str(i+1)+".txt --algorithm round-robin --randomSeed " + str(sed) + " --epsilon 0 --horizon 204800 ")
	for sed in range(seedLen):
		os.system( "./bandit.sh --instance ../instances/i-"+str(i+1)+".txt --algorithm epsilon-greedy --randomSeed " + str(sed) + " --epsilon 0.002 --horizon 204800 ")
	for sed in range(seedLen):
		os.system( "./bandit.sh --instance ../instances/i-"+str(i+1)+".txt --algorithm epsilon-greedy --randomSeed " + str(sed) + " --epsilon 0.02 --horizon 204800 ")
	for sed in range(seedLen):
		os.system( "./bandit.sh --instance ../instances/i-"+str(i+1)+".txt --algorithm epsilon-greedy --randomSeed " + str(sed) + " --epsilon 0.2 --horizon 204800 ")
	for sed in range(seedLen):
		os.system( "./bandit.sh --instance ../instances/i-"+str(i+1)+".txt --algorithm ucb --randomSeed " + str(sed) + " --epsilon 0 --horizon 204800 ")
	for sed in range(seedLen):
		os.system( "./bandit.sh --instance ../instances/i-"+str(i+1)+".txt --algorithm kl-ucb --randomSeed " + str(sed) + " --epsilon 0 --horizon 204800 ")
	for sed in range(seedLen):
		os.system( "./bandit.sh --instance ../instances/i-"+str(i+1)+".txt --algorithm thompson-sampling --randomSeed " + str(sed) + " --epsilon 0 --horizon 204800 ")
