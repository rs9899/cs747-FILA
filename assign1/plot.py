import matplotlib.pyplot as plt
import csv
import math
inst = {}
horizons = [50,200,800,3200,12800,51200,204800]
inst['../instances/i-1.txt']={}
inst['../instances/i-2.txt']={}
inst['../instances/i-3.txt']={}
algos=["round-robin", "epsilon-greedy0.002","epsilon-greedy0.02","epsilon-greedy0.2", "ucb", "kl-ucb", "thompson-sampling"]
##
# instance
### algo
###### horizon
######### [list of values]
with open('./submission/outputData.txt') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for l in csv_reader:
		ins = l[0]
		curr_dic = inst[ins]
		algo = l[1][1:]
		horizon = int(l[4][1:])
		seed = int(l[2][1:])
		value = float(l[-1][1:])
		eps = float(l[3][1:])
		if(algo=="epsilon-greedy"):
			algo=algo+str(eps)
		if(not (algo in curr_dic)):
			curr_dic[algo]={}
		curr_dic=curr_dic[algo]
		if(not (horizon in curr_dic)):
			curr_dic[horizon]=0
		curr_dic[horizon] +=value

for a in inst:
	for b in inst[a]:
		for c in inst[a][b]:
			# print(c)
			inst[a][b][c]/=50.0


for instances in ["../instances/i-1.txt","../instances/i-2.txt","../instances/i-3.txt"]:
	d=inst[instances]
	for a in algos:
		y_l = [d[a][h] for h in horizons]
		plt.plot([math.log(h) for h in horizons], [math.log(z) for z in y_l] )
	plt.legend(['round-robin', 'epsilon-greedy with epsilon=0.002', 'epsilon-greedy with epsilon=0.02', 'epsilon-greedy with epsilon=0.2','ucb','kl-ucb','thompson-sampling'], loc='upper left')
	plt.xlabel("horizon (in log scale)")
	plt.ylabel("Regret (in log scale)")
	plt.show()
