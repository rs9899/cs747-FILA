import matplotlib.pyplot as plt
import pandas as pd
src=open("outputData.txt","r")
first_line="instance,algorithm,randomSeed,epsilon,horizon,REG\n"    #Prepending string
oline=src.readlines()
#Here, we prepend the string we want to on first line
oline.insert(0,first_line)
src.close()

dest = open("output.txt", "w")
dest.writelines(oline)
df = pd.read_csv("output.txt")
df.groupby(["instance","algorithm","horizon","epsilon"]).mean().to_csv("helper.txt")
a = pd.read_csv("helper.txt")
horizon = [50,200,800,3200,12800,51200,204800]
for instance in ["../instances/i-1.txt","../instances/i-2.txt","../instances/i-3.txt"]:
	for algorithm in ["round-robin", "epsilon-greedy", "ucb", "kl-ucb", "thompson-sampling"]:
		if algorithm == "epsilon-greedy":
			for epsilon in [0.002, 0.02, 0.2]:
				reg_array = a.query("instance ==" + "'" + instance + "'" + " & algorithm == " + "'"+ algorithm + "'" + " & epsilon == " + str(epsilon))["REG"].tolist()
				plt.plot(horizon, reg_array)
		else:
			reg_array = a.query("instance ==" + "'" + instance + "'" + " & algorithm == " + "'"  + algorithm + "'")["REG"].tolist()
			plt.plot(horizon, reg_array)
	plt.legend(['roun-robin', 'epsilon-greedy with epsilon=0.002', 'epsilon-greedy with epsilon=0.02', 'epsilon-greedy with epsilon=0.2','ucb','kl-ucb','thompson-sampling'], loc='upper left')
	plt.xlabel("horizon")
	plt.ylabel("Regret")
	plt.show()
	# if instance == "../instances/i-1.txt":
	# 	plt.savefig("instance1.png")
	# elif instance == "../instances/i-2.txt":
	# 	plt.savefig("instance2.png")
	# else:
	# 	plt.savefig("instance3.png")