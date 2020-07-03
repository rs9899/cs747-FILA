import sys
import argparse
import numpy as np
import pulp

## Parse the arguments

parser = argparse.ArgumentParser()
parser.add_argument("--mdp" , required=True, help="path to the instance file")
parser.add_argument("--algorithm" , required=True, help="one of lp and hpi")

args = vars(parser.parse_args())

fl = open(args['mdp'] , 'r')
algo = args['algorithm']
############################################
# File Reading
S = int(fl.readline())
A = int(fl.readline())
# print(S , A)

T = np.zeros([S,A,S])
R = np.zeros([S,A,S])


for i in range(S):
    for j in range(A):
        ln = fl.readline()
        R[i,j,:] = np.array([float(k) for k in ln.strip().split('\t')])

for i in range(S):
    for j in range(A):
        ln = fl.readline()
        T[i,j,:] = np.array([float(k) for k in ln.strip().split('\t')])

gamma = float(fl.readline())

typ = fl.readline()
typ = str(typ.strip())


fl.close()
###########################
RT = R*T # To avoid multiplt calculation

np.random.seed(42) # just a check

def V_calc(T,RT,S,policy,gamma,typ = 'continuing'):
	## Av=b equation formation
    a_ = T[range(S),policy,:]
    b_ = RT[range(S),policy,:]
    b_ = np.sum(b_ , axis = 1)
    a_ = np.eye(S) - (gamma * a_)
    if typ == 'episodic':
    	# Last V is fixed for episodic so ....
        a_ = a_[:-1,:-1]
        b_ = b_[:-1]
    V = np.linalg.solve(a_,b_)
    if typ == 'episodic':
        V = np.pad(V, (0, 1), 'constant')
    return V

def Policy_calc(T,RT,S,V,gamma,typ = 'continuing'):
    Q_imd = RT + gamma * T * V.reshape([1,1,S]) 
    Q = np.sum(Q_imd,axis=2)
    new_policy = np.argmax(Q,axis = 1) ## Go for the best policy
    return new_policy

best_policy = np.random.randint(0,A,S)
best_value = np.random.rand(S)


if algo == 'hpi':
    policy = np.random.randint(0,A,S)
    # print(policy)
    max_iter = 10
    # while True:
    for iterr in range(max_iter): 
        V = V_calc(T,RT,S,policy,gamma)
        new_policy = Policy_calc(T,RT,S,V,gamma)
    #     print(list(zip(V,new_policy)))
        if np.sum(policy == new_policy) == S:
            best_policy = new_policy
            best_value = V_calc(T,RT,S,new_policy,gamma)
            best_policy = best_policy.flatten()
            best_value = best_value.flatten()
            break
        policy = new_policy
else:
    prob = pulp.LpProblem('MDP' , pulp.LpMinimize)
    val_list = []
    S_temp = S
    if typ == 'episodic':
        S_temp = S - 1
    for i in range(S_temp):
        variabl = 'V' + str(i)
        variabl = pulp.LpVariable(variabl)
        val_list.append(variabl)
    prob += pulp.lpSum(val_list), "Main Objective"
    ## Constraint
    RT_summed = np.sum(RT,axis = 2)
    for s in range(S_temp):
        for a in range(A):
            prob += pulp.lpSum([ T[s,a,s_p] * val_list[s_p] for s_p in range(S_temp)]) * gamma + RT_summed[s,a] <=val_list[s] , "for each s = "+ str(s)+",a = "+ str(a)+" constraint" 
    prob.solve()
    best_value = np.zeros([S,]).flatten()
    for v in prob.variables():
        best_value[int(str(v.name)[1:])] = v.varValue
    best_policy = Policy_calc(T,RT,S,best_value,gamma)


for iterr in range(S):
    print("{:.15f}\t{}".format(best_value[iterr] , best_policy[iterr]))


