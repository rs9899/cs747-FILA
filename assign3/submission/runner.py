import sys
import numpy as np

d_path = sys.argv[1]
f = open(d_path,'r')

numS = int(f.readline())
numA = int(f.readline())
gamm = float(f.readline())

## Reading the episode
data = f.readlines()
data = [x.split() for x in data ]
N = 0
S = []
A = []
R = []
for x in data[:-1]:
    N = N + 1
    S.append(int(x[0]))
    A.append(int(x[1]))
    R.append(float(x[2]))

S.append(int(data[-1][0]))

# TDLambda():
e = np.zeros(numS,)
V = np.zeros(numS,)
alph_0 = 0.5
lambd = 0.5
for episod in range(N):
    alph = alph_0 * (1.0/(1+(episod//100)))
    d = R[episod] + (gamm * V[S[episod+1]]) - V[S[episod]]
    e[S[episod]] += 1
    V = V + (alph*d*e)
    e = gamm*lambd*e
    
for i in range(numS):
    print(V[i])


## ERROR

printError = 0
if printError:
    f2 = open(sys.argv[2],'r')
    # data2 = f2.readlines()
    # data2 = [x.split() for x in data2 ]
    ###
    V_true = np.zeros(numS,)
    print("ERROR")
    for x in range(numS):
        V_true[x] = float(f2.readline())
    print(((V_true - V)**2 ).sum())