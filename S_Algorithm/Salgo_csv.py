import pandas as pd
import numpy as np
 

data = pd.read_csv("data - Sheet1.csv")
print(data,"\n")
 
d = np.array(data)[:,:-1]
print("\n The attributes are: ",d)
 
target = np.array(data)[:,-1]

def findS(c,t):
    for i, val in enumerate(t):
        if val == "Yes":
            specific_hypothesis = c[i].copy()
            break
             
    for i, val in enumerate(c):
        if t[i] == "Yes":
            for x in range(len(specific_hypothesis)):
                if val[x] != specific_hypothesis[x]:
                    specific_hypothesis[x] = '?'
                else:
                    pass
                 
    return specific_hypothesis
 
print("\n The final hypothesis is:",findS(d,target))
