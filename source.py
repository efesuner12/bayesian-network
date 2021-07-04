# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 22:36:45 2021

"""

# load the package
from pomegranate import *
import numpy as np

#PROBLEM 1

#distributions for node A and node B and node C
nodeA = DiscreteDistribution({'a0' : 0.1, 'a1' : 0.9})
nodeB = DiscreteDistribution({'b0' : 0.4, 'b1' : 0.6})
nodeC = DiscreteDistribution({'c0' : 0.3, 'c1' : 0.7})

#conditional distributions for node D and node E and node F
nodeD = ConditionalProbabilityTable(
    [[ 'a0', 'b0', 'd0', 0.9],
     [ 'a0', 'b0', 'd1', 0.1],
     [ 'a0', 'b1', 'd0', 0.7],
     [ 'a0', 'b1', 'd1', 0.3],
     [ 'a1', 'b0', 'd0', 0.6],
     [ 'a1', 'b0', 'd1', 0.4],
     [ 'a1', 'b1', 'd0', 0.1],
     [ 'a1', 'b1', 'd1', 0.9]], [nodeA, nodeB])

nodeE = ConditionalProbabilityTable(
    [[ 'c0', 'd0', 'e0', 0.9],
     [ 'c0', 'd0', 'e1', 0.1],
     [ 'c0', 'd1', 'e0', 0.8],
     [ 'c0', 'd1', 'e1', 0.2],
     [ 'c1', 'd0', 'e0', 0.7],
     [ 'c1', 'd0', 'e1', 0.3],
     [ 'c1', 'd1', 'e0', 0.1],
     [ 'c1', 'd1', 'e1', 0.9]], [nodeC, nodeD])

nodeF = ConditionalProbabilityTable(
    [[ 'd0', 'f0', 0.8],
     [ 'd0', 'f1', 0.2],
     [ 'd1', 'f0', 0.3],
     [ 'd1', 'f1', 0.7]], [nodeD])

#Implementation
#nodes
s1 = State(nodeA, name = "A")
s2 = State(nodeB, name = "B")
s3 = State(nodeC, name = "C")
s4 = State(nodeD, name = "D")
s5 = State(nodeE, name = "E")
s6 = State(nodeF, name = "F")

model = BayesianNetwork("Model")
model.add_states(s1, s2, s3, s4, s5, s6)

#edges
model.add_edge(s1, s4)
model.add_edge(s2, s4)
model.add_edge(s4, s5)
model.add_edge(s3, s5)
model.add_edge(s4, s6)

model.bake()

#quesion 4 results
print(model.predict_proba([[None, None, None, None, None, None]]))
print(model.probability([['a0', 'b1', 'c0', 'd1', 'e0', 'f1']]))
print(model.predict_proba([[None, None, None, None, None, 'f0']]))
print(model.predict_proba([['a0', 'b0', None, None, None, None]]))
print(model.predict_proba([[None, None, None, None, 'e1', None]]))


#PROBLEM 2

# load the dataset
x = np.loadtxt('data')

#build the model from the dataset
model1 = BayesianNetwork.from_samples(x, algorithm = 'exact-dp')

#the structure
print(model1.structure)

#question 3
print(model1.predict([[0,1,0,1,1,1,0,1,0,0,1, None]]))
print(model1.predict([[0,0,0,1,1,0,0,1,0,1,1, None]]))
print(model1.predict([[0,1,1,0,0,0,0,1,0,1,1, None]])) 
print(model1.predict([[1,0,1,1,0,1,0,1,1,0,0, None]])) 
print(model1.predict([[1,1,1,1,0,1,1,0,1,0,1, None]])) 
print(model1.predict([[0,1,0,0,0,1,0,0,1,0,1, None]]))




