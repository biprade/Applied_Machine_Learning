__author__ = 'biprade'
from copy import deepcopy
import sys
import numpy as np

Reward = [
					[-1,-1,-1, 10],
					[-1, -50,-1,-1],
					[-1,-1,-1,-1],
					[-1,-sys.maxint,-1,-sys.maxint],
					[-1,-1,-1,-1]
					]

previousValue=[[0 for x in range(4)] for x in range(5)]
row=5
column=4

transitionProbabilities={'Right':(0.8,0.2),'Left':1.0,'Up':(0.8,0.2),'Down':1.0}
count=0


def canConverge(TempValue,Value):
    matrix1=np.matrix(TempValue)
    matrix2=np.matrix(Value)

    if np.allclose(matrix1,matrix2):

        return True
    else:
        # print matrix1
        # print matrix2
        return False


Value=[[0 for x in range(4)] for x in range(5)]
while True:

    for i in range(0,row):
        for j in range(0,column):
           if Reward[i][j]!=-sys.maxint:
                    vmax=-sys.maxint
                    for action in transitionProbabilities:
                        if (action=='Left'):
                            if (j-1<0 or Reward[i][j-1]==-sys.maxint):
                                v=Reward[i][j]+0.9*transitionProbabilities[action]*Value[i][j]
                            else:
                                v=Reward[i][j]+0.9*transitionProbabilities[action]*Value[i][j-1]
                        elif (action=='Right'):
                            if(j+1>column-1 or Reward[i][j+1]==-sys.maxint):
                                v=0.9*transitionProbabilities[action][0]*Value[i][j]
                            else:
                                v=0.9*transitionProbabilities[action][0]*Value[i][j+1]
                            if (i+1>row-1 or Reward[i+1][j]==-sys.maxint):
                                v=v+0.9*transitionProbabilities[action][1]*Value[i][j]
                            else:
                                v=v+0.9*transitionProbabilities[action][1]*Value[i+1][j]
                            v=v+Reward[i][j]
                        elif (action=='Up'):
                            if (i-1<0 or Reward[i-1][j]==-sys.maxint):
                                v=0.9*transitionProbabilities[action][0]*Value[i][j]
                            else:
                                v=0.9*transitionProbabilities[action][0]*Value[i-1][j]
                            if (j-1<0 or Reward[i][j-1]==-sys.maxint):
                                v=v+0.9*transitionProbabilities[action][1]*Value[i][j]
                            else:
                                v=v+0.9*transitionProbabilities[action][1]*Value[i][j-1]
                            v=v+Reward[i][j]
                        elif (action=='Down'):
                            if(i+1>row-1 or Reward[i+1][j]==-sys.maxint):
                                v=Reward[i][j]+0.9*transitionProbabilities[action]*Value[i][j]
                            else:
                                v=Reward[i][j]+0.9*transitionProbabilities[action]*Value[i+1][j]
                        if v>=vmax:
                            vmax=v

                    Value[i][j]=vmax

    if (canConverge(previousValue,Value)):
        break;
    else:
        previousValue=deepcopy(Value)
        count+=1
print count
print np.array(Value)

