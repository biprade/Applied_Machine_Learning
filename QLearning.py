__author__ = 'biprade'
from numpy.random import choice
from copy import deepcopy
import sys
class Qvalue:


    def __init__(self):
        self.Value=dict()
        self.Value['Right']=0.0
        self.Value['Left']=0.0
        self.Value['Up']=0.0
        self.Value['Down']=0.0

    def isEqual(self,qVal):

            result=qVal.Value['Right']-self.Value['Right']==0 and qVal.Value['Left']-self.Value['Left']==0 and qVal.Value['Up']-self.Value['Up']==0 and qVal.Value['Down']-self.Value['Down']==0

            return result

    def getMaxQvalue(self):
        maxQvalueDirection=max(self.Value, key=self.Value.get)
        maxQvalue=self.Value.get(maxQvalueDirection)
        return (maxQvalueDirection,maxQvalue)

    def displayValue(self):
        for key in self.Value:
            print key," : ",self.Value[key],"  ",



def randomChoice(listOfChoices,listOfProbabilities):
    """
    This function returns an element randomly from the listOfChoices based on the probability distribution
    :param listOfChoices: a list of choices
    :param listOfProbabilities: a list of probability for selection for each of the choices
    :return: a choice
    """
    return choice(listOfChoices, 1, p=listOfProbabilities)





def isConverged(matrix1,matrix2):
    """
    This function checks if the 2 matrix are exactly same by values for convergence
    :param matrix1: Matrix of previous episode
    :param matrix2: Matrix of current episode
    :return: True if converged else False
    """

    for i in range(0,5):
        for j in range(0,4):
            if not matrix1[i][j].isEqual(matrix2[i][j]):

                return False
    return True



def getNextState(currentState,action):
    """
    This function returns the co-ordinates of the new state
    :param currentState:  Co-ordinates of the current state
    :param action: Action to be performed
    :return: The co-ordinates of the new state
    """
    # If its a goal state return the same state as the next state. Goal state is the absorbing state.
    if (currentState[0]==0 and currentState[1]==3):
        return (currentState[0],currentState[1],action)
    else:
        if action=='Left':
            x=currentState[0]
            y=currentState[1]-1
            direction=action
        elif action=='Right':
            choice=randomChoice(['Right','Down'],[0.8,0.2])
            if choice[0]=='Right':
                x=currentState[0]
                y=currentState[1]+1
            elif choice[0]=='Down':
                x=currentState[0]+1
                y=currentState[1]
            direction=choice[0]
        elif action=='Up':
            choice=randomChoice(['Up','Left'],[0.8,0.2])
            if choice[0]=='Up':
                x=currentState[0]-1
                y=currentState[1]
            elif choice[0]=='Left':
                x=currentState[0]
                y=currentState[1]-1
            direction=choice[0]
        elif action=='Down':
            x=currentState[0]+1
            y=currentState[1]
            choice=action
            direction=action
        if x < 0 or x > 4 :
            x=currentState[0]
        if y < 0 or y > 3 :
            y=currentState[1]
        if (x==3 and y==1) or (x==3 and y==3):
            x=currentState[0]
            y=currentState[1]
        # print x,y
        return (x,y,direction)

def isGoalReached(currentState):
    if (currentState[0]==0 and currentState[1]==3):
        return True
    return False

def getReward(currentState):
    x=currentState[0]
    y=currentState[1]
    if x==1 and y==1:
        return -50.0
    elif x==0 and y==3:
        return 10.0
    else:
        return -1.0

def displayResultMatrix(matrix):
    for i in range(0,5):
        for j in range(0,4):
             print "( ",i," , ",j," ) : ",
             matrix[i][j].displayValue()
             print
        print







def main():
    # Q-Value matrix
    QValueMatrix=[[Qvalue() for j in range (4)] for i in range (5)]


    prevMatrix=deepcopy(QValueMatrix)

    alpha=0.5
    epsilon=0.9
    gamma=0.9
    episodeNumber=0

    allowedActions=['Left','Right','Up','Down']

    while True:
        currentState=(4,0)
        # Randomly chosing explore or exploit
        choice=randomChoice(['explore','exploit'],[epsilon,1-epsilon])

        if choice=='explore':
            while not isGoalReached(currentState):
                        x=currentState[0]
                        y=currentState[1]

                        if not (x==3 and y==1) or not (x==3 and y==3):

                            randomAction=randomChoice(allowedActions,[0.25,0.25,0.25,0.25])
                            nextState=getNextState(currentState,randomAction[0])
                            x1=nextState[0]
                            y1=nextState[1]
                            QvalueOfNextState=QValueMatrix[x1][y1].getMaxQvalue()

                            QValueMatrix[x][y].Value[nextState[2]]=QValueMatrix[x][y].Value[nextState[2]]+alpha * (getReward((x,y))+gamma * QvalueOfNextState[1]- QValueMatrix[x][y].Value[nextState[2]])
                            currentState=nextState



        elif choice=='exploit':
            while not isGoalReached(currentState):
                    x=currentState[0]
                    y=currentState[1]
                    maxQvalue=-sys.maxint
                    maxQValueDirection= None
                    if not (x==3 and y==1) or not (x==3 and y==3):
                        nextStateSelected = 0
                        for action in allowedActions:
                            nextState=getNextState(currentState,action)
                            x1=nextState[0]
                            y1=nextState[1]
                            nextStateMaxQValue=QValueMatrix[x1][y1].getMaxQvalue()
                            if nextStateMaxQValue[1] > maxQvalue:
                                nextStateSelected = nextState
                                maxQvalue=nextStateMaxQValue[1]
                                maxQValueDirection=nextState[2]
                        QValueMatrix[x][y].Value[maxQValueDirection]=QValueMatrix[x][y].Value[maxQValueDirection]+alpha * (getReward((x,y))+gamma * maxQvalue- QValueMatrix[x][y].Value[maxQValueDirection])
                        currentState=nextStateSelected



        randomAction=randomChoice(allowedActions,[0.25,0.25,0.25,0.25])
        nextState=getNextState(currentState,randomAction[0]) # is this call necessary ?
        x1=nextState[0]
        y1=nextState[1]
        QValueMatrix[x1][y1].Value[nextState[2]]=QValueMatrix[x1][y1].Value[nextState[2]]+ alpha * (getReward((x1,y1))+gamma * QValueMatrix[x1][y1].getMaxQvalue()[1] - QValueMatrix[x1][y1].Value[nextState[2]])

        # Increase the episode number once the goal state is reached
        episodeNumber=episodeNumber+1

        # After every 10 episodes decrease the epsilon
        if (episodeNumber%100)==0:
            
            epsilon=epsilon/(1+epsilon)
        if isConverged(QValueMatrix,prevMatrix)== True:

            break
        else:
            prevMatrix=deepcopy(QValueMatrix)

    displayResultMatrix(QValueMatrix)
if __name__ == "__main__":
    main()