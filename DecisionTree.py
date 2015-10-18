
########################################################################################
# Author :Bipra De , Anand Sharma                                                      #
# Email used:  bde@indiana.edu , asaurabh@indiana.edu                                  #
# Date : 21-Sept-2015                                                                  #
# About : Decision Tree using ID3 mechanism                                            #
# Note : The training data file must have a feature header                             #
########################################################################################





from __future__ import division

import csv
import math
from pandas.core.frame import DataFrame
from locale import str


featureList=[]                          # list of features extracted from the training data
featureAndValueMapping={}               # A map of all the feature and their values extracted from the training data
root=None


def readCSVFile(filePath):
    """
    This method reads the CSV file and extract the list of features and their distinct values
    :param filePath:
    :return:listOfRows from training data
    """
    global featureList
    global featureAndValueMapping
    listOfRows=[]
    input=open(filePath,'rU')
    hasHeader=csv.Sniffer().has_header(input.readline())
    if not hasHeader:
        raise ValueError('The file has no header!!!')
    else:
        input.seek(0)
        csvObject=csv.reader(input)

        for row in csvObject:
            featureList=row
            break

        for num in range(0,len(featureList)):
            featureAndValueMapping[featureList[num]]=[]

        for row in csvObject:
             for num in range(0,len(featureList)):
                 if row[num] not in featureAndValueMapping[featureList[num]]:
                     featureAndValueMapping[featureList[num]].append(row[num])
        input.seek(0)
        for row in csvObject:
            listOfRows.append(row)
        listOfRows.pop(0)
        return listOfRows



def calculateEntropy(data):
    """
    Calculate the entropy of input data
    :param input data:
    :return:entropy of data
    """

    entropy=0
    for goalClass in featureAndValueMapping[featureList[len(featureList)-1]]:
        temp = [row for row in data if goalClass == row[len(featureList)-1]]

        p=len(temp)/len(data)
        if p==0:
            entropy+=0
        else:
            entropy=entropy+p * math.log(p,2)
    return -(entropy)


def findMajorityClass(data):
    """
    This method finds the majority class at leaf nodes
    :param data:
    :return: majortiy class to be used as Classification label
    """
    majorityClass=""
    maxCount=0
    for featureValue in featureAndValueMapping[featureList[len(featureList)-1]]:
        temp= [row for row in data if featureValue == row[len(featureList)-1]]
        if (len(temp)>maxCount):
            majorityClass=featureValue
            maxCount=len(temp)
    return majorityClass


def selectBestFeature(data,listOfFeatures):
    """
    This method returns the best feature from the current data distribution
    :param listOfFeatures:
    :param data:
    :return:best feature
    """
    isOnlyOneClassPresent=False
    isEmptyData=False
    bestFeature=""
    maxInformationGain=0;
    parentEntropy=0
    informationGain=0
    if len(data)==0:
        isEmptyData=True
    elif calculateEntropy(data)==0:
        isOnlyOneClassPresent=True
    else:
        parentEntropy=calculateEntropy(data)
        for feature in listOfFeatures:
            childEntropy=0
            totalDataAfterSplit=0
            for featureValue in featureAndValueMapping[feature]:
                tempData = [row for row in data if featureValue == row[featureList.index(feature)]]
                totalDataAfterSplit=totalDataAfterSplit+len(tempData)
            for featureValue in featureAndValueMapping[feature]:
                tempData = [row for row in data if featureValue == row[featureList.index(feature)]]
                lengthOfData=len(tempData)
                if lengthOfData==0:
                    entropy=0
                else:
                    entropy=calculateEntropy(tempData)
                childEntropy=childEntropy+entropy * (lengthOfData/totalDataAfterSplit)
            informationGain=parentEntropy-childEntropy

            if informationGain >= maxInformationGain:
                maxInformationGain=informationGain
                bestFeature=feature
    return bestFeature,isOnlyOneClassPresent,isEmptyData


"""
Node structure of the decision tree
"""
class Node:
    def __init__(self,featureName=None,isLeaf=None, majorityClass=None):
        if featureName is None:
            featureName=""
        else:
            self.featureName=featureName
        if isLeaf is None:
            self.isLeaf=False
        else:
            self.isLeaf=isLeaf
        if majorityClass is None:
            self.majorityClass=""
        else:
            self.majorityClass=majorityClass
        self.childNodes={}


def constructDecisionTree(trainingFilePath,depth):
    """
    This method calls the computeAndJoinDecisionTreeNodes() to create decision tree on the trained model
    :param trainingFilePath: path to the training file
    :param depth: depth for which the tree needs to be constructed

    """
    trainingData=readCSVFile(trainingFilePath)
    listOfFeatures=featureList[:]
    listOfFeatures.remove(featureList[len(featureList)-1])
    computeAndJoinDecisionTreeNodes(trainingData,listOfFeatures,Node("",False,False),depth,"")



def computeAndJoinDecisionTreeNodes(data,listOfFeatures,parentNode,depth,featureValue):
    """
    This method is used to construct the desion tree from the training data and for the given depth
    :param data: Training data
    :param listOfFeatures: list of features in the training data
    :param parentNode:
    :param depth:
    :param featureValue:
    :return: root node of the learned trained model
    """
    global root

    # base condition to end the recursion

    if len(data)==0 or parentNode.isLeaf or len(listOfFeatures)==0:
        return Node("",True,parentNode.majorityClass)

    feature=selectBestFeature(data,listOfFeatures)
    majorityClass=findMajorityClass(data)

    if depth==0:
        node=Node(feature[0],True,majorityClass)
        return node

    if feature[1]:                              # Only one class is present in the data distribution at the given node
        node= Node("",True,majorityClass)
        if root==None:
            root =node

    elif feature[2]:                            # There is no data  at the given node
        node=Node("",True,parentNode.majorityClass)
        if root==None:
            root =node

    else:                                       # This is a non leaf node which can be further split
        node=Node(feature[0],False,majorityClass)
        if root==None:
            root =node
        if(feature[0]==""):
            print parentNode.isLeaf,parentNode.majorityClass,parentNode.featureName,featureValue

        listOfFeatures.remove(feature[0])
        for featureValue in featureAndValueMapping[feature[0]]:
            tempdata=[row for row in data if featureValue == row[featureList.index(feature[0])]]
            node.childNodes[featureValue]=computeAndJoinDecisionTreeNodes(tempdata,listOfFeatures,node,depth-1,featureValue)
    return node




def displayTree(node,depth):
    """
    This method is used to travers the trained decision tree and print it
    :param node: root node of the trained decision tree

    """
    if node.isLeaf:
        return
    else:
        print "\n---------------------------------------------"+" Depth : "+str(depth)+"-------------------------------------------------\n"
        print "Feature Name : ", node.featureName
        print
        for key in node.childNodes:
            if node.childNodes[key].isLeaf:
                print "If Feature Value : ",  key, "\t\t\t\t\t  Then Assign Class :", node.childNodes[key].majorityClass
            else:
                print "If Feature Value : ", key,"\t\t\t\t\t\t  Then check for Feature ", node.childNodes[key].featureName
        print "\n\n"
        for key in node.childNodes:
            displayTree(node.childNodes[key],depth+1)


def classifyTestData(testFilePath,modelRoot):
    """
    This method calls the traverseDecisionTreeModel() to classify the test data on the trained model and generate Confusion matrix and error at the given depth
    :param testFilePath: Path to the test file
    :param modelRoot: Root node of the decision tree of the trained model

    """
    correctlyClassifiedInstances=0
    incorrectlyClassifiedInstances=0
    testDataList=[]
    input=open(testFilePath,'rU')
    csvObject=csv.reader(input)
    label = featureList[len(featureList) -1]
    classLabels = featureAndValueMapping.get(label)
    classLabelCount = len(classLabels)
    ConfusionMatrix = [[0 for x in range(int(classLabelCount))] for x in range(int(classLabelCount))]
    for row in csvObject:
        predictedLabel=traverseDecisionTreeModel(row,root)
        ConfusionMatrix[int(row[len(row)- 1]) - 1][int(predictedLabel) - 1] += 1

        if predictedLabel==row[len(row)-1]:
            correctlyClassifiedInstances+=1
        else:
            incorrectlyClassifiedInstances+=1
    df = DataFrame(ConfusionMatrix)
    df.columns = classLabels
    df.index = classLabels

    print "Confusion Matrix :: \n"
    print df
    print "Correctly Classified Instance ",correctlyClassifiedInstances
    print "Incorrectly Classified Instance ",incorrectlyClassifiedInstances



def traverseDecisionTreeModel(row,node):
    """
    Recursive function to itearte over the trained decision tree and classify the input test row
    :param row: input test data row
    :param node: root node of the trained decicion tree
    :return: class label
    """
    if not node.isLeaf:
        val=row[featureList.index(node.featureName)]
        direction=node.childNodes
        if val not in direction:
            return node.majorityClass
        nextNodeToVisit=direction[val]
        return traverseDecisionTreeModel(row,nextNodeToVisit)
    else:
        return node.majorityClass


trainingFilePath=input("Enter absolute path of training data : ")
depth=input("Enter depth of the decision tree : ")
constructDecisionTree(trainingFilePath,depth)
displayTree(root,0)
testFilePath=input("Enter absolute path of test data : ")
classifyTestData(testFilePath,root)











