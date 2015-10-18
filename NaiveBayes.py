from __future__ import division
from pandas.core.frame import DataFrame
import csv


featureList=[]                          # list of features extracted from the training data
featureAndValueMapping={}
naiveBayesModel={}
goalClassCount={}
priorProbability={}
classCount={}
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

def trainModel(trainData):
        global goalClassCount
        global priorProbability
        global naiveBayesModel
        global classCount
        global featureList
        global featureAndValueMapping
        sizeOfData=len(trainData)
        # Counting number of instances in the training data for each goal class
        for goalClass in featureAndValueMapping[featureList[len(featureList)-1]]:
            rowWithGoalClass=[row for row in trainData if goalClass == row[len(featureList)-1]]
            totalCount=len(rowWithGoalClass)
            goalClassCount[goalClass]=totalCount
            priorProbability[goalClass]=totalCount/sizeOfData
            classCount[goalClass]=totalCount

        for feature in featureList[:len(featureList)-1]:
            for featureValue in featureAndValueMapping[feature]:
                for targetValue in featureAndValueMapping[featureList[len(featureList)-1]]:
                    temp = [row for row in trainData if featureValue == row[featureList.index(feature)] and targetValue==row[len(featureList)-1]]
                    key=feature+":"+featureValue+"|"+featureList[len(featureList)-1]+":"+targetValue
                    naiveBayesModel[key]=(len(temp)+1)/(classCount[targetValue]+len(featureAndValueMapping[feature])) # Laplace correction




def classify(row):
    global naiveBayesModel
    global featureList
    global featureAndValueMapping
    maxprobability=-(float("inf"))
    predictedClass=None
    probability=1


    for targetValue in featureAndValueMapping[featureList[len(featureList)-1]]:
        index=0
        probability=1
        for col in row:
            key=featureList[index]+":"+col+"|"+featureList[len(featureList)-1]+":"+targetValue

            # Using Laplace correction
            if  key in naiveBayesModel:
                probability=probability*naiveBayesModel[key]
            else:
                probability=probability*(1/len(featureAndValueMapping[featureList[index]]))
            index=index+1
        if probability>maxprobability:
            maxprobability=probability
            predictedClass=targetValue
    return predictedClass

def runModelOnTest(testFilePath):
    classLabels = featureAndValueMapping.get(featureList[len(featureList) -1])
    classLabelCount = len(classLabels)
    ConfusionMatrix = [[0 for x in range(int(classLabelCount))] for x in range(int(classLabelCount))]
    input=open(testFilePath,'rU')
    csvObject=csv.reader(input)
    for row in csvObject:
        predictedLabel=classify(row[:len(row)-1])
        ConfusionMatrix[int(row[len(row)- 1])][int(predictedLabel)] += 1
        # print "Actual label : "+row[len(row)- 1]+"Class label : "+classify(row[:len(row)-1])
    df = DataFrame(ConfusionMatrix)
    df.columns = classLabels
    df.index = classLabels
    print df





trainingFilePath='/users/biprade/documents/Applied_ML/Programming_Assignment_2/zoo-train.csv' #input("Enter absolute path of training data : ")
trainingData=readCSVFile(trainingFilePath)
trainModel(trainingData)
testFilePath='/users/biprade/documents/Applied_ML/Programming_Assignment_2/zoo-test.csv'
runModelOnTest(testFilePath)