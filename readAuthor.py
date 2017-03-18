__author__ = 'penghao'
import numpy as np
import glob

def readauthor(type,path):
    d = '*.author'
    datapath = path + d
    datafile = glob.glob(datapath)
    row = len(datafile)
    totalAuthorList = []

    newFeaMatrix = np.zeros((row,3))
    for i in range(row):
        infile = np.loadtxt(datafile[i], dtype = str)
        for author in infile.tolist():
            totalAuthorList.append(author)
    l = len(totalAuthorList)

    authorMatrix = [[0 for col in range(3)] for row in range(l)]
    print totalAuthorList
    aName =[] # author name set
    for element in totalAuthorList:
        aName.append(element[1])
    aNameSet = set(aName)
    totalName = list(aNameSet)
    for idx,element in enumerate(totalAuthorList):
        authorMatrix[idx][0] = element[0]
        authorMatrix[idx][1] = element[1]
        authorMatrix[idx][2] = totalName.index(element[1])
    nameDict ={}
    for element in authorMatrix:
        nameDict[element[0]] = element[2]
    print authorMatrix
    print nameDict
    print nameDict['G6']
    return nameDict
    #print authorMatrix


readauthor('abortion','/Users/penghao/GitHub/ML_Stance/stance/authors/abortion/')