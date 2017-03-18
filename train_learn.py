__author__ = 'penghao'
import numpy as np
from sklearn import svm
from textblob import TextBlob
import glob
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM

from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def calfold(flag):

    fold1List =[]
    fold2List =[]
    fold3List =[]
    fold4List =[]
    fold5List =[]
    if flag == 'a':
        fold1 = np.loadtxt('/Users/penghao/GitHub/ML_Stance/stance/folds/abortion_folds/Fold-1',dtype=object)
        fold2 = np.loadtxt('/Users/penghao/GitHub/ML_Stance/stance/folds/abortion_folds/Fold-2',dtype=object)
        fold3 = np.loadtxt('/Users/penghao/GitHub/ML_Stance/stance/folds/abortion_folds/Fold-3',dtype=object)
        fold4 = np.loadtxt('/Users/penghao/GitHub/ML_Stance/stance/folds/abortion_folds/Fold-4',dtype=object)
        fold5 = np.loadtxt('/Users/penghao/GitHub/ML_Stance/stance/folds/abortion_folds/Fold-5',dtype=object)
        abMatrix = np.load('abMatrix.npy')
    if flag == 'g':
        fold1 = np.loadtxt('/Users/penghao/GitHub/ML_Stance/stance/folds/gayRights_folds/Fold-1',dtype=object)
        fold2 = np.loadtxt('/Users/penghao/GitHub/ML_Stance/stance/folds/gayRights_folds/Fold-2',dtype=object)
        fold3 = np.loadtxt('/Users/penghao/GitHub/ML_Stance/stance/folds/gayRights_folds/Fold-3',dtype=object)
        fold4 = np.loadtxt('/Users/penghao/GitHub/ML_Stance/stance/folds/gayRights_folds/Fold-4',dtype=object)
        fold5 = np.loadtxt('/Users/penghao/GitHub/ML_Stance/stance/folds/gayRights_folds/Fold-5',dtype=object)
        abMatrix = np.load('gayRMatrix.npy')
    if flag == 'm':
        fold1 = np.loadtxt('/Users/penghao/GitHub/ML_Stance/stance/folds/marijuana_folds/Fold-1',dtype=object)
        fold2 = np.loadtxt('/Users/penghao/GitHub/ML_Stance/stance/folds/marijuana_folds/Fold-2',dtype=object)
        fold3 = np.loadtxt('/Users/penghao/GitHub/ML_Stance/stance/folds/marijuana_folds/Fold-3',dtype=object)
        fold4 = np.loadtxt('/Users/penghao/GitHub/ML_Stance/stance/folds/marijuana_folds/Fold-4',dtype=object)
        fold5 = np.loadtxt('/Users/penghao/GitHub/ML_Stance/stance/folds/marijuana_folds/Fold-5',dtype=object)
        abMatrix = np.load('marijuanaMatrix.npy')
    if flag == 'o':
        fold1 = np.loadtxt('/Users/penghao/GitHub/ML_Stance/stance/folds/obama_folds/Fold-1',dtype=object)
        fold2 = np.loadtxt('/Users/penghao/GitHub/ML_Stance/stance/folds/obama_folds/Fold-2',dtype=object)
        fold3 = np.loadtxt('/Users/penghao/GitHub/ML_Stance/stance/folds/obama_folds/Fold-3',dtype=object)
        fold4 = np.loadtxt('/Users/penghao/GitHub/ML_Stance/stance/folds/obama_folds/Fold-4',dtype=object)
        fold5 = np.loadtxt('/Users/penghao/GitHub/ML_Stance/stance/folds/obama_folds/Fold-5',dtype=object)
        abMatrix = np.load('obamaMatrix.npy')
    for idx,item in enumerate(abMatrix[:,1]):
        if item in fold1:
            fold1List.append(idx)
        if item in fold2:
            fold2List.append(idx)
        if item in fold3:
            fold3List.append(idx)
        if item in fold4:
            fold4List.append(idx)
        if item in fold5:
            fold5List.append(idx)

    #print fold1List
    return fold1List,fold2List,fold3List,fold4List,fold5List

def calAcc(flag,foldIndex,featureMatrix):
    if flag == 'a':
        abMatrix = np.load('abMatrix.npy')
        #aUnigramMatrix = np.load('anewUnigramMatrix.npy')

    if flag == 'g':
        abMatrix = np.load('gayRMatrix.npy')
        #aUnigramMatrix = np.load('gnewUnigramMatrix.npy')
    if flag == 'm':
        abMatrix = np.load('marijuanaMatrix.npy')
        #aUnigramMatrix = np.load('mnewUnigramMatrix.npy')
    if flag == 'o':
        abMatrix = np.load('obamaMatrix.npy')
        #aUnigramMatrix = np.load('oUnigramMatrix.npy')
    #print flag
    xData = featureMatrix
    xTarget= abMatrix[:,5]

    fold1List,fold2List,fold3List,fold4List,fold5List = calfold(flag)
    if foldIndex == 5:
        foldTrain = fold1List+fold2List+fold3List+fold4List
        foldTest = fold5List
    if foldIndex == 4:
        foldTrain = fold1List+fold2List+fold3List+fold5List
        foldTest = fold4List
    if foldIndex == 3:
        foldTrain = fold1List+fold2List+fold5List+fold4List
        foldTest = fold3List
    if foldIndex == 2:
        foldTrain = fold1List+fold5List+fold3List+fold4List
        foldTest = fold2List
    if foldIndex == 1:
        foldTrain = fold5List+fold2List+fold3List+fold4List
        foldTest = fold1List

    X_train = xData[foldTrain]
    y_train = xTarget[foldTrain]
    X_test = xData[foldTest]
    y_test = xTarget[foldTest]
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(xData, xTarget, test_size=0.2, random_state=0)
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    #print(clf.predict(X_train))
    print clf.score(X_test, y_test)

    #model = ChainCRF()
    #ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=10)
    #ssvm.fit(X_train, y_train)
    #print ssvm.score(X_test, y_test)

    #clf2 = GaussianNB()
    #clf2.fit(X_train, y_train)
    #print clf2.score(X_test, y_test)

def calCLF(flag,foldIndex):
    if flag == 'a':
        abMatrix = np.load('abMatrix.npy')
        featureMatrix = np.load('anewUnigramMatrix.npy')

    if flag == 'g':
        abMatrix = np.load('gayRMatrix.npy')
        featureMatrix = np.load('gnewUnigramMatrix.npy')
    if flag == 'm':
        abMatrix = np.load('marijuanaMatrix.npy')
        featureMatrix = np.load('mnewUnigramMatrix.npy')
    if flag == 'o':
        abMatrix = np.load('obamaMatrix.npy')
        featureMatrix = np.load('oUnigramMatrix.npy')

    fold1List,fold2List,fold3List,fold4List,fold5List = calfold(flag)
    if foldIndex == 5:
        foldTrain = fold1List+fold2List+fold3List+fold4List
        foldTest = fold5List
    if foldIndex == 4:
        foldTrain = fold1List+fold2List+fold3List+fold5List
        foldTest = fold4List
    if foldIndex == 3:
        foldTrain = fold1List+fold2List+fold5List+fold4List
        foldTest = fold3List
    if foldIndex == 2:
        foldTrain = fold1List+fold5List+fold3List+fold4List
        foldTest = fold2List
    if foldIndex == 1:
        foldTrain = fold5List+fold2List+fold3List+fold4List
        foldTest = fold1List
    #print flag
    xData = featureMatrix
    xTarget= abMatrix[:,5]
    X_train = xData[foldTrain]
    y_train = xTarget[foldTrain]
    X_test = xData[foldTest]
    y_test = xTarget[foldTest]
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)


    predictionList = clf.predict(xData)
    #print clf.score(X_test, y_test)
    return predictionList

def calUniAcc():
    print 'a'
    for i in range(1,6):
        calAcc('a',i,np.load('anewUnigramMatrix.npy'))
    print 'g'
    for i in range(1,6):
        calAcc('g',i,np.load('gnewUnigramMatrix.npy'))
    print 'm'
    for i in range(1,6):
        calAcc('m',i,np.load('mnewUnigramMatrix.npy'))
    print 'o'
    for i in range(1,6):
        calAcc('o',i,np.load('oUnigramMatrix.npy'))

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
    #print totalAuthorList
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
    #print authorMatrix
    #print nameDict
    #print nameDict['G6']
    return nameDict
    #print authorMatrix

def newFeatureSet(orginalMatrix,uniMatrix,indict,type,foldIdx):
    l= len(orginalMatrix[:,0]) #the row of matrix
    newFeaMatrix = np.zeros((l,6)) #length, rebuttal, sentiment, Pos word, neg word
    #print orginalMatrix[:,5]
    for idx,element in enumerate(orginalMatrix[:,2]): #length of the doc
        newFeaMatrix[idx][0] = int(len(element))

    for idx,element in enumerate(orginalMatrix[:,6]):# rebuttal
        if element == 'oppose':
            newFeaMatrix[idx][1] = -1
            #print newFeaMatrix[idx][1]
            #print element
        if element == 'support':
            newFeaMatrix[idx][1] = 1
        if element == 'null':
            newFeaMatrix[idx][1] = 0

    # Add pro and cons, sentiment
    sList = []
    perPos= []
    perNeg = []
    for element in orginalMatrix[:,2]:
        sentence = ''
        wordCount = len(element)
        posCount = 0
        negCount = 0
        for words in element:
            sentence += words + " "
            wordblob = TextBlob(words)
            try:
                s=wordblob.sentiment.polarity
                if s > 0:
                    posCount += 1
                if s<0 :
                    negCount += -1

            except:
                pass
        perPos.append(float(negCount)/len(element))
        perNeg.append(float(posCount)/float(len(element)))
        try:
            blob = TextBlob(sentence)
            sentiment=blob.sentiment.polarity
            sList.append(sentiment)
            '''
            if sentiment > 0:
                sList.append(1)
            if sentiment < 0:
                sList.append(-1)
            if sentiment == 0:
                sList.append(0)
            '''
        except:
            sList.append(0)
            pass
    #####AC####
    for idx,sentiment in enumerate(sList):
        newFeaMatrix[idx][2] = sentiment
        newFeaMatrix[idx][3] = -1 # 0 is take in the index, becuase there are some of files are broken, take -1 to avoid conflict
        if orginalMatrix[idx][1] in indict: #file name

            newFeaMatrix[idx][3] = indict[orginalMatrix[idx][1]] # author index

    predictionList = calCLF(type,foldIdx)
    #print predictionList
    for idx,element in enumerate(predictionList):
        if element == '':
            predictionList[idx] = 0
    #print predictionList
    authorList = newFeaMatrix[:,3] #author index list
    authorSet = list(set(authorList))   #unique authors
    authorDict ={}
    for author in authorSet:
        ac = 0
        for idx, element in enumerate(authorList):

            if element == author:

                if int(predictionList[idx]):
                    ac += int(predictionList[idx]) #author stance by original clf


        if ac > 0:
            authorDict[int(author)] = 1
        if ac < 0:
            authorDict[int(author)] = -1

    authorDict[-1] = 0
    for idx,sentiment in enumerate(sList): # same as in range of length
        if newFeaMatrix[idx][3] in authorDict:
            authorIndex = newFeaMatrix[idx][3]
            newFeaMatrix[idx][4] = authorDict[authorIndex]
    '''
    ####UC#####
    rebuttalList = newFeaMatrix[:,1]  #rebuttal
    idList = orginalMatrix[:,3] #id
    pidList = orginalMatrix[:,4] #pid
    for idx, element in enumerate(idList):
        for i, pid in enumerate(pidList):

            if pid == element:
                #print pid , element #previous stance
                #print rebuttalList[i]
                newFeaMatrix[i][5]=int(predictionList[idx])*int(rebuttalList[i])
                #print newFeaMatrix[i][4],newFeaMatrix[i][5]
    #for element in newFeaMatrix[:,5]:
        #print element
    '''
    newF = np.concatenate((uniMatrix, newFeaMatrix), axis=1)
    return newF

aDict =readauthor('abortion','/Users/penghao/GitHub/ML_Stance/stance/authors/abortion/')
gDict =readauthor('gay','/Users/penghao/GitHub/ML_Stance/stance/authors/gayRights/')
mDict =readauthor('m','/Users/penghao/GitHub/ML_Stance/stance/authors/marijuana/')
oDict =readauthor('o','/Users/penghao/GitHub/ML_Stance/stance/authors/obama/')
def calAccG(type):
    if type == 'a':
        abMatrix = np.load('abMatrix.npy')
        aUnigramMatrix = np.load('anewUnigramMatrix.npy')

        print type
        for i in range(1,6):
            newF = newFeatureSet(abMatrix,aUnigramMatrix,aDict,type,i)
            calAcc(type,i,newF)
    if type == 'g':
        abMatrix = np.load('gayRMatrix.npy')
        aUnigramMatrix = np.load('gnewUnigramMatrix.npy')

        print type
        for i in range(1,6):
            newF = newFeatureSet(abMatrix,aUnigramMatrix,gDict,type,i)
            calAcc(type,i,newF)
    if type == 'm':
        abMatrix = np.load('marijuanaMatrix.npy')
        aUnigramMatrix = np.load('mnewUnigramMatrix.npy')

        print type
        for i in range(1,6):
            newF = newFeatureSet(abMatrix,aUnigramMatrix,mDict,type,i)
            calAcc(type,i,newF)
    if type == 'o':
        abMatrix = np.load('obamaMatrix.npy')
        aUnigramMatrix = np.load('oUnigramMatrix.npy')

        print type
        for i in range(1,6):
            newF = newFeatureSet(abMatrix,aUnigramMatrix,oDict,type,i)
            calAcc(type,i,newF)


calAccG('a')
calAccG('g')
calAccG('m')
calAccG('o')
#calUniAcc()



