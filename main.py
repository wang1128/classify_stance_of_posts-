__author__ = 'penghao'
import numpy as np
#from nltk import bigrams
#from nltk import trigrams
#from nltk import ngrams

def calWordList(type):
    if type == 'a':
        abMatrix = np.load('abMatrix.npy')
    if type == 'g':
        abMatrix = np.load('gayRMatrix.npy')
    if type == 'm':
        abMatrix = np.load('marijuanaMatrix.npy')
    if type == 'o':
        abMatrix = np.load('obamaMatrix.npy')
    wordList =[]
    unigramList = []
    countList =[]
    row = abMatrix.shape[0]

    for i in range(0,row):
        wordList.extend(abMatrix[i][2])
    #for element in wordList:
    #    print (element, wordList.count(element))
    #newcountList = sorted(countList, key=lambda word: word[1])
    #print newcountList
    tokens = [token.lower() for token in wordList if len(token) > 1]# unigram
    setOfword = set(tokens)
    setCopy = list(setOfword)
    for word in setCopy: # get rid of some stop words
        if wordList.count(word)>3000 or wordList.count(word)<3:
            setOfword.remove(word)
    unigramList = list(setOfword)

    return unigramList,abMatrix

def calUniFeature(unigramList,inMatrix):
    col = len(unigramList)
    row = inMatrix.shape[0]
    count = 0
    #featureSetUni = [[0 for col in range(col)] for row in range(row)]
    featureSetUni = np.zeros((row,col))
    #print len(featureSetUni[1914])
    for idx, word in enumerate(unigramList):
        print idx/100
        for i in range(0,row):
            #print i
            for element in inMatrix[i][2]:
                if word.lower() == element.lower():
                    featureSetUni[i][idx] = 1
                    count = count + 1

                    #continue

    return featureSetUni
'''
unigramList,inMatrix = calWordList('a')
featureSetUni = calUniFeature(unigramList,inMatrix)
np.save('aUnigramMatrix',featureSetUni)
print featureSetUni[0]

unigramList,inMatrix = calWordList('g')
featureSetUni = calUniFeature(unigramList,inMatrix)
np.save('gnewUnigramMatrix',featureSetUni)
print featureSetUni[0]

unigramList,inMatrix = calWordList('m')
featureSetUni = calUniFeature(unigramList,inMatrix)
np.save('mnewUnigramMatrix',featureSetUni)
print featureSetUni[0]

unigramList,inMatrix = calWordList('o')
featureSetUni = calUniFeature(unigramList,inMatrix)
np.save('oUnigramMatrix',featureSetUni)
print featureSetUni[0]

'''
