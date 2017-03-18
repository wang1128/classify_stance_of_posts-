__author__ = 'penghao'

import numpy as np
import glob

def readfile(type,path):
    d = '*.data'
    m = '*.meta'
    datapath = path + d
    metapath = path + m
    datafile = glob.glob(datapath)
    metafile = glob.glob(metapath)

    row = len(datafile)
    print row
    fileNameList = []
    docMatrix = []
    IDList = []
    PIDList = []
    StanceList = []
    rebuttalList = []
    abMatrix = [[0 for col in range(7)] for row in range(row)]

    for i in range(row+1):
        fileName = datafile[i].replace(path, "")
        fileNum = fileName.replace(".data", "")
        metaName = metafile[i].replace(path, "")
        metaNum = metaName.replace(".meta", "")

        infile = np.loadtxt(datafile[i], dtype = str)
        metafileload = np.loadtxt(metafile[i], dtype = str)


        docMatrix.append(infile.tolist())
        fileNameList.append(fileNum)
        IDList.append(metafileload[0].replace("ID=", ""))
        PIDList.append(metafileload[1].replace("PID=", ""))
        StanceList.append(metafileload[2].replace("Stance=", ""))
        rebuttalList.append(metafileload[3].replace("rebuttal=", ""))


        abMatrix[i][0] = type # type,file,doc,Id,Pid,stance,rebuttal
        abMatrix[i][1] = fileNameList[i]
        abMatrix[i][2] = docMatrix[i]
        abMatrix[i][3] = IDList[i]
        abMatrix[i][4] = PIDList[i]
        abMatrix[i][5] = StanceList[i]
        abMatrix[i][6] = rebuttalList[i]
    return abMatrix

abMatrix = readfile('abortion','/Users/penghao/GitHub/ML_Stance/stance/abortion/')
gayRMatrix = readfile('gayRights','/Users/penghao/GitHub/ML_Stance/stance/gayRights/')
marijuanaMatrix = readfile('marijuana','/Users/penghao/GitHub/ML_Stance/stance/marijuana/')
obamaMatrix = readfile('obama','/Users/penghao/GitHub/ML_Stance/stance/obama/')

a = np.array(abMatrix, dtype=object) # save list to np array
g = np.array(gayRMatrix, dtype=object)
m = np.array(marijuanaMatrix, dtype=object)
o = np.array(obamaMatrix, dtype=object)

np.save('abMatrix',a)
np.save('gayRMatrix',g)
np.save('marijuanaMatrix',m)
np.save('obamaMatrix',o)

print m[625]
