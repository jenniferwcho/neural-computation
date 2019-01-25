import os
import scipy.io as spio
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from numpy import *
from scipy.io import loadmat
from matplotlib.backends.backend_pdf import PdfPages as pdf
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model

#load data
matData = loadmat('data_v1_binned_static.mat')

print('Number of Sessions in this Study:', matData['data'].shape[0])


# for x in range(0,17):
#     print('Number of Trials in Session', x+1,':',matData['data'][x][0][0][0][5].shape[3])
#     print('Number of Conditions in Session', x+1,':',matData['data'][x][0][0][0][5].shape[1])
#     print('')

#plot spike counts for all trials
# for x in range(0,17):
#     sess = matData['data'][x][0][0][0][5]
#     spikeCount = np.sum(sess,axis=2)
#     sc = np.reshape(spikeCount,[spikeCount.shape[0],spikeCount.shape[1]*spikeCount.shape[2]])
#     print('Total Number of Trials in Session',x+1,':',sc.shape[1])
#     spikeCount5 = sc[4,:]
#     spikeCount16 = sc[15,:]
#     pl.scatter(spikeCount5, spikeCount16, color= 'b')
#     pl.xlabel('Neuron 5')
#     pl.ylabel('Neuron 16')
#     pl.title('Spike Counts of All Trials for Neuron 5 and 16')
#     pl.show()

# #training/test matrices per session (#17 sessions)
# for x in range(0,17):
#     sess = matData['data'][x][0][0][0][5]
#     spikeCount = np.sum(sess,axis=2)
#     sc = np.reshape(spikeCount,[spikeCount.shape[0],spikeCount.shape[1]*spikeCount.shape[2]])
#     #print('number of trials in session',x+1,':', sc.shape[1])
#     #split data (train and test) 80 to 20 percent ratio
#     trainSpike, testSpike = train_test_split(sc,test_size=0.2)
#     #for contrast, separate into test and train data (80 to 20 ratio)
#     sessionData = matData['data'][x][0][0][0]
#     binnedSpikes = sessionData[5]
#     numCells, numConditions, numTimeBins, numTrialsPerCond = binnedSpikes.shape
#     conditions = sessionData[2][0]
#     trainCon, testCon = train_test_split(conditions,test_size=0.2)
#
#     logreg = linear_model.LogisticRegression()
#     logreg.fit(trainSpike, trainCon)
#     predicted_test_contrast = Z = logreg.predict(testSpike)
#     performance = np.sum(predicted_test_contrast == test_contrast)
#

dimRedMethod ="fa"
for sessionIdx in range(17):
    sess = matData['data'][x][0][0][0][5]
    spikeCount = np.sum(sess,axis=2)
    sc = np.reshape(spikeCount,[spikeCount.shape[0],spikeCount.shape[1]*spikeCount.shape[2]])
    #print('number of trials in session',x+1,':', sc.shape[1])


    sessionData = matData['data'][sessionIdx][0][0][0]
    binnedSpikes = sessionData[5]
    numCells, numConditions, numTimeBins, numTrialsPerCond = binnedSpikes.shape
    conditions = sessionData[2][0]

    #check the order...first 5(contrast,orientation)
    #contrast is never 0, only orientation is

    if conditions[0][0][0][0] == 0:
        contrastIdx = 1
    else:
        contrastIdx = 0


    contrastLevels = np.zeros(numConditions)
    for actCond in range(numConditions):
         contrastLevels[actCond] = conditions[actCond][0][0][0]

    lowContrastBinnedSpikes = binnedSpikes[:, contrastLevels == min(contrastLevels), :, :]
    highContrastBinnedSpikes = binnedSpikes[:, contrastLevels == max(contrastLevels), :, :]
    lowContrastSpikeCounts = np.sum(lowContrastBinnedSpikes, axis=2)
    highContrastSpikeCounts = np.sum(highContrastBinnedSpikes, axis=2)
     # reshape & plot the data here. if you want to plot more than one session at once, use either separate figure objects, subplots, or save into files
    lcSpike = np.reshape(lowContrastSpikeCounts,[lowContrastSpikeCounts.shape[0],lowContrastSpikeCounts.shape[1]*lowContrastSpikeCounts.shape[2]])
    hcSpike = np.reshape(highContrastSpikeCounts, [highContrastSpikeCounts.shape[0], highContrastSpikeCounts.shape[1]*highContrastSpikeCounts.shape[2]])


    #print(lcSpike.shape[1])
    #print(hcSpike.shape[1])

    #matrix 'A'=lcSpike
    #matrix 'B'=hcSpike
    concatenatedSC = np.concatenate((lcSpike,hcSpike))


    #apply PCA
    pca = PCA(n_components = 3)
    concComponents = pca.fit_transform(concatenatedSC.T).T

    #factor analysis
    #fAnalysis = FactorAnalysis(n_components=3)
    #concComponents = fAnalysis.fit_transform(concatenatedSC.T).T

    #LogisticRegression

    #separate the concComponent into lcComp and hcComp
    nLc = lcSpike.shape[0]
    nHc = hcSpike.shape[0]

    lcComp=concComponents[:, 0:nLc]
    hcComp=concComponents[:, nLc+1:nLc+nHc]

    f, axarr = pl.subplots(2, 2)
    x=axarr[0,0].scatter(lcComp[0], lcComp[1], color='b')
    y=axarr[0,0].scatter(hcComp[0], hcComp[1], color='m')
    axarr[0,0].set_xlabel('component1')
    axarr[0,0].set_ylabel('component2')
    axarr[0,0].legend((x,y), ('Low Contrast', 'High Contrast'), scatterpoints=1, loc='lower right', ncol=1,fontsize=4)
    x=axarr[0,1].scatter(lcComp[1], lcComp[2], color='b')
    y=axarr[0,1].scatter(hcComp[1], hcComp[2], color='m')
    axarr[0,1].set_xlabel('component2')
    axarr[0,1].set_ylabel('component3')
    axarr[0,1].legend((x,y), ('Low Contrast', 'High Contrast'), scatterpoints=1, loc='lower right', ncol=1,fontsize=4)
    x=axarr[1,0].scatter(lcComp[0], lcComp[2], color='b')
    y=axarr[1,0].scatter(hcComp[0], hcComp[2], color='m')
    axarr[1,0].set_xlabel('component1')
    axarr[1,0].set_ylabel('component3')
    axarr[1,0].legend((x,y), ('Low Contrast', 'High Contrast'), scatterpoints=1, loc='lower right', ncol=1,fontsize=4)
    pdf.savefig('irp\\irpFig_sess%d_%s.pdf' % (sessionIdx,dimRedMethod))

    # x = pl.scatter(lcComp[0], lcComp[1], color='b')
    # y = pl.scatter(hcComp[0],hcComp[1],color='m')
    # pl.xlabel('component1')
    # pl.ylabel('component2')
    # pl.legend((x,y), ('Low Contrast', 'High Contrast'), scatterpoints=1, loc='lower right', ncol=1,fontsize=8)
    # pdf.savefig('irp\\irpFig_sess%d.pdf' % sessionIdx)
    # pl.close()


    # pl.scatter(lcSpike, hcSpike, color='b')
    # pl.xlabel('low contrast spikes')
    # pl.ylabel('high contrast spikes')
    # pdf.savefig('irp\\irpFig_sess%d.pdf' % sessionIdx)
    # pl.close()
