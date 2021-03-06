from getEmbeddings import getEmbeddings
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
#import scikitplot.plotters as skplt
import sklearn.metrics as skm
import pandas as pd
import seaborn as sn

def plot_cmat(yte, ypred):
        #    '''Plotting confusion matrix'''
        #    skplt.plot_confusion_marix(yte,ypred)
        #    plt.show()

    print("Plotting confusion matrix...\n")
    cm = skm.confusion_matrix(yte, ypred, labels=[0,1])
    df_cm = pd.DataFrame(cm, range(2), range(2))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}) # font size
    plt.savefig('cm-nb-anmol.png')

def show_f1(yte, ypred):
    accuracy_score = skm.accuracy_score(yte, ypred)
    f1_all = skm.f1_score(yte, ypred, average=None)
    print("Accuracy is" + str(accuracy_score))
    print("F1 score for real news: "+ str(f1_all[0]))
    print("F1 score for fake news: "+ str(f1_all[1]))
    print(skm.classification_report(yte, ypred, labels=[0,1]))


xtr,xte,ytr,yte = getEmbeddings("datasets/coronaCombinedShuffled.csv")
np.save('./xtr', xtr)
np.save('./xte', xte)
np.save('./ytr', ytr)
np.save('./yte', yte)

xtr = np.load('./xtr.npy')
xte = np.load('./xte.npy')
ytr = np.load('./ytr.npy')
yte = np.load('./yte.npy')

gnb = GaussianNB()
gnb.fit(xtr,ytr)
y_pred = gnb.predict(xte)
m = yte.shape[0]
n = (yte != y_pred).sum()
print("Accuracy = " + format((m-n)/m*100, '.2f') + "%") 

plot_cmat(yte, y_pred)
show_f1(yte, y_pred)
