from getEmbeddings import getEmbeddings
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
#import scikitplot.plotters as skplt
import sklearn.metrics as skm

def plot_cmat(clf, xte, yte):
    skm.plot_confusion_matrix(clf, xte, yte, labels=[0,1], display_labels=['Real', 'Fake'])
    plt.savefig('svm-cf-anmol.png')
    
def show_f1(yte, ypred):
    f1_all = skm.f1_score(yte, ypred, average=None)
    print("F1 score for real news: "+ str(f1_all[0]))
    print("F1 score for fake news: "+ str(f1_all[1]))
    print(skm.classification_report(yte, ypred, labels = [0,1]))
    
xtr,xte,ytr,yte = getEmbeddings("datasets/coronaCombinedShuffled.csv")
np.save('./xtr', xtr)
np.save('./xte', xte)
np.save('./ytr', ytr)
np.save('./yte', yte)

xtr = np.load('./xtr.npy')
xte = np.load('./xte.npy')
ytr = np.load('./ytr.npy')
yte = np.load('./yte.npy')

clf = SVC()
clf.fit(xtr, ytr)
y_pred = clf.predict(xte)
m = yte.shape[0]
n = (yte != y_pred).sum()
#print("Accuracy = " + format((m-n)/m*100, '.2f') + "%") 

plot_cmat(clf, xte, yte)
show_f1(yte, y_pred)
