import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support,accuracy_score

def get_desc(losses,fpr,tpr,thresholds):
    normalDataLoss=losses[0]
    attackDataLoss=losses[1]

    truePositiveRate = []
    falsePositiveRate = []
    threshold = []
    recall = []
    precision = []
    specificity = []
    f1_measure = []
    accuracy = []

    for rate in range(10, 20, 1) :
        truePositiveRate.append(tpr[np.where(tpr>(rate*0.05))[0][0]])
        falsePositiveRate.append(fpr[np.where(tpr>(rate*0.05))[0][0]])
        recall.append(truePositiveRate[-1])
        precision.append((truePositiveRate[-1]*len(attackDataLoss))/(truePositiveRate[-1]*len(attackDataLoss)+falsePositiveRate[-1]*len(normalDataLoss)))
        specificity.append(1-falsePositiveRate[-1])
        f1_measure.append((2*recall[-1]*precision[-1])/(precision[-1]+recall[-1]))
        threshold.append(thresholds[np.where(tpr>(rate*0.05))[0][0]])
        accuracy.append((truePositiveRate[-1]*len(normalDataLoss)+falsePositiveRate[-1]*len(attackDataLoss))/(len(attackDataLoss)+len(normalDataLoss)))
    frames = pd.DataFrame({'true positive rate' : truePositiveRate,
                    'false positive rate' : falsePositiveRate,
                    'recall' : recall,
                    'precision' : precision,
                    'specificity' : specificity,
                    'f1-measure' : f1_measure,
                    'threshold' : threshold,
                    'accuracy' : accuracy})
    return frames

def make_roc(loss,label,ans_label=1,make_desc=False,make_plot=False):
    normalDataLoss=[]
    attackDataLoss=[]

    for i in range(len(loss)):
        if(label[i]==ans_label):
            attackDataLoss.append(loss[i])
        else:
            normalDataLoss.append(loss[i])

    print("Normal data loss(%d): %f" % (len(normalDataLoss), np.average(np.array(normalDataLoss))))
    print("Attack data loss(%d): %f" % (len(attackDataLoss), np.average(np.array(attackDataLoss))))

    allDataLoss = normalDataLoss+attackDataLoss
    print("Sum : ", len(allDataLoss))
    print(len(normalDataLoss))
    allLabel = [0]*len(normalDataLoss)+[1]*len(attackDataLoss)
    allDataLoss=np.array(allDataLoss).flatten()

    auc=metrics.roc_auc_score(np.array(allLabel), np.array(allDataLoss))
    print('AUC Score {}'.format(auc))

    if make_plot:
        fpr, tpr, thresholds = metrics.roc_curve(np.array(allLabel), np.array(allDataLoss), pos_label=1, drop_intermediate=False)
        fig=plt.figure()
        roc=fig.add_subplot(1,1,1)
        lw = 2
        roc.plot(fpr, tpr, color='darkorange', lw=lw, )
        roc.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        roc.set_xlim([0.0, 1.0])
        roc.set_ylim([0.0, 1.05])
        roc.set_xlabel('False Positive Rate')
        roc.set_ylabel('True Positive Rate')
        roc.set_title('ROC-curve')
        roc.legend(loc="lower right")
    else:
        fig=None

    if make_desc:
        desc=get_desc([normalDataLoss,attackDataLoss],fpr,tpr,thresholds)
    else:
        desc=None

    return auc,fig,desc

#APRF
def prf(y_true,y_pred):
    accuracy = accuracy_score(y_true,y_pred)
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(accuracy, precision, recall, f_score))
