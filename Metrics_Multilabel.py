from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import label_ranking_average_precision_score, precision_score, recall_score, hamming_loss
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, label_ranking_loss,fbeta_score

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """    label_ranking=label_ranking_loss(Test_label, Prob_label)

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()





def Accuracy_Multilabel(y_true,y_pred):
        TP1=0
        FN1=0  #missed
        FP1=0  #false detection
        TN1=0  # 0 detected as 0
        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[1]):
                if (y_pred[i,j]==1) & (y_true[i,j]==1):
                    TP1=TP1+1
                elif (y_pred[i, j] == 0) & (y_true[i, j] == 1):
                    FN1 = FN1 + 1

                elif (y_pred[i, j] == 1) & (y_true[i, j] == 0):
                    FP1 = FP1 + 1

                elif (y_pred[i, j] == 0) & (y_true[i, j] == 0):
                    TN1 = TN1 + 1

        Sen=TP1/(TP1+FN1)       ### Recall
        Spe=TN1/(TN1+FP1)       ### Spec


        return(Sen,Spe)



def metrics_function(Test_label, Prob_label):
    #print(class_names)
    [Se_entropy, Sp_entropy] = Accuracy_Multilabel(Test_label, Prob_label)
    mAP = average_precision_score(Test_label, Prob_label, average='micro')
    prec_score = precision_score(Test_label, Prob_label, average='micro')
    rec_score = recall_score(Test_label, Prob_label, average='micro')
    ham = hamming_loss(Test_label, Prob_label, sample_weight=None)
    f1score=f1_score(Test_label, Prob_label, average='micro')
    f2score = fbeta_score(Test_label, Prob_label, beta=0.5,average='micro')
    label_ranking=label_ranking_loss(Test_label, Prob_label)
    print("Results on test...............")
    print("Recall:", Se_entropy)
    print("Specifity:", Sp_entropy)
    print('Average:', (Se_entropy + Sp_entropy) / 2)
    print('Average precision mAP:', mAP)
    print('Ham loss:', ham)
    print('Precsion:', prec_score)
    print('recall score:', rec_score)
    print('f1score:', f1score)
    print('f2score:', f2score)
    print("label_ranking:",label_ranking)

    #print('label ranking loss',label_ranking)
    # Compute confusion matrix

