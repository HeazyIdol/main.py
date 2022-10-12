from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def valutazione(y_test, y_score):

    accuracy = metrics.accuracy_score(y_test, y_score)
    precision = metrics.precision_score(y_test, y_score)
    recall = metrics.recall_score(y_test, y_score)
    f1_score = metrics.f1_score(y_test, y_score)
    precision_recall_f1score_support = metrics.precision_recall_fscore_support(y_test,
                                                               y_score, average=None, zero_division=1)
    confusion = confusion_matrix(y_true=y_test, y_pred=y_score)

    return accuracy, precision, recall, f1_score, precision_recall_f1score_support, confusion