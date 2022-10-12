import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
from skfeature.function.similarity_based import fisher_score

'''
Pearson
Il coefficiente di correlazione ha valori compresi tra -1 e 1, Un valore più vicino a 0 implica una correlazione più debole (lo 0 esatto non implica alcuna correlazione)
Un valore più vicino a 1 implica una correlazione positiva più forte
Un valore più vicino a -1 implica una correlazione negativa più forte
'''
def selectFeaturesUsingPearsonCorrCoeff(dataFrame, class_attr, min_value):

    corr = dataFrame.corr() #correlazione

    cor_target = abs(corr[class_attr])

    relevant_features = cor_target[cor_target > min_value] #Le feature rilevanti saranno quelle con maggior correlazione, ovvero quelle maggiori del valore minimo

    print(relevant_features)

    return [relevant_features.axes[0][i] for i in range(1, len(relevant_features))]

'''
Information Gain
Misura la dipendenza tra due variabili.Il punteggio MI (Mutual-Information) cadrà nell'intervallo da 0 a ∞. Maggiore è il valore, più stretta è la connessione tra questa funzionalità e l'obiettivo.
'''

def selectFeaturesUsingInformationGain(dataFrame, class_attr, min_value):
    imp_feat = mutual_info_classif(dataFrame, dataFrame[class_attr])

    relevant_features = pd.Series(imp_feat, dataFrame.columns[:])

    relevant_features = relevant_features[relevant_features > min_value] #Le feature rilevanti saranno quelle con maggior correlazione, ovvero quelle maggiori del valore minimo

    print(relevant_features)

    return [relevant_features.axes[0][i] for i in range(1, len(relevant_features))]

def selectFeaturesUsingFisherScore(dataFrame, class_attr, min_value):
    rel_feature = fisher_score.fisher_score(dataFrame.to_numpy(), dataFrame[class_attr].to_numpy())

    relevant_features = pd.Series(rel_feature, dataFrame.columns[:])

    #print(relevant_features[relevant_features > min_value])

    relevant_features = relevant_features[relevant_features > min_value]

    return [relevant_features.axes[0][i] for i in range(1, len(relevant_features))]

def chooseFeatureSelectionMethod(method, dataFrame, class_attr, min_value_arr):
    switcher = {
        0: selectFeaturesUsingPearsonCorrCoeff(dataFrame, class_attr, min_value_arr[0]),
        1: selectFeaturesUsingInformationGain(dataFrame, class_attr, min_value_arr[1]),
        2: selectFeaturesUsingFisherScore(dataFrame, class_attr, min_value_arr[2])
    }

    return switcher.get(method)