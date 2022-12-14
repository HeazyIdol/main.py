import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import balancing
import classification
import featureSelection
import normalization
import evaluation
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import csv

scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'f1_score': make_scorer(f1_score)}

header = ['Algorithm', 'FS Method', 'Balancing Method', 'Normalized', 'class', 'accuracy', 'precision', 'recall', 'f1-score', 'support']

feature_selection = ['Pearson Coeff', 'Information Gain', 'Fisher Score']

algs = ['Knn', 'SVM', 'Decision Tree', 'XGBoost', 'Random Forest']

classi = ['0', '1']

balancing_methods = ['Random Over Sampler', 'Random Under Sampler', 'Not balanced']

normalized = ['Standardization', 'MinMax Scaler', 'Not Normalized']

num_class = 2
num_splits = 5

train_index_list = []
test_index_list = []
metrics_mean_for_class = np.zeros((num_class, 4))

file_name = 'results.csv'

df_genes = pd.read_csv('GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.Amygdala.0.5.filt.transp', delimiter='\t')
df_ages = pd.read_csv('GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.Amygdala.sampid.age', delimiter='\t')

df_tot = df_genes.join(df_ages.set_index('SAMPID'), on='SAMPID')
# df.columns = columns
#

primary_ages = {'20-29': 'under 50', '30-39': 'under 50', '40-49': 'under 50', '50-59': 'over 50', '60-69': 'over 50', '70-79': 'over 50'}
ages = {'under 50': 0, 'over 50': 1}

#print(df_ages['AGE'].value_counts())

df_ages['AGE'] = df_ages['AGE'].map(primary_ages)

df_tot = df_genes.join(df_ages.set_index('SAMPID'), on='SAMPID')

df_tot['AGE'] = df_tot['AGE'].map(ages)

df_tot = df_tot.set_index('SAMPID')

index = df_tot.index
head = df_tot.columns

age_col = df_tot['AGE'].to_list()

#print(age_col)

y = df_tot['AGE']

print(df_tot.shape)

#print(df_tot.head())

#print(index)

print(df_ages['AGE'].value_counts())

metric_mat = np.zeros((2, 4))

#d_b = r'\\'

with open(file_name, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

strfKFold = StratifiedKFold(n_splits=num_splits, shuffle=True)
for train_index, test_index in strfKFold.split(df_tot[df_tot.columns[:18589]], df_tot[df_tot.columns[18589]]):
    train_index_list.append(train_index)
    test_index_list.append(test_index)

df_used = pd.DataFrame(normalization.chooseNormalizationMethod(0, df_tot[df_tot.columns[:18589]]))
df_used['AGE'] = age_col

df_used.set_index(index)

for c in range(0, len(algs)):
    b4 = False

    for k in range(0, len(feature_selection)):
        b3 = False

        rel_features = featureSelection.chooseFeatureSelectionMethod(k, df_used, 'AGE', [0.4, 0.105, 18500])

        # X = df_used[rel_features]

        X = df_used[rel_features].to_numpy()

        rel_features_df = pd.DataFrame(rel_features, columns=['Id'])

        rel_features_df.to_csv(feature_selection[k] + '.csv')

        for b in range(0, len(balancing_methods)):
            b2 = False

            #print(feature_selection[k] + '\n')

            for n in range(0, len(normalized)):
                b1 = False

                if n == 2:
                    df_used = df_tot
                else:
                    df_used = pd.DataFrame(normalization.chooseNormalizationMethod(n, df_tot[df_tot.columns[:18589]]))
                    df_used['AGE'] = age_col

                    df_used.set_index(index)

                fold_count = 1
                alg_mean_accuracy = 0
                alg_mean_precision = 0
                alg_mean_recall = 0
                alg_mean_f1_score = 0

                metrics_mean_for_class = np.zeros((num_class, 4))

                for p in range(0, num_splits):
                    X_train, X_test = X[train_index_list[p]], X[test_index_list[p]]
                    y_train, y_test = y[train_index_list[p]], y[test_index_list[p]]

                    if b == 2:
                        X_train_bal, y_train_bal = X_train, y_train
                        print('Dataset non bilanciato\n', y_train_bal.value_counts())
                    else:
                        X_train_bal, y_train_bal = balancing.chooseBalancingMethod(b, X_train, y_train)
                        print('Dataset bilanciato con', balancing_methods[b], '\n', y_train_bal.value_counts())

                    model, y_score = classification.chooseClassificationAlg(c, X_train_bal, X_test, y_train_bal,
                                                                            arguments={'kernel': 'linear', 'eval_metric': 'mlogloss', 'n_estimators': 100})

                    accuracy_tot, precision_tot, recall_tot, f1_score_tot, precision_recall_f1score_support, confusion = evaluation.valutazione(y_test, y_score)
                    print(algs[c] + ' ' + str(fold_count) + '\n')

                    print('Accuracy: ', accuracy_tot, '\n')
                    print('Precision: ', precision_tot, '\n')
                    print('Recall: ', recall_tot, '\n')
                    print('F1 Score: ', f1_score_tot, '\n')

                    print('Confusion Matrix')
                    print(confusion)

                    alg_mean_accuracy = alg_mean_accuracy + accuracy_tot
                    # alg_mean_precision = alg_mean_precision + precision_tot
                    # alg_mean_recall = alg_mean_recall + recall_tot
                    # alg_mean_f1_score = alg_mean_f1_score + f1_score_tot

                    for i in range(0, num_class):
                        count = 0

                        #print(classification_report(y_test, y_score, labels=[i], zero_division=1))

                        row = [algs[c] + ' ' + str(fold_count), feature_selection[k], balancing_methods[b], normalized[n], classi[i], '']
                        for metric in precision_recall_f1score_support:
                            row.append(metric[i])
                            metric_mat[i, count] = metric[i]

                            metrics_mean_for_class[i, count] = metrics_mean_for_class[i, count] + metric[i]

                            count = count + 1

                        with open(file_name, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)

                    # strn = ""
                    #
                    # if not b4:
                    #     strn = strn + algs[c] + "                 & "
                    #     b4 = True
                    # else:
                    #     strn = strn + "                    & "
                    #
                    # if not b3:
                    #     strn = strn + feature_selection[k] + "    & "
                    #     b3 = True
                    # else:
                    #     strn = strn + "                 & "
                    #
                    # if not b2:
                    #     strn = strn + balancing_methods[b] + "  & "
                    #     b2 = True
                    # else:
                    #     strn = strn + "                     & "
                    #
                    # if not b1:
                    #     strn = strn + normalized[n] + " & "
                    #     b1 = True
                    # else:
                    #     strn = strn + "                & "
                    #
                    # strn = strn + str(fold_count) + " &          "
                    #
                    #
                    # for a in range(0, 4):
                    #     strn = strn + "& "
                    #     strn = strn + str(metric_mat[0, a]) + "      & "
                    #     strn = strn + str(metric_mat[1, a]) + "                "
                    #
                    # strn = strn + "     " + d_b

                    if p == num_splits - 1:
                        with open(file_name, mode='a', newline='') as file:
                            writer = csv.writer(file)

                            for m in range(0, num_class):
                                writer.writerow([algs[c] + ' mean', feature_selection[k], balancing_methods[b], normalized[n], classi[m], alg_mean_accuracy/num_splits, metrics_mean_for_class[m, 0] / num_splits, metrics_mean_for_class[m, 1] / num_splits,
                                                metrics_mean_for_class[m, 2] / num_splits])

                    #print(strn)

                    # if p == num_splits - 1:
                    #     print("                    &                  &                      &                 & media & " + str(alg_mean_accuracy/num_splits) + "         & " +
                    #             str(metrics_mean_for_class[0, 0] / num_splits) + "      & " + str(metrics_mean_for_class[1, 0] / num_splits) + "                & " +
                    #             str(metrics_mean_for_class[0, 1] / num_splits) + "      & " + str(metrics_mean_for_class[1, 1] / num_splits) + "                & " +
                    #             str(metrics_mean_for_class[0, 2] / num_splits) + "      & " + str(metrics_mean_for_class[1, 2] / num_splits) + "                & " +
                    #             str(metrics_mean_for_class[0, 3] / num_splits) + "      & " + str(metrics_mean_for_class[1, 3] / num_splits) + "                     " + d_b)

                    fold_count = fold_count + 1