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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # Serve per non far stampare i "Future Warning", che sono dei warning che avvisano di cambiare delle funzioni sostituendole con altre, perché quelle attualmente utilizzate verranno rimosse in versioni future

scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'f1_score': make_scorer(f1_score)}

header = ['Algorithm', 'FS Method', 'Balancing Method', 'Normalized', 'class', 'accuracy', 'precision', 'recall', 'f1-score', 'support']

algs = ['Knn', 'SVM', 'Decision Tree', 'XGBoost', 'Random Forest']
feature_selection = ['Pearson Coeff', 'Information Gain', 'Fisher Score']
balancing_methods = ['Random Over Sampler', 'Random Under Sampler', 'Not balanced']
normalization_methods = ['Standardization', 'MinMax Scaler', 'Not Normalized']
classification_methods = ['kNeighborsClassifierModel', 'svmLinearModel', 'decisionTreeClassifierModel', 'XGBoostClassifierModel', 'randomForestClassifierModel']
classi = ['0', '1']

# Normalizzazione, standardizzazione, feature_selection

num_class = 2
num_splits = 5

train_index_list = []
test_index_list = []
metrics_mean_for_class = np.zeros((num_class, 4))


df_genes = pd.read_csv(r'C:\Users\giuse\Desktop\Tesi\GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.Amygdala.0.5.filt.transp', delimiter='\t')
df_ages = pd.read_csv(r'C:\Users\giuse\Desktop\Tesi\GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.Amygdala.sampid.age', delimiter='\t')

primary_ages = {'20-29': 'under 50', '30-39': 'under 50', '40-49': 'under 50', '50-59': 'over 50', '60-69': 'over 50', '70-79': 'over 50'} #Dizionario con range età al quale è associata la condizione over/under 50
ages = {'under 50': 0, 'over 50': 1}

df_ages['AGE'] = df_ages['AGE'].map(primary_ages)

df_tot = df_genes.join(df_ages.set_index('SAMPID'), on='SAMPID')

df_tot['AGE'] = df_tot['AGE'].map(ages)

df_tot = df_tot.set_index('SAMPID')  # Imposta come indice (prima colonna) il SAMPID, ed elimina quindi l'indice progressivo (1, 2, 3, ...), eliminando così una colonna. Nel nuovo df_tot sarà inserito df_tot.set_index che è il df_tot con l'indice aggiornato.

index = df_tot.index
head = df_tot.columns

age_col = df_tot['AGE'].to_list()

y = df_tot['AGE']

print(df_tot.shape)

print(df_ages['AGE'].value_counts())

metric_mat = np.zeros((2, 4))

with open('results.csv', mode='w', newline='') as file:  # 'file_name' è 'results.csv'. L'assegnazione viene fatta a riga 37. L'istruzione with permette di chiudere il file automaticamente alla fine del blocco stesso, e ci evita di scrivere il metodo 'file_name.closed()'
    writer = csv.writer(file)  # writer sarà la particolare istanza applicata su 'f' di csv.writer. Rimane quindi una funzione chiamabile
    writer.writerow(header)  # Stiamo dando a writer il comando di scrivere la lista 'header' come prima riga del file 'results.csv'

strfKFold = StratifiedKFold(n_splits=num_splits, shuffle=True)  # 'StratifiedKFold' è la funzione che splitta in 'n_splits' il campione che abbiamo, tra dati di allenamento e dati di test
for train_index, test_index in strfKFold.split(df_tot[df_tot.columns[:18589]], df_tot[df_tot.columns[18589]]):  # train_index, test_index vengono dichiarati direttamente nel "for" e iniziano da 1 entrambi. I dati di training sono dalla colonna 1 alla 18589, mentre quelli di test sono solo la colonna 1859
    train_index_list.append(train_index)  # Inserisce i dati splittati nella lista di training
    test_index_list.append(test_index)  # Inserisce i dati splittati nella lista di test

df_used = pd.DataFrame(normalization.chooseNormalizationMethod(0, df_tot[df_tot.columns[:18589]]))  # L'argomento di pd.DataFrame(arg) è il dataset normalizzato, che quindi viene messo in df_used, considerato che sarà quello utilizzato per i successivi passaggi. Dovrà poi essere bilanciato
df_used['AGE'] = age_col

df_used.set_index(index)  # POTENZIALE ERRORE, come in |df_tot = df_tot.set_index('SAMPID')|, anche in questo caso avrebbe dovuto effetturare l'assegnazione. Senza assegnazione, la modifica all'indice viene persa

# SCELTA ALGORITMO
for a in range(0, len(algs)):

    # FEATURE SELECTION
    for f in range(0, len(feature_selection)):

        rel_features = featureSelection.chooseFeatureSelectionMethod(f, df_used, 'AGE', [0.4, 0.105, 18500])  # Relevant features. choosefeatureselection ritorna il metodo selezionato, il quale sarà fatto partire e ritornerà una lista. La lista che torna sarà assegnata a rel_features, che sarà, ovviamente, di tipo lista. Questa lista conterrà le colonne che sono feature rilevanti

        df_rel_feat_used = df_used[rel_features].to_numpy()  # X è il dataframe in cui sono inserite le feauture rilevanti (secondo il metodo usato) di df_used. La notazione df_used[rel_features] seleziona le colonne corrispondenti a rel_features, che è una lista, quindi verranno messe tutte le colonne corrispondenti ad elementi di quella lista. In generale, df['nome'] permette di accedere alla colonna nome contenuta in df.

        rel_features_df = pd.DataFrame(rel_features, columns=['Id'])  # Inserisce le feature rilevanti in un nuovo dataframe

        rel_features_df.to_csv(feature_selection[f] + '.csv')  # Crea un file csv contenente il dataframe rel_feature_df. Assegna il nome al file concatenando l'elemento k-esimo della lista feature_selection (riga 20) con l'estensione .csv

        # BALANCING
        for b in range(0, len(balancing_methods)):
            X_train, X_test = df_rel_feat_used[train_index_list[b]], df_rel_feat_used[test_index_list[b]]
            y_train, y_test = y[train_index_list[b]], y[test_index_list[b]]

            X_train_bal, y_train_bal = balancing.chooseBalancingMethod(b, X_train, y_train)  # Il metodo a destra dell'assegnazione restituisce 2 dataframe
            print('Bilanciamento Dataset: ', balancing_methods[b], '\n', y_train_bal.value_counts())  # balancing_methods[b] restituisce il nome del metodo di bilanciamento, preso dalla rispettiva lista. Dopodiché, viene mostrato il numero di righe di y_train_bal

            # NORMALIZATION
            for n in range(0, len(normalization_methods)):  # 'normalized' è lista dichiarata a riga 24

                df_used = pd.DataFrame(normalization.chooseNormalizationMethod(n, df_tot[df_tot.columns[:18589]]))  # Normalizza df_tot e lo inserisce in df_used
                df_used['AGE'] = age_col  # Siccome ha normalizzato TUTTE le colonne, compresa quella AGE, con quest'istruzione rimette il valore corretto di età nell'apposita colonna

                df_used.set_index(index)  # Imposta a df_tot.index l'indice di df_used

                fold_count = 1
                alg_mean_accuracy = 0
                alg_mean_precision = 0
                alg_mean_recall = 0
                alg_mean_f1_score = 0

                metrics_mean_for_class = np.zeros((num_class, 4))  # è una matrice di dimensioni numclass * 4

                # CLASSIFICATION (invece di num_splits, sarebbe opportuno utilizzare una variabile classification di dimensione 5)
                for c in range(0, len(classification_methods)):
                    model, y_score = classification.chooseClassificationAlg(a, X_train_bal, X_test, y_train_bal, arguments={'kernel': 'linear', 'eval_metric': 'mlogloss', 'n_estimators': 100})

                    # EVALUATION (è stato messo allo stesso livello di indentazione del blocco precedente perché c'è solo un metodo e non avrebbe alcun senso impostare un ciclo
                    accuracy_tot, precision_tot, recall_tot, f1_score_tot, precision_recall_f1score_support, confusion = evaluation.valutazione(y_test, y_score)
                    print(algs[a] + ' ' + str(fold_count) + '\n')

                    print('Accuracy: ', accuracy_tot, '\n')
                    print('Precision: ', precision_tot, '\n')
                    print('Recall: ', recall_tot, '\n')
                    print('F1 Score: ', f1_score_tot, '\n')

                    print('Confusion Matrix')
                    print(confusion)

                    # ATTENZIONE: tutta questa roba qua sotto è commentata perché è, in un modo o nell'altro, ottenuta con le istruzioni a riga circa 225.
                    alg_mean_accuracy = alg_mean_accuracy + accuracy_tot  # Questo verrà successivamente diviso per num_splits, perché è una media
                    # alg_mean_precision = alg_mean_precision + precision_tot
                    # alg_mean_recall = alg_mean_recall + recall_tot
                    # alg_mean_f1_score = alg_mean_f1_score + f1_score_tot

                    for i in range(0, num_class):
                        count = 0

                        row = [algs[a] + ' ' + str(fold_count), feature_selection[f], balancing_methods[b], normalization_methods[n], classi[i], '']  # L'ultimo elemento sono dei doppi apici, e servono per dire che in accuracy non deve essere inserito niente
                        for metric in precision_recall_f1score_support:  # Per ogni metrica presente in precision_recall_f1score_support (scorre la tupla), aggiungila alla riga (row). Le metriche presenti sono precision, recall, f1score, support.
                            row.append(metric[i])
                            metric_mat[i, count] = metric[i]  # metric_mat è praticamente identico (nel contenuto) a precision_recall_f1score_support. La differenza è nel tipo di dato con cui è rappresentato, precision_recall_f1score_support è una tupla, metric_mat è un array. Forse è stato fatto per una migliore operabilità da parte del programma.

                            metrics_mean_for_class[i, count] = metrics_mean_for_class[i, count] + metric[i]  # è una sommatoria che ad ogni giro aggiunge metric[i]. Alcune misure sono contate num_splits volte, perché sommate. Poi, alla fine del codice, verranno divise per num_splits per ottenere una media

                            count = count + 1

                        with open('results.csv', mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)

                    if c == num_splits - 1:  # Quando p è uguale a 4 (quindi siamo all'ultima iterata) oltre a stampare le combinazioni precedenti, stampa anche un'ulteriore coppia di righe con "mean", invece di fold count
                        with open('results.csv', mode='a', newline='') as file:
                            writer = csv.writer(file)

                            for m in range(0, num_class):
                                writer.writerow([algs[a] + ' mean', feature_selection[f], balancing_methods[b], normalization_methods[n], classi[m], alg_mean_accuracy / num_splits, metrics_mean_for_class[m, 0] / num_splits, metrics_mean_for_class[m, 1] / num_splits, metrics_mean_for_class[m, 2] / num_splits])

                    fold_count = fold_count + 1