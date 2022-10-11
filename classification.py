from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def kNeighborsClassifierModel(X_train, X_test, y_train):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_score = model.predict(X_test)

    return model, y_score

def svmLinearModel(X_train, X_test, y_train, kern):
    model = svm.SVC(kernel=kern)
    model.fit(X_train, y_train)
    y_score = model.predict(X_test)

    return model, y_score

def decisionTreeClassifierModel(X_train, X_test, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_score = model.predict(X_test)

    return model, y_score

def XGBoostClassifierModel(X_train, X_test, y_train, eval_metric):
    model = XGBClassifier(use_label_encoder=False, eval_metric=eval_metric)
    model.fit(X_train, y_train)
    y_score = model.predict(X_test)

    return model, y_score

def randomForestClassifierModel(X_train, X_test, y_train, n_estimators):
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    y_score = model.predict(X_test)

    return model, y_score

def chooseClassificationAlg(alg, X_train, X_test, y_train, arguments):
    switcher = {
        0: kNeighborsClassifierModel(X_train, X_test, y_train),
        1: svmLinearModel(X_train, X_test, y_train, arguments['kernel']),
        2: decisionTreeClassifierModel(X_train, X_test, y_train),
        3: XGBoostClassifierModel(X_train, X_test, y_train, arguments['eval_metric']),
        4: randomForestClassifierModel(X_train, X_test, y_train, arguments['n_estimators'])
    }

    return switcher.get(alg)