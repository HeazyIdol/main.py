from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# def balanceDataset(X_train, y_train):
#     print(y_train.value_counts())
#
#     over_sampler = RandomOverSampler(random_state=42)
#     X_bal_over, y_bal_over = over_sampler.fit_resample(X_train, y_train)
#
#     print(y_bal_over.value_counts())
#
#     under_sampler = RandomUnderSampler(random_state=42)
#     X_bal_under, y_bal_under = under_sampler.fit_resample(X_train, y_train)
#
#     print(y_bal_under.value_counts())
#
#     return

def balanceDataSetWithOverSampler(X_train, y_train):
    return RandomOverSampler(random_state=42).fit_resample(X_train, y_train)

def balanceDataSetWithUnderSampler(X_train, y_train):
    return RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)

def chooseBalancingMethod(method, X_train, y_train):
    switcher = {
        0: balanceDataSetWithOverSampler(X_train, y_train),
        1: balanceDataSetWithUnderSampler(X_train, y_train)
    }

    return switcher.get(method)