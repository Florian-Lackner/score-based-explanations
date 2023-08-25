import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from math import ceil


def main():
    # DATA = pd.read_csv("src/data/german_credit_data.csv")
    # IGNORE = ["ID", "Risk"]
    # STAY_NUMERIC = ["Job"]
    # CHOSEN = {
    #     'Age': 4,
    #     'Credit amount': 4,
    #     'Duration': 3
    # }
    # DATA = pd.read_csv("src/data/blood-transfusion-service-center.csv")
    # IGNORE = ["Donated"]
    # STAY_NUMERIC = []
    # CHOSEN = {
    #     'Recency': 5,
    #     'Frequency': 4,
    #     'Monetary': 4,
    #     'Time': 5
    # }
    DATA = pd.read_csv("src/data/creditworthiness.csv")
    IGNORE = ['class']
    STAY_NUMERIC = ['installment_commitment', 'residence_since', 'existing_credits', 'num_dependents']
    CHOSEN = {
        'age': 4,
        'credit_amount': 4,
        'duration': 3
    }
    #show_heatmap(DATA)

    #data2 = pd.read_csv("src/data/credit_record.csv")
    #DATA = pd.merge(data, data2, on="ID", how="inner")

    for i,feature in enumerate(DATA.columns.drop(IGNORE).tolist()):
        if feature in CHOSEN:
            print(f"{str(i+1).rjust(3)}) Feature {feature}:")
            print_bins(CHOSEN[feature], DATA[feature])
        elif not pd.api.types.is_numeric_dtype(DATA[feature]) or feature in STAY_NUMERIC:
            print(f"{str(i+1).rjust(3)}) Feature {feature}: {DATA[feature].unique()}")
        else:
            print(f"{str(i+1).rjust(3)}) Feature {feature}:")
            show_histogram(DATA[feature])
        #print(f"'{feature}': {DATA[feature].unique()}")




def show_histogram(data):
    fig = plt.figure(figsize=(16,9))

    fig.add_subplot(4,4,(1,2))
    plt.bar(data.value_counts().index, data.value_counts().values)

    fig.add_subplot(4,4,(3,4))
    plt.hist(data, edgecolor='black')

    for i in range(2,13):
        bins = histedges_equalN(data, i)
        #bins = [edge_case(data, interpolation[i], interpolation[i+1], interpolation[i+2]) for i in range(len(interpolation)-2)]
        #bins[-1] += 1
        if len(set(bins)) == len(bins):
            fig.add_subplot(4,4,i+4)
            print_bins(i, data, bins)


    plt.show()


def histedges_equalN(data, nbin):
    return np.interp(np.linspace(0, len(data), nbin + 1),
                     np.arange(len(data)),
                     np.sort(data))


def print_bins(i, data, bins=None):
    if bins is None:
        bins = histedges_equalN(data, i)
    n, bins, patches = plt.hist(data, edgecolor='black', bins=bins)
    labels = [f"<{int(bins[1])+1}:{len(data[data<bins[1]])}"]+[f"{int(bins[j+1])+1}-{ceil(bins[j+2])}:{len(data[data<bins[j+2]])-len(data[data<bins[j+1]])}" for j in range(len(bins)-3)]+[f">{ceil(bins[-2])}:{len(data[data>=bins[-2]])}"]
    print(f'\t{i}/{len(data)//i}: ({[0]+bins[1:].tolist()}, {labels})')
    #print(f'{i}: ({bins.tolist()}, {labels})')


def show_heatmap(data):
    encoder = OneHotEncoder()
    categorical_columns = list(filter(lambda f: pd.api.types.is_string_dtype(data[f]), data.columns))
    encoded_data = encoder.fit_transform(data[categorical_columns])
    one_hot_data = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(), dtype=bool)
    data = pd.concat([data.drop(columns=categorical_columns), one_hot_data], axis=1)

    corr = data.corr()
    hm = sns.heatmap(corr, annot=True, xticklabels=True, yticklabels=True)



    plt.show()

if __name__ == "__main__":
    main()