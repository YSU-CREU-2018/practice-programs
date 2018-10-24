import numpy as np
import pandas as pd
from apyori import apriori


class Main:

    store_data = pd.read_csv("store_data.csv", header=None)
    store_data.head()
    print(len(store_data))

    records = []
    n = len(store_data)
    for i in range(0, n):
        for j in range(0, 20):
            records.append(
                [
                    str(
                        store_data.values[i, j]
                    )
                ]
            )
    print(records)
    # records = []
    # for i in range(0, 7501):
    #     records.append([str(store_data.values[i,j]) for j in range(0, 20)])

    association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)

    association_results = list(association_rules)

    print(len(association_results))
