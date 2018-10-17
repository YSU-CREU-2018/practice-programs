# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:20:20 2018

@author: xlu
"""

"""
Spyder Editor

This is for the Alice example in the proposal.
"""

import numpy as np  
import pandas as pd  

from apyori import apriori

from mlxtend.frequent_patterns import apriori

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#input data
a = [[1,0,0,0,0],
     [1,0,0,0,1],
     [1,0,0,1,1],
     [0,1,0,0,1],
     [0,1,0,1,0]]


ary=np.array(a,dtype=bool)
df = pd.DataFrame(ary, columns=('Deal 1', 'Deal 2', 'Deal 3','Deal 4','Deal 5'))
#generate frequent itemsets
result=apriori(df,min_support=.01,use_colnames=True)

#association rules
from mlxtend.frequent_patterns import association_rules
results=association_rules(result, metric="confidence", min_threshold=0.30)

df_results=pd.DataFrame(results)


newdf=df_results[['antecedents', 'consequents','support','confidence', 'lift']]
newdf.to_csv("appriori_results.csv") #Save to csv formart for detailed view
#print(newdf.head())


#recommendation for Alice
A_results=newdf[newdf['antecedents'] == {'Deal 1'}]

