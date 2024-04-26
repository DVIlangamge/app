import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import association_rules, apriori
import plotly.express as px
import seaborn as sns

st.title('''  Market Basket Analysis Results''')

#load dataset
df = pd.read_excel('cleaned_dataset_global (1).xlsx')

df["single_transaction"] = df["Customer ID"].astype(str)+'_'+df['Order Date'].astype(str) #do not display

df1 = pd.crosstab(df['single_transaction'],df['Sub-Category'])

## MBA for whole dataset

# Encoding data 
def encode(item_freq):
    res = 0
    if item_freq > 0:
        res = 1
    return res
    
basket_input = df1.map(encode)

frequent_itemsets = apriori(basket_input, min_support = 0.001, use_colnames=True)

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold=1)

###### MBA using Segments to train the dataset
## For Segment 1

s1 = (df[df["Segment"] == "Consumer"]
     .groupby(["Order ID", "Sub-Category"])["Quantity"]
     .sum().unstack().reset_index().fillna(0)
     .set_index("Order ID"))

def encode_units(x):
    if x <=0:
        return 0
    if x >=1:
        return 1

s1_sets = s1.map(encode_units)

frequent_itemsets_s1 = apriori(s1_sets, min_support=0.001, use_colnames = True)
rules_s1 = association_rules(frequent_itemsets_s1, metric = "lift", min_threshold=1)

## For Segment 2
s2 = (df[df["Segment"] == "Home Office"]
     .groupby(["Order ID", "Sub-Category"])["Quantity"]
     .sum().unstack().reset_index().fillna(0)
     .set_index("Order ID"))

def encode_units(x):
    if x <=0:
        return 0
    if x >=1:
        return 1

s2_sets = s2.applymap(encode_units)

frequent_itemsets_s2 = apriori(s2_sets, min_support=0.001, use_colnames = True)
rules_s2 = association_rules(frequent_itemsets_s2, metric = "lift", min_threshold=1)

## For Segment 3

s3 = (df[df["Segment"] == "Corporate"]
     .groupby(["Order ID", "Sub-Category"])["Quantity"]
     .sum().unstack().reset_index().fillna(0)
     .set_index("Order ID"))


def encode_units(x):
    if x <=0:
        return 0
    if x >=1:
        return 1

s3_sets = s3.applymap(encode_units)

frequent_itemsets_s3 = apriori(s3_sets, min_support=0.001, use_colnames = True)
rules_s3 = association_rules(frequent_itemsets_s3, metric = "lift", min_threshold=1)


