import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import association_rules, apriori

st.title('''  Market Basket Analysis Results''')

#load dataset
df = pd.read_excel('cleaned_dataset_global (1).xlsx')

df["single_transaction"] = df["Customer ID"].astype(str)+'_'+df['Order Date'].astype(str)

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
st.write(rules)
###### MBA using Segments to train the dataset
## For Segment 1

s1 = (df[df["Segment"] == "Consumer"]
     .groupby(["Order ID", "Sub-Category"])["Quantity"]
     .sum().unstack().reset_index().fillna(0)
     .set_index("Order ID"))
s1


def encode_units(x):
    if x <=0:
        return 0
    if x >=1:
        return 1

s1_sets = s1.map(encode_units)

frequent_itemsets_s1 = apriori(s1_sets, min_support=0.001, use_colnames = True)
rules_s1 = association_rules(frequent_itemsets_s1, metric = "lift", min_threshold=1)

#Heat map for the whole dataset

st.subheader('''Recommended Items from the dataset''')
heatmap_data = rules.pivot(index='antecedents', columns='consequents', values='lift')
fig, ax = plt.subplots()
ax.pcolor(heatmap_data)
ax.set_xlabel('Consequents')
ax.set_ylabel('Antecedents')
#sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot(fig)

#options of Lift
speed_options = ['1','2','3','4','5','6','7','8','9','10']
Lift = st.select_slider("Lift Threshold", options=speed_options)


Segment = df['Segment'].unique()
selected_column2 = st.selectbox("Select Segment :", Segment)

