import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import association_rules, apriori
import plotly.express as px
import seaborn as sns

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

#vis1(heatmap) code
heatmap_data1 = rules_s1.pivot(index='antecedents', columns='consequents', values='lift')
fig1, ax1 = plt.subplots()
ax1.pcolor(heatmap_data1)
ax1.set_xlabel('Consequents')
ax1.set_ylabel('Antecedents')
sns.heatmap(heatmap_data1, annot=True, cmap='coolwarm', fmt=".2f", ax=ax1)

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

#Data Visualization for Segment 2
heatmap_data2 = rules_s2.pivot(index='antecedents', columns='consequents', values='lift')
fig2, ax2 = plt.subplots()
ax2.pcolor(heatmap_data2)
ax2.set_xlabel('Consequents')
ax2.set_ylabel('Antecedents')
sns.heatmap(heatmap_data2, annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)

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

#Data Visualization for Segment 3
heatmap_data3 = rules_s3.pivot(index='antecedents', columns='consequents', values='lift')
fig3, ax3 = plt.subplots()
ax3.pcolor(heatmap_data3)
ax3.set_xlabel('Consequents')
ax3.set_ylabel('Antecedents')
sns.heatmap(heatmap_data3, annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)

#add a header
st.header('  Minger Company', divider = 'red')

#pie chart for categories
Category = 'Furniture', 'Office Supplies', 'Technology'
Quantity = [6023, 5968, 6016]
fig_p, ax_p = plt.subplots()
ax_p.set_title('Quantity percentage of each category')
ax_p.pie(Quantity, labels = Category, autopct='%1.1f%%', startangle = 90)
ax_p.axis('equal')
st.pyplot(fig_p)

#Sales for each Segment by Market
#Adding a selectbox
mOptions = ['None','Africa','APAC','Canada','EMEA','EU','LATAM','US']
selectbox = st.selectbox('Select the Market:',options = mOptions)

#Visualizations
if selectbox == 'Africa':
    fig_s1, ax_s1 = plt.subplots()
    data_s1 = df[df['Market'] == 'Africa']
    sns.barplot(data_s1, x='Segment', y='Sales')
    st.pyplot(fig_s1)

if selectbox == 'APAC':
    fig_s2, ax_s2 = plt.subplots()
    data_s2 = df[df['Market'] == 'APAC']
    sns.barplot(data_s2, x='Segment', y='Sales')
    st.pyplot(fig_s2)

if selectbox == 'Canada':
    fig_s3, ax_s3 = plt.subplots()
    data_s3 = df[df['Market'] == 'Canada']
    sns.barplot(data_s3, x='Segment', y='Sales')
    st.pyplot(fig_s3)

if selectbox == 'EMEA':
    fig_s4, ax_s4 = plt.subplots()
    data_s4 = df[df['Market'] == 'EMEA']
    sns.barplot(data_s4, x='Segment', y='Sales')
    st.pyplot(fig_s4)

if selectbox == 'EU':
    fig_s5, ax_s5 = plt.subplots()
    data_s5 = df[df['Market'] == 'EU']
    sns.barplot(data_s5, x='Segment', y='Sales')
    st.pyplot(fig_s5)

if selectbox == 'LATAM':
    fig_s6, ax_s6 = plt.subplots()
    data_s6 = df[df['Market'] == 'LATAM']
    sns.barplot(data_s6, x='Segment', y='Sales')
    st.pyplot(fig_s6)

if selectbox == 'US':
    fig_s7, ax_s7 = plt.subplots()
    data_s7 = df[df['Market'] == 'US']
    sns.barplot(data_s7, x='Segment', y='Sales')
    st.pyplot(fig_s7)

#add subheader
st.subheader('Market Basket Analysis Results: Recommended Items from the dataset')

#Data Visualization for the dataset
heatmap_data = rules.pivot(index='antecedents', columns='consequents', values='lift')
fig, ax = plt.subplots()
ax.pcolor(heatmap_data)
ax.set_xlabel('Consequents')
ax.set_ylabel('Antecedents')
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot(fig)

#selecting Segments
Segment_names = ['Consumer','Home Office','Corporate']
#add radio button
Segment = st.radio('Select Segment:', Segment_names)

#Data visualization for each segment
if 'Consumer' in Segment:
    st.pyplot(fig1)
if 'Home Office' in Segment:
    st.pyplot(fig2)
if 'Corporate' in Segment:
    st.pyplot(fig3)
