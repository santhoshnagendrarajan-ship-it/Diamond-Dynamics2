import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import os
from sklearn.preprocessing import StandardScaler

with open('RandomForestmodel.pkl', "rb") as f:
    rf_model = pickle.load(f)

with open('kmeans_best_model.pkl', "rb") as f:
    kmeans_model = pickle.load(f)

with open('scaler.pkl', "rb") as f:
    scaler = pickle.load(f)
# Constants
INR_RATE = 82.0  # approximate conversion factor (adjust if needed)

st.set_page_config(page_title="Diamond Price & Market Segment Predictor", layout="centered")

st.title("Diamond Price & Market Segment Predictor 💎")
st.markdown("Use the form below to predict a diamond's price (INR) and its market segment (cluster).")



cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order = list("JIHGFED")[::-1]  # -> ['D','E','F','G','H','I','J']
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
carat_order=['Light','Medium','Heavy']
carat = st.number_input("Carat", min_value=0.2, max_value=5.0, value=1.0, step=0.01)
depth = st.number_input("Depth", min_value=50.0, max_value=80.0, value=61.0, step=0.1)
table = st.number_input("Table", min_value=50.0, max_value=100.0, value=57.0, step=0.1)
x = st.number_input("Length (x)", min_value=3.0, max_value=10.0, value=5.0, step=0.01)
y = st.number_input("Width (y)", min_value=3.0, max_value=10.0, value=5.0, step=0.01)
z = st.number_input("Depth (z)", min_value=2.0, max_value=6.0, value=3.0, step=0.01)

color = st.selectbox("Color", color_order)
cut = st.selectbox("Cut", cut_order)
clarity = st.selectbox("Clarity", clarity_order)

    # Ordinal mappings

''''crow = {}
crow['carat'] = carat
    # reasonable defaults for depth/table taken from training df medians
    #row['depth'] = df['depth'].median() if 'depth' in df.columns else 61.0
    #row['table'] = df['table'].median() if 'table' in df.columns else 57.0
crow['depth'] = depth
crow['table'] = table
crow['x'] = x
crow['y'] = y
crow['z'] = z
crow['volume'] = x * y * z
    # price_per_carat unknown at prediction time => set NaN so preprocessor imputes median
crow['price_per_carat'] = np.nan
crow['dimension_ratio'] = (x + y) / (2 * z) if z != 0 else 0.0
    # categorical
crow['cut'] = cut
crow['color'] = color
crow['clarity'] = clarity'''
dimension_ratio = (x + y) / (2 * z) if z != 0 else 0.0
price_per_carat = 1000.0  # placeholder, will be imputed
    # carat category
if carat < 0.5:
    cat = 'Light'
elif carat <= 1.5:
    cat = 'Medium'
else:
    cat = 'Heavy'
#crow['carat_category'] = cat
carat_category = cat
volume = x * y * z
cut_mapping = {v: i for i, v in enumerate(cut_order)}
color_mapping = {v: i for i, v in enumerate(color_order)}
clarity_mapping = {v: i for i, v in enumerate(clarity_order)}
carat_mapping={v: i for i, v in enumerate(carat_order)}
# Apply mapping
'''crow['cut_encoded'] = cut_mapping[crow['cut']]
crow['color_encoded'] = color_mapping[crow['color']]
crow['clarity_encoded'] = clarity_mapping[crow['clarity']]
crow['carat_encoded'] = carat_mapping[crow['carat_category']]'''
cut_encoded = cut_mapping[cut]
color_encoded = color_mapping[color]
clarity_encoded = clarity_mapping[clarity]
carat_encoded = carat_mapping[carat_category]

#df=drop(columns=['price','dimension_ratio','cut', 'color', 'clarity', 'carat_category'])
pred_data = np.array([[carat, depth, table, x, y, z, volume, price_per_carat, cut_encoded,
                            color_encoded, clarity_encoded, carat_encoded]])

#X = pd.DataFrame([crow])
#p_d=X.drop(columns=['cut', 'color', 'clarity', 'carat_category','dimension_ratio'])

if st.button("Predict Price"):
    
    #price_pred = rf_model.predict(X=p_d)[0]
    price_pred = rf_model.predict(pred_data)[0]
    st.success(f"Predicted Price: ₹{price_pred*INR_RATE:,.2f} (approx.)")

#c_d=X.drop(columns=['cut', 'color', 'clarity', 'carat_category', 'price_per_carat'])
cluser_data = np.array([[carat, depth, table, x, y, z, volume, price_per_carat, dimension_ratio,
                          cut_encoded, color_encoded, clarity_encoded, carat_encoded]])
if st.button("cluster"):    
    scaled = scaler.transform(cluser_data)
    labels = kmeans_model.predict(scaled)       

    cluster = labels[0]
    # Map cluster number to descriptive name                    
    c_n_dict ={0:"Premium Heavy Diamonds",1:"Mid-range Balanced Diamonds",2:"Affordable Small Diamonds"
    }
    #summary['n_n']= df_clusters['cluster'].map(c_n_dict)
    #name_for_pred = c_n_dict.get(labels, 'Unknown / Noise')
    # 1) list comprehension (explicit, simple)
    name_for_pred = [c_n_dict.get(int(l), 'Unknown / Noise') for l in labels]
    #name_for_pred = name_map.get(cluster, 'Unknown / Noise')
    st.info(f"Predicted Market Segment: {name_for_pred[0]}")


st.markdown('---')
st.caption('Note: This app uses pre-trained model artifacts expected to be in the project root: `ann_preprocessor.pkl`, `ann_state.pth`, `Kmeans_best_model.pkl`, and `diamonds.csv`. Adjust INR conversion factor as needed.')
