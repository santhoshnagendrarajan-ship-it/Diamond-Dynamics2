import streamlit as st
import numpy as np
import pickle

with open('gf.pkl', "rb") as f:
 gf = pickle.load(f)

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
x = st.number_input("Length (x)", min_value=3.0, max_value=10.0, value=5.0, step=0.01)
y = st.number_input("Width (y)", min_value=3.0, max_value=10.0, value=5.0, step=0.01)
z = st.number_input("Depth (z)", min_value=2.0, max_value=6.0, value=3.0, step=0.01)

color = st.selectbox("Color", color_order)
cut = st.selectbox("Cut", cut_order)
clarity = st.selectbox("Clarity", clarity_order)


dimension_ratio = (x + y) / (2 * z) if z != 0 else 0.0
price_per_carat = 8.0  # placeholder, will be imputed
    # carat category
if carat < 0.5:
    cat = 'Light'
elif carat <= 1.5:
    cat = 'Medium'
else:
    cat = 'Heavy'

carat_category = cat
volume = x * y * z
cut_mapping = {v: i for i, v in enumerate(cut_order)}
color_mapping = {v: i for i, v in enumerate(color_order)}
clarity_mapping = {v: i for i, v in enumerate(clarity_order)}
carat_mapping={v: i for i, v in enumerate(carat_order)}
# Apply mapping

cut_encoded = cut_mapping[cut]
color_encoded = color_mapping[color]
clarity_encoded = clarity_mapping[clarity]
carat_encoded = carat_mapping[carat_category]


pred_data =np.array([[np.log1p(volume), price_per_carat,  np.log1p(carat), carat_encoded
                            ,color_encoded, clarity_encoded]])



if st.button("Predict Price"):
    

    price_pred = gf.predict(pred_data)[0]
    st.session_state["pred_price"] = np.expm1(price_pred)
    st.success(f"Predicted Price: ₹{np.expm1(price_pred)*INR_RATE:,.2f} (approx.)")


if st.button("Cluster"):

    if "pred_price" not in st.session_state:
        st.warning("Please predict the price first.")
    else:

        cluser_data = np.array([[volume, st.session_state["pred_price"], price_per_carat, carat, carat_encoded,
       color_encoded, clarity_encoded]])
   
    scaled = scaler.transform(cluser_data)
    labels = kmeans_model.predict(scaled)       

    cluster = labels[0]
    # Map cluster number to descriptive name                    
    c_n_dict ={0:"Premium Heavy Diamonds",1:"Mid-range Balanced Diamonds",2:"Affordable Small Diamonds"
    }

    name_for_pred = [c_n_dict.get(int(l), 'Unknown / Noise') for l in labels]

    st.info(f"Predicted Market Segment: {name_for_pred[0]}")


st.markdown('---')
