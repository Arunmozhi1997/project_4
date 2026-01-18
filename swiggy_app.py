import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, hstack


#  Load Data & Trained Objects


@st.cache_data
def load_data():
    return pd.read_csv("C:/Users/HP/swiggy/cleaned_data.csv")

@st.cache_resource
def load_models():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return scaler, encoder
c_df = load_data()
scaler, encoder = load_models()

num_cols = ["rating", "rating_count", "cost"]
cat_cols = ["city", "cuisine"]

#  Encode Full Dataset 


scaled_num = scaler.transform(c_df[num_cols])
num_sparse = csr_matrix(scaled_num)

cat_sparse = encoder.transform(c_df[cat_cols])

encoded_data = hstack((num_sparse, cat_sparse)).tocsr()

# Streamlit app


st.set_page_config(page_title="Swiggy Restaurant Recommender", layout="wide")

st.title(" Swiggy Restaurant Recommendation System")
st.markdown("Get restaurant recommendations based on **city, cuisine, rating & budget**.")

# Sidebar Inputs
st.sidebar.header(" User Preferences")
city = st.sidebar.selectbox(
    "Select City",
    sorted(c_df["city"].unique())
)
cuisine = st.sidebar.selectbox(
    "Select Cuisine",
    sorted(c_df["cuisine"].unique())
)
rating = st.sidebar.slider(
    "Minimum Rating",
    min_value=1.0,
    max_value=5.0,
    value=4.0,
    step=0.1
)
rating_count = st.sidebar.number_input(
    "Minimum Rating Count",
    min_value=0,
    value=50,
    step=10
)
cost = st.sidebar.slider(
    "Budget (Cost for Two)",
    min_value=int(c_df["cost"].min()),
    max_value=int(c_df["cost"].max()),
    value=300,
    step=50
)
top_n = st.sidebar.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10
)
#  Recommendation Logic


if st.sidebar.button(" Recommend Restaurants"):
        # Create user dataframe
    user_input = {
        "rating": rating,
        "rating_count": rating_count,
        "cost": cost,
        "city": city,
        "cuisine": cuisine
    }
    user_df = pd.DataFrame([user_input])

    # Transform user input
    user_scaled = scaler.transform(user_df[num_cols])
    user_num_sparse = csr_matrix(user_scaled)

    user_cat_encoded = encoder.transform(user_df[cat_cols])

    user_vector = hstack((user_num_sparse, user_cat_encoded)).tocsr()

    # Compute cosine similarity
    similarities = cosine_similarity(user_vector, encoded_data)[0]

    # Get Top-N indices
    top_indices = np.argsort(similarities)[::-1][:top_n]

    # Fetch recommendations
    recommendations = c_df.iloc[top_indices][
        ["name", "city", "rating", "rating_count", "cost", "cuisine"]
    ].reset_index(drop=True)

    # Display Output


    st.subheader(" Recommended Restaurants")    
    st.dataframe(
        recommendations,
        use_container_width=True
    )

    # Rating Visualization
    st.subheader(" Ratings of Recommended Restaurants")
    st.bar_chart(
        recommendations.set_index("name")["rating"]
    )
else:
    st.info(" Select preferences from the sidebar and click **Recommend Restaurants**")