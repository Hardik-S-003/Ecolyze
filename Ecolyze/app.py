import streamlit as st
import pandas as pd
from pymongo import MongoClient
from google.cloud import bigquery
from google.oauth2 import service_account
import json
import os

MONGO_URI = st.secrets.get("MONGO_URI", os.getenv("MONGO_URI"))
PROJECT_ID = st.secrets.get("PROJECT_ID", os.getenv("PROJECT_ID"))
DATASET_NAME = "eco_ai_dataset"
TABLE_NAME = "emissions"
ML_MODEL_NAME = "emissions_forecast"

gcp_creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
credentials = service_account.Credentials.from_service_account_info(
    gcp_creds_dict,
    universe_domain="googleapis.com"  
)

bq_client = bigquery.Client(
    credentials=credentials,
    project=PROJECT_ID,
    client_options={"api_endpoint": "https://bigquery.googleapis.com"}  
)

@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv")
    df = df[["country", "year", "co2", "population"]].dropna()
    df = df[df["year"] >= 2000].reset_index(drop=True)
    return df

@st.cache_resource
def push_to_bigquery():
    df = load_data()
    dataset_ref = bigquery.DatasetReference(PROJECT_ID, DATASET_NAME)
    try:
        bq_client.get_dataset(dataset_ref)
    except Exception:
        bq_client.create_dataset(bigquery.Dataset(dataset_ref))
    table_ref = dataset_ref.table(TABLE_NAME)
    job = bq_client.load_table_from_dataframe(df, table_ref)
    job.result()
    return df

def query_summary(year):
    QUERY = f'''
        SELECT country, SUM(co2) AS total_co2
        FROM `{PROJECT_ID}.{DATASET_NAME}.{TABLE_NAME}`
        WHERE year = {year}
        GROUP BY country
        ORDER BY total_co2 DESC
        LIMIT 5
    '''
    return bq_client.query(QUERY).to_dataframe()

def store_to_mongo(df):
    client = MongoClient(MONGO_URI)
    db = client["eco_db"]
    collection = db["emissions_data"]
    collection.delete_many({})
    collection.insert_many(df.to_dict(orient="records"))
    return True

def create_ml_model():
    query = f'''
        CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET_NAME}.{ML_MODEL_NAME}`
        OPTIONS(model_type='linear_reg', input_label_cols=['co2']) AS
        SELECT year, population, co2
        FROM `{PROJECT_ID}.{DATASET_NAME}.{TABLE_NAME}`
        WHERE country = 'India'
    '''
    bq_client.query(query).result()

def predict_co2():
    query = f'''
        SELECT year, predicted_co2
        FROM ML.PREDICT(MODEL `{PROJECT_ID}.{DATASET_NAME}.{ML_MODEL_NAME}`,
            (SELECT year, population FROM `{PROJECT_ID}.{DATASET_NAME}.{TABLE_NAME}` 
             WHERE country='India' AND year >= 2015))
    '''
    return bq_client.query(query).to_dataframe()

# UI
st.title("Ecolyze 🌿")
st.write("Analyze CO₂ emissions with BigQuery ML, MongoDB Atlas & Streamlit")

year = st.selectbox("Choose a year:", list(range(2000, 2023)))

if st.button("Run Analysis"):
    with st.spinner("Loading and analyzing data..."):
        push_to_bigquery()
        summary_df = query_summary(year)
        store_to_mongo(summary_df)
        create_ml_model()
        prediction_df = predict_co2()

        st.success("Data processed and stored successfully!")
        st.subheader(f"Top 5 CO₂ Emitting Countries in {year}")
        st.dataframe(summary_df)
        st.bar_chart(summary_df.set_index("country"))

        st.subheader("Predicted CO₂ for India (from 2015)")
        st.line_chart(prediction_df.set_index("year"))

st.info("Built with Google BigQuery + MongoDB Atlas + Streamlit | Project: Ecolyze")
