import numpy as np
from scipy import spatial
import openai
import os
import pandas as pd
from openai import AzureOpenAI



client = AzureOpenAI(
    api_key="",
    api_version="",
    azure_endpoint=""
)

# Function to get embeddings
def generate_embeddings(text):
    response = client.embeddings.create(input=[text], model="text-embedding-ada-002-model")
    return response.data[0].embedding
    
# Read CSV data
df = pd.read_csv("campaign_data.csv")

# Concatenate relevant columns into a single text representation
df['Concat'] = (
    df["Campaign ID"].astype(str) + " " +
    df['Campaign Name'].astype(str) + " " +
    df['Impressions'].astype(str) + " " +
    df['Clicks'].astype(str) + " " +
    df['Conversions'].astype(str) + " " +
    df['Spend'].astype(str) + " " +
    df['Revenue'].astype(str) 
)

# Generate embeddings for the 'Concat' column
df['ada_v2'] = df['Concat'].apply(lambda x: generate_embeddings(x))

# Generate embeddings for the 'ProductID' column
df['ada_v1'] = df['Campaign ID'].apply(lambda x: generate_embeddings(x))
# df.to_csv('campaign_data_embedded_4.csv', index=False)
