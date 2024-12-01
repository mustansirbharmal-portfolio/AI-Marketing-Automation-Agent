import openai  # Ensure you have installed the `openai` package
from flask import Flask, render_template, jsonify, request, session
import pandas as pd
import rules
from openai import AzureOpenAI
from datetime import datetime
import tiktoken
import re
from scipy.spatial import distance
from flask_session import Session
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

client = AzureOpenAI(
    api_key="",
    api_version="",
    azure_endpoint=""
)

# Load CSV data
csv_file = "campaign_data.csv"  # Path to your CSV file
df = pd.read_csv(csv_file)
csv_file2 = "campaign_data_embedded_4.csv"
df2 = pd.read_csv(csv_file2)

@app.route('/')
def home():
    # Convert DataFrame to a list of dictionaries
    # orient="records" means each row of the DataFrame is converted into a dictionary, 
    # where the column names are the keys, and the cell values are the dictionary values.
    campaigns = df.to_dict(orient="records")     # Convert DataFrame to a list of dictionaries

    # Generate optimization actions (replace `rules.get_campaign_optimization_actions` if needed)
    actions = rules.get_campaign_optimization_actions(campaigns)


    # The zip() function is used to pair up each element of the campaigns list with the corresponding element from the actions list.
    campaign_data = zip(campaigns, actions) # Combine campaigns and actions

    return render_template('index.html', campaign_data=campaign_data)

  
@app.route('/api/generate-insights/<campaign_id>', methods=['POST'])
def generate_insights(campaign_id):
    # Find the campaign in the DataFrame
    campaign = df[df["Campaign ID"].astype(str) == str(campaign_id)].to_dict(orient="records")
    if not campaign:
        return jsonify({"error": "Campaign not found"}), 404
    campaign = campaign[0]

    # Safeguard calculations to prevent division errors
    ctr = (campaign['Clicks'] / campaign['Impressions']) * 100 if campaign['Impressions'] > 0 else 0
    roas = (campaign['Revenue'] / campaign['Spend']) if campaign['Spend'] > 0 else 0
    cpa = (campaign['Spend'] / campaign['Conversions']) if campaign['Conversions'] > 0 else 0

    # Prepare input data for the ChatGPT API
    input_data = (
        f"Campaign Name: {campaign['Campaign Name']}\n"
        f"CTR: {ctr:.2f}%\n"
        f"ROAS: {roas:.2f}x\n"
        f"CPA: {cpa:.2f}\n"
        f"Conversions: {campaign['Conversions']}\n"
        f"Spend: {campaign['Spend']}\n"
    )

    # Call the OpenAI ChatCompletion API
    try:
        messages = [
            {"role": "system", "content": "You are a customer assistant that provides insights in clear and well-structured paragraphs based on the given campaign data."},
            {"role": "user", "content": input_data},
        ]
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0
        )

        # Retrieve and process the response
        response_message = response.choices[0].message.content.strip()

        # Split the response into meaningful paragraphs
        formatted_response = response_message.split('\n\n')

        return jsonify({"insights": formatted_response})

    except Exception as e:
        return jsonify({"error": f"Internal server error: {e}"}), 500

    
@app.route('/powered_insights/<campaign_id>', methods=['GET'])
def powered_insights(campaign_id):
    # Find the campaign in the DataFrame
    campaign = df[df["Campaign ID"].astype(str) == str(campaign_id)].to_dict(orient="records")
    if not campaign:
        return "Campaign not found", 404
    campaign = campaign[0]

    # Safeguard calculations to prevent division errors
    ctr = (campaign['Clicks'] / campaign['Impressions']) * 100 if campaign['Impressions'] > 0 else 0
    roas = (campaign['Revenue'] / campaign['Spend']) if campaign['Spend'] > 0 else 0
    cpa = (campaign['Spend'] / campaign['Conversions']) if campaign['Conversions'] > 0 else 0

    # Prepare input data for the ChatGPT API
    input_data = (
        f"Campaign Name: {campaign['Campaign Name']}\n"
        f"CTR: {ctr:.2f}%\n"
        f"ROAS: {roas:.2f}x\n"
        f"CPA: {cpa:.2f}\n"
        f"Conversions: {campaign['Conversions']}\n"
        f"Spend: {campaign['Spend']}\n"
    )

    # Call the OpenAI ChatCompletion API
    try:
        messages = [
            {"role": "system", "content": "You are a marketing expert. Your task is to analyze the campaign data and provide insights related to trends, anomalies, and opportunities."},
            {"role": "user", "content": f"Analyze this campaign data and identify trends, anomalies, and opportunities:\n\n{input_data}"}
        ]
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        )

        # Process the response and extract insights
        insights = response.choices[0].message.content.strip()

        return render_template('powered_insights.html', insights=insights, campaign_name=campaign["Campaign Name"])

    except Exception as e:
        return jsonify({"error": f"Internal server error: {e}"}), 500

    
@app.route('/report/<campaign_id>')
def report(campaign_id):
    # Retrieve campaign details and optimization actions from the CSV
    # This filters the DataFrame to find rows where the Campaign ID matches the provided campaign_id
    # to_dict(orient="records"): This converts the filtered DataFrame into a list of dictionaries, 
    # where each dictionary represents a campaign record.
    campaign = df[df["Campaign ID"].astype(str) == str(campaign_id)].to_dict(orient="records")
    
    if not campaign:
        return "Campaign not found", 404

    campaign = campaign[0]  # There should only be one result

    # Retrieve insights and recommendations for the specific campaign
    actions = rules.get_campaign_optimization_actions([campaign])

    # Get the most recent insights from the actions
    insights = actions[0]["Insights"]

    # Prepare data for the report
    report_data = {
        'campaign_name': campaign["Campaign Name"],
        'report_date': datetime.today().strftime("%Y-%m-%d"),
        'ctr': (campaign['Clicks'] / campaign['Impressions']) * 100 if campaign['Impressions'] > 0 else 0,
        'roas': (campaign['Revenue'] / campaign['Spend']) if campaign['Spend'] > 0 else 0,
        'cpa': (campaign['Spend'] / campaign['Conversions']) if campaign['Conversions'] > 0 else 0,
        'conversions': campaign['Conversions'],
        'spend': campaign['Spend'],
        'insights': insights
    }


    # The **report_data syntax unpacks the dictionary so that the keys of report_data become variables available in the report.html template.
    return render_template('report.html', **report_data)

# @app.route('/visualization/<campaign_id>')
# def visualization(campaign_id):
#     # Filter data for the selected campaign
#     campaign_data = df[df["Campaign ID"].astype(str) == str(campaign_id)]

#     if campaign_data.empty:
#         return "Campaign not found", 404

#     # Parse the date and metric data
#     campaign_data['Date'] = pd.to_datetime(campaign_data['Date'], errors='coerce')  # Ensure dates are valid
#     campaign_data = campaign_data.dropna(subset=['Date'])  # Drop rows with invalid dates
#     campaign_data = campaign_data.sort_values(by='Date')  # Sort by date

#     # Extract time series data
#     dates = campaign_data['Date'].dt.strftime('%Y-%m-%d').tolist()
#     roas_data = campaign_data['Revenue'] / campaign_data['Spend']  # Compute ROAS
#     ctr_data = (campaign_data['Clicks'] / campaign_data['Impressions']) * 100  # Compute CTR
#     budget_data = campaign_data['Spend']  # Use Spend as the Budget Trend

#     return render_template(
#         'visualization.html',
#         campaign=campaign_data.iloc[0].to_dict(),  # Single campaign's metadata
#         dates=dates,
#         roas=roas_data.tolist(),
#         ctr=ctr_data.tolist(),
#         budget=budget_data.tolist(),
#     )

@app.route('/visualization')
def visualization():
    global df
    if df is None:
        return "Error: Dataframe could not be loaded. Please check the CSV file.", 500

    # Proceed with existing logic
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Ensure dates are valid
    df = df.dropna(subset=['Date']).sort_values(by='Date')  # Drop invalid dates and sort

    # This line extracts all unique dates from the Date column of the df DataFrame and formats them into a list of strings.
    dates = df['Date'].dt.strftime('%Y-%m-%d').unique().tolist()

    # This line calculates the Return on Ad Spend (ROAS) for each date and stores it in the roas_data list.
    roas_data = (df.groupby('Date')['Revenue'].sum() / df.groupby('Date')['Spend'].sum()).fillna(0).tolist()

    # This line calculates the Click-Through Rate (CTR) for each date and stores it in the ctr_data list.
    ctr_data = ((df.groupby('Date')['Clicks'].sum() / df.groupby('Date')['Impressions'].sum()) * 100).fillna(0).tolist()

    # This line calculates the total Spend for each date and stores it in the budget_data list.
    budget_data = df.groupby('Date')['Spend'].sum().tolist()

    return render_template(
        'visualization.html',
        dates=dates,
        roas=roas_data,
        ctr=ctr_data,
        budget=budget_data
    )

@app.route('/store_campaign_id', methods=['POST'])
def store_campaign_id():
    data = request.get_json()
    campaign_id = data.get('campaign_id')
    print(f"Received campaign ID: {campaign_id}")
    session['desired_Id'] = campaign_id
    
    print(f"Stored Campaign ID in session: {session['desired_Id']}")
    result= "Campaign ID stored successfully"
    return '', 204  # No content returned, just an acknowledgment of success


@app.route('/ask', methods=['POST'])
def ask_route():
    data = request.get_json()       # Gets the JSON data sent in the request body.
    user_query = data.get('query')  # Extracts the query key from the JSON data.

    campaign_id = session.get('desired_Id')
    print(f"Stored campaign ID in session: {campaign_id}")

    print(campaign_id)
    print(type(campaign_id))

    response_message = ask(user_query, campaign_id,df2, token_budget=4096 - 100, print_message=False)
    return jsonify({"response": response_message})


# Cleans text by removing HTML tags and extra whitespace.
def clean_text(text):
    cleaned_text = re.sub(r'<.*?>', '', text)
    cleaned_text = re.sub(r'[\t\r\n]+', '', cleaned_text)
    return cleaned_text

def generate_embeddings(text, model="text-embedding-3-large-model"):
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def cosine_similarity(a, b):
    # np.dot(a, b): Computes the dot product between the two vectors.
    # np.linalg.norm(a): Computes the Euclidean norm (magnitude) of vector a.
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


import ast  # for converting embeddings saved as strings back to arrays
import numpy as np

def strings_ranked_by_relatedness(query: str, df2: pd.DataFrame, campaign_id: str, top_n: int = 100):
    df_filtered = df2[df2['Campaign ID'] == campaign_id]

    # Get the embedding for the query
    query_embedding = client.embeddings.create(
        model="text-embedding-ada-002-model",
        input=query
    ).data[0].embedding # The .data[0].embedding extracts the embedding from the API response 
                        # (which is typically a list of embeddings for a batch of inputs, so we access the first element in this list).
    
    # Calculate relatedness for each row in the DataFrame
    results = []
    for _, row in df_filtered.iterrows():
        text = row["Concat"]
        embedding_str = row["ada_v2"]

        # Convert the string representation of the embedding back to a NumPy array
        # This is necessary because embeddings are typically stored as strings in CSV or databases and 
        # need to be converted back to arrays for calculations.
        embedding = np.array(ast.literal_eval(embedding_str))
        
        # Calculate cosine similarity
        relatedness = 1 - distance.cosine(query_embedding, embedding)
        results.append((text, relatedness))
    
    # Sort the results by relatedness in descending order
    # The reverse=True ensures that the most related texts come first.
    # e.g: results = [("Text1", 0.9), ("Text2", 0.7), ("Text3", 0.95)]
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Extract the strings and their relatedness
    strings = [item[0] for item in results]
    relatednesses = [item[1] for item in results]
    
    return strings, relatednesses


#  Returns the number of tokens in a string based on the model being used (e.g., GPT-4).
def num_tokens(text: str, model: str = "gpt-4") -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(
    query: str,
    df2: pd.DataFrame,
    campaign_id: str,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, _ = strings_ranked_by_relatedness(query, df2, campaign_id)
    introduction = 'You are a customer assistant that answers questions or give information about text entered by the user from the given data. The Characters before the fisrt space are the Campaign Ids.'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nConcat:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    campaign_id: str,
    df2: pd.DataFrame = df,
    model: str = "gpt-4",
    token_budget: int = 4096 - 100,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df2, campaign_id, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You are a customer assistant that answers questions based on the given campaign data."},
        {"role": "user", "content": message}, 
    ]
    response = client.chat.completions.create(
        model="gpt-4",  # Directly pass model here instead of in query_message
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content.strip()

     # Split the response into meaningful paragraphs
    formatted_response = response_message.split('\n\n')

    return formatted_response

# @app.route('/api/generate-insights/<campaign_id>', methods=['POST'])
# def generate_insights(campaign_id): 
#     # Find the campaign in the DataFrame
#     campaign = df[df["Campaign ID"].astype(str) == str(campaign_id)].to_dict(orient="records")
#     if not campaign:
#         return jsonify({"error": "Campaign not found"}), 404
#     campaign = campaign[0]

#     # Safeguard calculations to prevent division errors
#     ctr = (campaign['Clicks'] / campaign['Impressions']) * 100 if campaign['Impressions'] > 0 else 0
#     roas = (campaign['Revenue'] / campaign['Spend']) if campaign['Spend'] > 0 else 0
#     cpa = (campaign['Spend'] / campaign['Conversions']) if campaign['Conversions'] > 0 else 0

#     # Prepare input data for the ChatGPT API
#     input_data = (
#         f"Campaign Name: {campaign['Campaign Name']}\n"
#         f"CTR: {ctr:.2f}%\n"
#         f"ROAS: {roas:.2f}x\n"
#         f"CPA: {cpa:.2f}\n"
#         f"Conversions: {campaign['Conversions']}\n"
#         f"Spend: {campaign['Spend']}\n"
#     )
    
#     # Call the OpenAI ChatCompletion API
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a marketing expert."},
#                 {"role": "user", "content": f"Analyze this campaign data and provide actionable insights:\n\n{input_data}"}
#             ],
#             max_tokens=500,
#             temperature=0.3,
#         )
#         insights = response['choices'][0]['message']['content'].split("\n")
#         return jsonify({"insights": insights})

#     except Exception as e:
#         return jsonify({"error": f"Internal server error: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)
