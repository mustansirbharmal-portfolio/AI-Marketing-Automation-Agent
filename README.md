# AI-Marketing-Automation-Agent
# Campaign Insights and Chatbot System

## Overview
The **Campaign Insights and Chatbot System** is a solution designed to process marketing campaign data, generate insights, and assist with decision-making using AI-powered tools. It uses a data-driven architecture, leveraging **Azure OpenAI's text-embedding-ada-002 model** for embeddings and the **Azure ChatGPT API** to generate insights and creative ideas.

## Features
- Data-driven pipeline for campaign data processing.
- Generates embeddings from campaign data using **Azure OpenAI text-embedding-ada-002 model**.
- Provides marketing insights through **Azure ChatGPT API**.
- Real-time interaction with a chatbot for personalized marketing recommendations.

## Architecture
The system uses the **Data-driven architecture** to generate insights and ideas using **Azure ChatGPT API** and **Retrieval-Augmented Generation (RAG)** method for the chatbot, where the chatbot queries relevant embedded data based on user inputs and provides actionable insights. It operates without a traditional database by relying on embedded CSV files that contain campaign data.

## Tools and Libraries
- **Azure OpenAI API** for embeddings and insights generation.
- **Python** libraries for data preprocessing:
  - `pandas` for data manipulation.
  - `requests` for API calls.
  - `numpy` for numerical operations.
- **CSV** file format for campaign data storage.

## Setup Instructions

### 1. Clone the repository
git clone https://github.com/your-username/Campaign-Insights-Chatbot-System.git<br>
cd Campaign-Insights-Chatbot-System<br>

### 2. Install Dependencies
Ensure you have Python 3.6+ installed. Then, install the required Python libraries using pip:<br>
**pip install -r requirements.txt**

### 3. Set up Azure OpenAI API credentials
**(i)** Sign up for an Azure account.<br>
**(ii)** Set up the OpenAI API and get your API key.<br>
**(iii)** Copy the **API Key, API Version and Azure Endpoint** and paste it in below line of code of app.py and embedding.py file:<br>

client = AzureOpenAI(<br>
    api_key="",<br>
    api_version="",<br>
    azure_endpoint=""<br>
)

### 4. Prepare the CSV data
- A sample CSV file (campaign_data.csv) is included in the repository to test the system.<br>
- Or use your file which includes of Campaign ID, Campaign Name, Impressions, Clicks, Conversions, Spend, Revenue and Status of CSV File Format.

If you are using your file campaign data file then make sure you embed the file in embedding.py file and then run the file using command "python embedding.py"

### 5. Run the system
python app.py


This project is licensed under the MIT License - see the LICENSE file for details





