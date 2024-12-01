# from pymongo import MongoClient
import pandas as pd
from datetime import timedelta, datetime

# # Connect to MongoDB
# client = MongoClient("mongodb://localhost:27017/")
# db = client["marketing_db"]
# collection = db["campaigns"]

# Function to fetch campaign data and apply optimization rules
def get_campaign_optimization_actions(campaigns):
    # Convert the campaign data into a DataFrame
    df = pd.DataFrame(campaigns)

    # Generate a Date column (for example, assigning today's date and going back by a number of days for each row)
    # Here, we assume the data is from the past 10 days, you can modify this logic as needed
    today = datetime.today()

    # Generates a sequence of valid dates and stores it in the "Date" column.
    df["Date"] = pd.date_range(end=datetime.today(), periods=len(df), freq="D")

    # Ensure the 'Date' column is a datetime type
    if "Date" in df.columns:

        # Ensures all values in the "Date" column are proper datetime objects.
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

    # Define TARGET_CPA
    TARGET_CPA = 50  # Example target CPA

    """
    CTR: How often people click on your ad. (Higher is better)
    CPR: How much it costs to reach someone. (Lower is better)
    ROAS: How much money you earn from your ad spend. (Higher is better)
    """

    # Add performance metrics
    df["CTR"] = (df["Clicks"] / df["Impressions"]) * 100
    df["Cost_Per_Conversion"] = df["Spend"] / df["Conversions"]
    df["ROAS"] = df["Revenue"] / df["Spend"]

    # List to store actions for each campaign
    actions = []

    # row is a Pandas Series representing the data in a single row of the DataFrame.
    # The underscore _ is used because we don’t need the index of each row (just the row data). 
    # However, if you wanted to use the index, you could use for index, row in df.iterrows():.

    for _, row in df.iterrows():
        action = {"Campaign ID": row["Campaign ID"], "Actions": [], "Insights": []}

        # Rule 1: Pause Campaign if CTR < 1% in the last 3 days
        if "Date" in df.columns:
            # Calculate CTR for the last 3 days (including the current day)
            # timedelta: A function from the datetime module that represents a duration (here, 3 days).
            last_3_days = df[(df["Date"] > row["Date"] - timedelta(days=3)) & (df["Date"] <= row["Date"])]
            ctr_last_3_days = (last_3_days["Clicks"].sum() / last_3_days["Impressions"].sum()) * 100

            if ctr_last_3_days < 1:
                action["Actions"].append("Pause")
                action["Insights"].append("CTR is below 1% in the last 3 days. Improve ad creatives or targeting.")
        
        # Rule 2: Pause Campaign if Cost Per Conversion > 3x the Target CPA
        if row["Cost_Per_Conversion"] > 3 * TARGET_CPA:
            action["Actions"].append("Pause")
            action["Insights"].append("Cost per Conversion is too high. Adjust targeting or bids.")
        
        # Rule 3: Increase Budget if ROAS > 4
        if row["ROAS"] > 4:
            action["Actions"].append("Increase Budget")
            action["Insights"].append("ROAS is excellent. Consider scaling up the campaign.")

        # Rule 4: Decrease Budget if ROAS < 1.5
        if row["ROAS"] < 1.5:
            action["Actions"].append("Decrease Budget")
            action["Insights"].append("ROAS is too low. Consider reducing budget or adjusting targeting.")
        
        # Rule 5: Increase Budget if Conversions increased by more than 20% week-over-week
        # Calculate conversions for the last 2 weeks
        # .weekday() returns the weekday of the given date, where Monday is 0 and Sunday is 6.
        
        current_week_start = row["Date"] - timedelta(days=row["Date"].weekday())  # Start of current week (Monday)
        last_week_start = current_week_start - timedelta(days=7)  # Start of previous week

        # The condition df["Date"] >= current_week_start ensures that only rows from the current week 
        # (from Monday to the end of the current week) are selected.
        # The condition df["Date"] < current_week_start + timedelta(days=7) ensures that the data is less than the start of the next week.
        current_week_data = df[(df["Date"] >= current_week_start) & (df["Date"] < current_week_start + timedelta(days=7))]

        
        last_week_data = df[(df["Date"] >= last_week_start) & (df["Date"] < last_week_start + timedelta(days=7))]

        current_week_conversions = current_week_data["Conversions"].sum()
        last_week_conversions = last_week_data["Conversions"].sum()


        # This checks if conversions have increased by more than 20% compared to the previous week.
        if last_week_conversions > 0 and current_week_conversions > last_week_conversions * 1.2:
            action["Actions"].append("Increase Budget")
            action["Insights"].append("Conversions increased by more than 20% week-over-week. Consider scaling up the campaign.")
        
        # Rule 6: Decrease Budget if ROAS < 1.5 for 5 consecutive days
        # Calculate ROAS for the last 5 days

        # This ensures that the "Date" in the DataFrame is greater than 5 days before row["Date"], meaning it 
        # selects data starting from 5 days ago.

        # last_5_days filters the DataFrame to include data from 5 days before the current row’s 
        # date (row["Date"]) up to and including the row["Date"].
        last_5_days = df[(df["Date"] > row["Date"] - timedelta(days=5)) & (df["Date"] <= row["Date"])]

        # low_roas_count calculates how many of the filtered rows have a ROAS value less than 1.5, which can help in 
        # identifying underperforming campaigns or days where the advertising budget might need adjustments.
        low_roas_count = len(last_5_days[last_5_days["ROAS"] < 1.5])

        if low_roas_count == 5:
            action["Actions"].append("Decrease Budget")
            action["Insights"].append("ROAS is below 1.5 for 5 consecutive days. Consider reducing budget or adjusting targeting.")
        
        actions.append(action)

        # df.to_csv('campaign_data.csv', index=False)

    
    return actions

