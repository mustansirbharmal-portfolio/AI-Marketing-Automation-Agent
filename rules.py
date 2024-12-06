import pandas as pd
from datetime import timedelta, datetime

# Function to fetch campaign data and apply optimization rules
def get_campaign_optimization_actions(campaigns):
    # Convert the campaign data into a DataFrame
    df = pd.DataFrame(campaigns)

    # Generate a Date column (assigning today's date and going back by a number of days for each row)
    today = datetime.today()
    df["Date"] = pd.date_range(end=today, periods=len(df), freq="D")

    # Ensure the 'Date' column is a datetime type
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

    # Iterate over each campaign
    for _, row in df.iterrows():
        action = {"Campaign ID": row["Campaign ID"], "Actions": [], "Insights": []}

        # ---------------------
        # Rule 1: Pause Campaign
        # ---------------------
        # Rule 1a: Pause if CTR < 1% in the last 3 days
        if "Date" in df.columns:
            last_3_days = df[(df["Date"] > row["Date"] - timedelta(days=3)) & (df["Date"] <= row["Date"])]
            total_clicks_last_3_days = last_3_days["Clicks"].sum()
            total_impressions_last_3_days = last_3_days["Impressions"].sum()
            ctr_last_3_days = (total_clicks_last_3_days / total_impressions_last_3_days) * 100 if total_impressions_last_3_days > 0 else 0

            if ctr_last_3_days < 1:
                action["Actions"].append("Pause")
                action["Insights"].append("CTR is below 1% in the last 3 days. Improve ad creatives or targeting.")

        # Rule 1b: Pause if Cost Per Conversion > 3x TARGET_CPA
        if row["Cost_Per_Conversion"] > 3 * TARGET_CPA:
            action["Actions"].append("Pause")
            action["Insights"].append("Cost per Conversion is too high. Adjust targeting or bids.")

        # Initialize flags to determine if budget actions should be applied
        decrease_budget_triggered = False

        # ---------------------
        # Rule 2: Decrease Budget
        # ---------------------
        # Rule 2a: Decrease Budget if ROAS < 1.5
        if row["ROAS"] < 1.5:
            action["Actions"].append("Decrease Budget")
            action["Insights"].append("ROAS is too low. Consider reducing budget or adjusting targeting.")
            decrease_budget_triggered = True

        # Rule 2b: Decrease Budget if ROAS < 1.5 for 5 consecutive days
        last_5_days = df[(df["Date"] > row["Date"] - timedelta(days=5)) & (df["Date"] <= row["Date"])]
        low_roas_count = len(last_5_days[last_5_days["ROAS"] < 1.5])

        if low_roas_count == 5:
            # To avoid duplicate "Decrease Budget" action if already triggered by Rule 2a
            if "Decrease Budget" not in action["Actions"]:
                action["Actions"].append("Decrease Budget")
                action["Insights"].append("ROAS is below 1.5 for 5 consecutive days. Consider reducing budget or adjusting targeting.")
            decrease_budget_triggered = True

        # ---------------------
        # Rule 3: Increase Budget
        # ---------------------
        # Only apply increase budget rules if no decrease budget actions were triggered
        if not decrease_budget_triggered:
            # Rule 3a: Increase Budget if ROAS > 4
            

            # Rule 3b: Increase Budget if Conversions increased by more than 20% week-over-week
            current_week_start = row["Date"] - timedelta(days=row["Date"].weekday())  # Start of current week (Monday)
            last_week_start = current_week_start - timedelta(days=7)  # Start of previous week

            current_week_data = df[(df["Date"] >= current_week_start) & (df["Date"] < current_week_start + timedelta(days=7))]
            last_week_data = df[(df["Date"] >= last_week_start) & (df["Date"] < last_week_start + timedelta(days=7))]

            current_week_conversions = current_week_data["Conversions"].sum()
            last_week_conversions = last_week_data["Conversions"].sum()

            # This checks if conversions have increased by more than 20% compared to the previous week.
            if last_week_conversions > 0 and current_week_conversions > last_week_conversions * 1.2:
            
             action["Insights"].append("Conversions increased by more than 20% week-over-week. Consider scaling up the campaign.")

            if last_week_conversions > 0 and current_week_conversions > last_week_conversions * 1.2 or row["ROAS"] > 4:
             action["Actions"].append("Increase Budget")

            if row["ROAS"] > 4:
             action["Insights"].append("ROAS is excellent. Consider scaling up the campaign.")

        # Append the action for the current campaign
        actions.append(action)

    # df.to_csv('campaign_data.csv', index=False)

    return actions
