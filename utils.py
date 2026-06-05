import pandas as pd

def load_data():
    df = pd.read_csv("smart_meter_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def calculate_bill(units):
    return units * 7

def calculate_carbon(units):
    return units * 0.82
