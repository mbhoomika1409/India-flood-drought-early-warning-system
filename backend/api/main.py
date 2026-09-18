from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and district data
model = joblib.load("/Users/bhoomikamt/EarthPulse/backend/models/xgboost_india_v2.pkl")
df_districts = pd.read_csv("/Users/bhoomikamt/EarthPulse/backend/data/india_districts_ml.csv")

# District coordinates for NASA API
COORDS = {
    "UDUPI": (13.3409, 74.7421),
    "MYSORE": (12.2958, 76.6394),
    "BIKANER": (28.0229, 73.3119),
    "DAVANGERE": (14.4644, 75.9218),
    "MUMBAI": (19.0760, 72.8777),
    "CHENNAI": (13.0827, 80.2707),
    "HYDERABAD": (17.3850, 78.4867),
    "DELHI": (28.6139, 77.2090),
    "BANGALORE URB": (12.9716, 77.5946),
    "PUNE": (18.5204, 73.8567),
    "KOLKATA": (22.5726, 88.3639),
    "JAIPUR": (26.9124, 75.7873),
    "PATNA": (25.5941, 85.1376),
    "GUWAHATI": (26.1445, 91.7362),
    "BHOPAL": (23.2599, 77.4126),
    "LUCKNOW": (26.8467, 80.9462),
    "RANCHI": (23.3441, 85.3096),
    "SHIMLA": (31.1048, 77.1734),
    "DEHRADUN": (30.3165, 78.0322),
    "EAST KHASI HI": (25.5788, 91.8933),
}

class DistrictInput(BaseModel):
    district: str
    state: str

def fetch_live_nasa(lat, lon):
    end   = datetime.today()
    start = end - timedelta(days=60)
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=PRECTOTCORR,T2M,RH2M,WS2M"
        f"&community=RE"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start.strftime('%Y%m%d')}"
        f"&end={end.strftime('%Y%m%d')}"
        f"&format=JSON"
    )
    response = requests.get(url, timeout=30)
    data = response.json()
    params = data['properties']['parameter']
    df = pd.DataFrame(params)
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    df.columns = ['Rainfall','Temperature','Humidity','WindSpeed']
    
    # Replace NASA missing values with NaN
    df = df.replace(-999, np.nan)
    df = df.replace(-9999, np.nan)
    df = df.dropna()
    
    return df

@app.post("/predict")
def predict(input: DistrictInput):
    district = input.district.upper().strip()
    state    = input.state.upper().strip()

    # Find district in dataset
    exact = df_districts[df_districts['DISTRICT'].str.upper() == district]
    if len(exact) == 0:
        return {"error": f"District '{input.district}' not found. Check spelling."}

    row = exact.iloc[0]

    # Build historical features
    months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    features = pd.DataFrame([{
        **{m: row[m] for m in months},
        'monsoon_total': row['monsoon_total'],
        'winter_total':  row['winter_total'],
        'summer_total':  row['summer_total'],
        'annual_total':  row['annual_total'],
    }])

    # Historical prediction
    pred  = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    labels = {0: "Drought Risk", 1: "Normal", 2: "Flood Risk"}

    # Try live NASA data if coordinates available
    live_data = None
    if district in COORDS:
        try:
            lat, lon = COORDS[district]
            df_live = fetch_live_nasa(lat, lon)
            live_rain = df_live['Rainfall'].sum()
            live_temp = df_live['Temperature'].mean()
            live_humidity = df_live['Humidity'].mean()
            live_data = {
                "last_60_days_rainfall": round(live_rain, 1),
                "avg_temperature": round(live_temp, 1),
                "avg_humidity": round(live_humidity, 1),
                "data_source": "NASA POWER Satellite (Live)"
            }
        except:
            live_data = None

    # Dynamic top factors
    factors = []
    if row['monsoon_total'] > 1500:
        factors.append("Very high monsoon rainfall — flood prone zone")
    elif row['monsoon_total'] < 400:
        factors.append("Very low monsoon rainfall — drought prone zone")
    else:
        factors.append("Monsoon rainfall within normal range")

    if row['annual_total'] > 2000:
        factors.append("High annual rainfall district")
    elif row['annual_total'] < 500:
        factors.append("Low annual rainfall — arid zone")

    if row['JUL'] > 300:
        factors.append("July rainfall historically very high")
    if row['JUN'] > 300:
        factors.append("June rainfall historically very high")

    if live_data:
        if live_data['last_60_days_rainfall'] > 300:
            factors.append(f"Live: Heavy rainfall in last 60 days ({live_data['last_60_days_rainfall']}mm)")
        elif live_data['last_60_days_rainfall'] < 50:
            factors.append(f"Live: Very low rainfall in last 60 days ({live_data['last_60_days_rainfall']}mm)")

    return {
        "district": row['DISTRICT'],
        "state": row['STATE_UT_NAME'],
        "prediction": labels[pred],
        "flood_risk":   round(float(proba[2]) * 100, 1),
        "drought_risk": round(float(proba[0]) * 100, 1),
        "normal":       round(float(proba[1]) * 100, 1),
        "top_factors":  factors,
        "live_data":    live_data
    }

@app.get("/")
def root():
    return {"message": "EarthPulse API running! 🌍"}