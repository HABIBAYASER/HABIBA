import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def forecast_sales(file, store, dept):
    df = pd.read_csv(file.name)

    # تنظيف وتنسيق البيانات
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    filtered = df[(df['Store'] == store) & (df['Dept'] == dept)]

    if filtered.empty:
        return None, "⚠️ No data found for the selected Store and Dept."

    X = filtered[['Year', 'Month', 'Day']]
    y = filtered['Weekly_Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    plt.figure(figsize=(10, 5))
    plt.plot(filtered['Date'], filtered['Weekly_Sales'], label='Actual Sales')
    plt.title(f'Store {store} | Dept {dept} - Weekly Sales')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    return plt, f"MAE: {mae:.2f}"

demo = gr.Interface(
    fn=forecast_sales,
    inputs=[
        gr.File(label="Upload Sales CSV File"),
        gr.Number(label="Store Number", value=1),
        gr.Number(label="Department Number", value=1),
    ],
    outputs=[
        gr.Plot(label="Sales Forecast Plot"),
        gr.Textbox(label="Error (MAE)")
    ],
    title="Sales Forecasting & Optimization",
    description="Upload Walmart sales data and predict future sales by store and department."
)

demo.launch()

