

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def forecast_sales(csv_file, store_num, dept_num):
    try:
        df = pd.read_csv(csv_file)

        required_cols = {'Store', 'Dept', 'Week', 'Year', 'Weekly_Sales'}
        if not required_cols.issubset(df.columns):
            return None, "CSV must contain the columns: Store, Dept, Week, Year, Weekly_Sales"

        filtered_df = df[(df['Store'] == store_num) & (df['Dept'] == dept_num)]

        if filtered_df.empty:
            return None, "No data found for this store and department combination."

        X = filtered_df[['Store', 'Dept', 'Week', 'Year']]
        y = filtered_df['Weekly_Sales']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.xlabel('Actual Sales')
        plt.ylabel('Predicted Sales')
        plt.title(f'Actual vs Predicted Sales (MAE: {mae:.2f})')
        plt.grid(True)

        return plt, f"Mean Absolute Error: {mae:.2f}"

    except Exception as e:
        return None, f"Error: {str(e)}"

demo = gr.Interface(
    fn=forecast_sales,
    inputs=[
        gr.File(label="Upload Sales CSV File", file_types=[".csv"]),
        gr.Number(label="Store Number", value=1),
        gr.Number(label="Department Number", value=1),
    ],
    outputs=[
        gr.Plot(label="Sales Forecast Plot"),
        gr.Textbox(label="Error (MAE) or Message")
    ],
    title="Sales Forecasting & Optimization",
    description="Upload Walmart sales data (.csv) and predict weekly sales by store and department using a machine learning model."
)

if __name__ == "__main__":
    demo.launch()
