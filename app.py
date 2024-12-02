from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)
CORS(app)

# Add a root route to prevent 404 on "/"
@app.route('/')
def index():
    return 'Welcome to the Sales Analysis API! Please POST your data to /analyze.'

@app.route('/analyze', methods=['POST'])
def analyze():
    # Ensure a file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    
    file = request.files['file']

    # Check if the file is a valid CSV file
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400
    
    try:
        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(file)

        # Validate dataset columns
        if 'Month' not in df.columns or 'Sales' not in df.columns:
            return jsonify({'error': 'Dataset must contain "Month" and "Sales" columns.'}), 400

        # Process data
        df['Month'] = pd.to_datetime(df['Month'])
        df.set_index('Month', inplace=True)
        X = np.arange(len(df)).reshape(-1, 1)  # X is the time index (0, 1, 2, ...)
        y = df['Sales'].values

        # Train linear regression model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)

        # Plot the actual vs predicted sales
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, y, label='Actual Sales', marker='o')
        plt.plot(df.index, y_pred, label='Predicted Sales', linestyle='--')
        plt.title('Sales Analysis')
        plt.xlabel('Month')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid()

        # Save plot to a base64 string to return as part of the JSON response
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return jsonify({
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'plot': plot_url
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing the file: {str(e)}'}), 400

# Entry point for Gunicorn to run the app
if __name__ == '__main__':
    app.config['ENV'] = 'production'  # Set Flask to production mode
    app.config['DEBUG'] = False  # Disable debug mode in production
    app.config['TESTING'] = False  # Disable testing mode

    # This will not be used when deploying with Gunicorn
    # app.run(host='0.0.0.0', port=5000)
