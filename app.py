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


# Add root route to prevent 404 on "/"
@app.route('/')
def index():
    return 'Welcome to the Sales Analysis API! Please POST your data to /analyze.'


@app.route('/analyze', methods=['POST'])
def analyze():
    # Upload dataset
    file = request.files['file']
    df = pd.read_csv(file)

    # Validate dataset
    if 'Month' not in df.columns or 'Sales' not in df.columns:
        return jsonify({'error': 'Dataset must contain "Month" and "Sales" columns.'}), 400

    # Process data
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Sales'].values

    # Train model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)

    # Plot graph
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, y, label='Actual Sales', marker='o')
    plt.plot(df.index, y_pred, label='Predicted Sales', linestyle='--')
    plt.title('Sales Analysis')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid()

    # Save plot to base64 string
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


if __name__ == '__main__':
    # Set Flask to production mode
    app.config['ENV'] = 'production'  # This sets the environment to production
    app.config['DEBUG'] = False  # Disable debug mode

    # Use the PORT environment variable provided by Render
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set

    # Run the app on 0.0.0.0 and use the correct port
    app.run(host='0.0.0.0', port=port)
