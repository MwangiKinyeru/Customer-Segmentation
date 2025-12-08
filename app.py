# app.py - Customer Segmentation Web Application
from flask import Flask, request, jsonify, render_template
import joblib
import json
import numpy as np
import os

app = Flask(__name__)

# 1. CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'Models')


# 2. LOAD MODELS
def load_models():
    """Load trained models and business rules"""
    kmeans_path = os.path.join(MODELS_DIR, 'kmeans.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    rules_path = os.path.join(MODELS_DIR, 'business_rules.json')
    
    kmeans = joblib.load(kmeans_path)
    scaler = joblib.load(scaler_path)
    
    with open(rules_path, 'r') as f:
        business_rules = json.load(f)
    
    return kmeans, scaler, business_rules

# Load models at startup
kmeans, scaler, business_rules = load_models()

# 3. PREDICTION LOGIC
def predict_customer_segment(recency, frequency, monetary):
    """
    Predict customer segment based on RFM values
    1. Check for outliers (High_Spender, Power_Shopper, Elite)
    2. Use KMeans for regular customers (Regular, Lapsed, Occasional, Premium)
    """
    # Get thresholds
    mv_threshold = business_rules['outlier_thresholds']['monetary']
    freq_threshold = business_rules['outlier_thresholds']['frequency']
    
    # Check for outliers
    if monetary > mv_threshold and frequency > freq_threshold:
        cluster_name = "Elite"
    elif monetary > mv_threshold:
        cluster_name = "High_Spender"
    elif frequency > freq_threshold:
        cluster_name = "Power_Shopper"
    else:
        # Regular customer - use KMeans
        features = np.array([[recency, frequency, monetary]])
        scaled_features = scaler.transform(features)
        cluster_num = kmeans.predict(scaled_features)[0]
        
        # Map to cluster names
        if cluster_num == 0:
            cluster_name = "Regular"
        elif cluster_num == 1:
            cluster_name = "Lapsed"
        elif cluster_num == 2:
            cluster_name = "Occasional"
        else:  # cluster_num == 3
            cluster_name = "Premium"
    
    # Get display name and format response
    display_name = business_rules["cluster_mapping"][cluster_name]
    response = business_rules["response_template"].format(segment=display_name)
    
    return response, display_name, cluster_name

# 4. FLASK ROUTES
@app.route('/')
def home():
    """Landing page"""
    return render_template('index.html')

@app.route('/predict-form')
def predict_form():
    """Prediction form page"""
    return render_template('predict_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction form submission"""
    try:
        recency = float(request.form['recency'])
        frequency = float(request.form['frequency'])
        monetary = float(request.form['monetary'])
        
        response, display_name, cluster_name = predict_customer_segment(
            recency, frequency, monetary
        )
        
        return render_template('result.html', 
                             response=response,
                             segment=display_name,
                             recency=recency,
                             frequency=frequency,
                             monetary=monetary)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/about')
def about():
    """Information about customer segments"""
    return render_template('about.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint"""
    try:
        data = request.json
        recency = float(data['recency'])
        frequency = float(data['frequency'])
        monetary = float(data['monetary'])
        
        response, display_name, cluster_name = predict_customer_segment(
            recency, frequency, monetary
        )
        
        return jsonify({
            'prediction': response,
            'segment': display_name,
            'cluster_code': cluster_name,
            'inputs': {
                'recency': recency,
                'frequency': frequency,
                'monetary': monetary
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'thresholds': business_rules['outlier_thresholds']
    })

# 5. RUN APPLICATION
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)