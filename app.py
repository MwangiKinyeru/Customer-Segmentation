# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import json
import numpy as np
import os

app = Flask(__name__)

# Load all your saved assets
def load_models():
    """Load your trained models and business rules"""
    try:
        # Load your KMeans model and scaler
        kmeans = joblib.load('Models/kmeans.pkl')
        scaler = joblib.load('Models/scaler.pkl')
        
        # Load your business rules
        with open('Models/business_rules.json', 'r') as f:
            business_rules = json.load(f)
        
        print("✅ All models loaded successfully!")
        return kmeans, scaler, business_rules
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise

# Load everything at startup
kmeans, scaler, business_rules = load_models()

def predict_customer_segment(recency, frequency, monetary):
    """
    Your complete prediction logic:
    1. Check if customer is an outlier (A, B, C)
    2. If not outlier, use KMeans model (Regular, Lapsed, Occasional, Premium)
    """
    # Get your thresholds
    mv_threshold = business_rules['outlier_thresholds']['monetary']
    freq_threshold = business_rules['outlier_thresholds']['frequency']
    
    # Check for outliers (YOUR business rules)
    if monetary > mv_threshold and frequency > freq_threshold:
        cluster_name = "Elite"
    elif monetary > mv_threshold:
        cluster_name = "High_Spender"
    elif frequency > freq_threshold:
        cluster_name = "Power_Shopper"
    else:
        # Regular customer - use your KMeans model
        # Scale the input (same as training)
        features = np.array([[recency, frequency, monetary]])
        scaled_features = scaler.transform(features)
        
        # Predict with KMeans
        cluster_num = kmeans.predict(scaled_features)[0]
        
        # Map to your actual cluster names
        if cluster_num == 0:
            cluster_name = "Regular"
        elif cluster_num == 1:
            cluster_name = "Lapsed"
        elif cluster_num == 2:
            cluster_name = "Occasional"
        else:  # cluster_num == 3
            cluster_name = "Premium"
    
    # Get the display name from business rules
    display_name = business_rules["cluster_mapping"][cluster_name]
    
    # Format the response using your template
    response = business_rules["response_template"].format(segment=display_name)
    
    return response, display_name, cluster_name

# 📱 Route 1: Homepage (Web Interface)
@app.route('/')
def home():
    """Show the input form"""
    return render_template('index.html')

# 📱 Route 2: Handle Web Form Submission
@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission from website"""
    try:
        # Get data from form
        recency = float(request.form['recency'])
        frequency = float(request.form['frequency'])
        monetary = float(request.form['monetary'])
        
        # Make prediction
        response, display_name, cluster_name = predict_customer_segment(recency, frequency, monetary)
        
        # Show result
        return render_template('result.html', 
                             response=response,
                             segment=display_name,
                             recency=recency,
                             frequency=frequency,
                             monetary=monetary)
    except Exception as e:
        return render_template('error.html', error=str(e))

# 🔧 Route 3: API Endpoint (for developers)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API for programmatic access"""
    try:
        data = request.json
        recency = float(data['recency'])
        frequency = float(data['frequency'])
        monetary = float(data['monetary'])
        
        response, display_name, cluster_name = predict_customer_segment(recency, frequency, monetary)
        
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

# 🩺 Route 4: Health check
@app.route('/health')
def health():
    """Check if the app is running"""
    return jsonify({
        'status': 'healthy',
        'message': 'Customer Segmentation API is running',
        'model_loaded': True
    })

# 🏠 Route 5: About page
@app.route('/about')
def about():
    """Information about the segments"""
    return render_template('about.html', 
                          thresholds=business_rules['outlier_thresholds'],
                          segments=business_rules['cluster_mapping'])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)