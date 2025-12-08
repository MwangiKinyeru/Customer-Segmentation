# app.py - Customer Segmentation Web Application
from flask import Flask, request, jsonify, render_template
import joblib
import json
import numpy as np
import os

app = Flask(__name__)

# ============================================
# 0. PATH CONFIGURATION - CORRECTED
# ============================================
# Get the absolute path to your project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"🔍 Base directory: {BASE_DIR}")

# Your models are in 'Models' folder (plural)
MODELS_DIR = os.path.join(BASE_DIR, 'Models')

print(f"📁 Models directory: {MODELS_DIR}")
print(f"📁 Directory exists: {os.path.exists(MODELS_DIR)}")

# ============================================
# 1. LOAD MODELS AND BUSINESS RULES
# ============================================
def load_models():
    """Load your trained models and business rules"""
    try:
        # Define file paths
        kmeans_path = os.path.join(MODELS_DIR, 'kmeans.pkl')
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        rules_path = os.path.join(MODELS_DIR, 'business_rules.json')
        
        print(f"\n🔍 Checking for model files:")
        print(f"   • kmeans.pkl: {'✅ FOUND' if os.path.exists(kmeans_path) else '❌ MISSING'}")
        print(f"   • scaler.pkl: {'✅ FOUND' if os.path.exists(scaler_path) else '❌ MISSING'}")
        print(f"   • business_rules.json: {'✅ FOUND' if os.path.exists(rules_path) else '❌ MISSING'}")
        
        # Load your KMeans model
        print("\n📦 Loading KMeans model...")
        kmeans = joblib.load(kmeans_path)
        print(f"   ✅ Loaded: {kmeans.n_clusters} clusters")
        
        # Load your scaler
        print("📦 Loading scaler...")
        scaler = joblib.load(scaler_path)
        print(f"   ✅ Loaded: {type(scaler).__name__}")
        
        # Load your business rules
        print("📦 Loading business rules...")
        with open(rules_path, 'r') as f:
            business_rules = json.load(f)
        
        print(f"   ✅ Monetary threshold: ${business_rules['outlier_thresholds']['monetary']:,.2f}")
        print(f"   ✅ Frequency threshold: {business_rules['outlier_thresholds']['frequency']} purchases")
        
        print("\n🎉 All models loaded successfully!")
        return kmeans, scaler, business_rules
        
    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        
        # Show what's actually in the directory
        if os.path.exists(MODELS_DIR):
            print(f"\n📂 Actual files in {MODELS_DIR}:")
            for file in os.listdir(MODELS_DIR):
                full_path = os.path.join(MODELS_DIR, file)
                size = os.path.getsize(full_path) / 1024  # KB
                print(f"   - {file} ({size:.1f} KB)")
        else:
            print(f"\n📂 {MODELS_DIR} does not exist!")
            
        # Exit since we know files exist
        raise

# Load everything at startup
print("\n" + "="*60)
print("🚀 Starting Customer Segmentation Web App")
print("="*60)
kmeans, scaler, business_rules = load_models()

# ============================================
# 2. PREDICTION LOGIC
# ============================================
def predict_customer_segment(recency, frequency, monetary):
    """
    Your complete prediction logic:
    1. Check if customer is an outlier (High_Spender, Power_Shopper, Elite)
    2. If not outlier, use KMeans model (Regular, Lapsed, Occasional, Premium)
    """
    # Get your thresholds from JSON
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
        
        # Predict with KMeans (returns 0, 1, 2, or 3)
        cluster_num = kmeans.predict(scaled_features)[0]
        
        # Map KMeans number to your cluster names
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

# ============================================
# 3. FLASK ROUTES
# ============================================

# Route 1: Homepage (Landing Page)
@app.route('/')
def home():
    """Show the landing page"""
    return render_template('index.html')

# Route 2: Prediction Form Page
@app.route('/predict-form')
def predict_form():
    """Show the prediction form"""
    return render_template('predict_form.html')

# Route 3: Handle Form Submission
@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission from website"""
    try:
        # Get data from form
        recency = float(request.form['recency'])
        frequency = float(request.form['frequency'])
        monetary = float(request.form['monetary'])
        
        # Make prediction
        response, display_name, cluster_name = predict_customer_segment(
            recency, frequency, monetary
        )
        
        # Show result
        return render_template('result.html', 
                             response=response,
                             segment=display_name,
                             recency=recency,
                             frequency=frequency,
                             monetary=monetary)
    except Exception as e:
        return render_template('error.html', error=str(e))

# Route 4: About Page
@app.route('/about')
def about():
    """Show information about customer segments"""
    return render_template('about.html')

# Route 5: API Endpoint (for developers)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API for programmatic access"""
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

# Route 6: Health check
@app.route('/health')
def health():
    """Check if the app is running"""
    return jsonify({
        'status': 'healthy',
        'message': 'Customer Segmentation API is running',
        'model_loaded': True,
        'thresholds': business_rules['outlier_thresholds'],
        'clusters': list(business_rules['cluster_mapping'].keys()),
        'models_directory': MODELS_DIR
    })

# Route 7: Quick test example
@app.route('/test-example')
def test_example():
    """Test with example customer data"""
    # Test cases
    test_data = [
        ("Premium Example", 30, 15, 8500),
        ("Regular Example", 45, 8, 1200),
        ("Elite Example", 10, 25, 15000),
    ]
    
    results = []
    for name, recency, frequency, monetary in test_data:
        response, display_name, cluster_name = predict_customer_segment(
            recency, frequency, monetary
        )
        results.append({
            'name': name,
            'inputs': f"R={recency}, F={frequency}, M=${monetary}",
            'prediction': response,
            'segment': display_name
        })
    
    # Create HTML response
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Examples</title>
        <style>
            body { font-family: Arial; padding: 20px; background: #f5f5f5; }
            .test-card { background: white; border-radius: 10px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .success { border-left: 5px solid #28a745; }
            h1 { color: #333; }
            a { display: inline-block; margin-top: 20px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>🧪 Test Predictions</h1>
        <p>Testing with example customer data:</p>
    """
    
    for result in results:
        html += f"""
        <div class="test-card success">
            <h3>{result['name']}</h3>
            <p><strong>Inputs:</strong> {result['inputs']}</p>
            <p><strong>Prediction:</strong> {result['prediction']}</p>
            <p><strong>Segment:</strong> {result['segment']}</p>
        </div>
        """
    
    html += """
        <br>
        <a href="/predict-form">📊 Try Your Own Prediction</a>
        <a href="/" style="background: #6c757d; margin-left: 10px;">🏠 Back to Home</a>
    </body>
    </html>
    """
    
    return html

# ============================================
# 4. RUN THE APPLICATION
# ============================================
if __name__ == '__main__':
    # Display startup information
    print("\n" + "="*60)
    print("🎯 CUSTOMER SEGMENTATION WEB APP READY")
    print("="*60)
    print(f"💰 Monetary Threshold: ${business_rules['outlier_thresholds']['monetary']:,.2f}")
    print(f"🛒 Frequency Threshold: {business_rules['outlier_thresholds']['frequency']} purchases")
    print(f"📊 Clusters Available: {len(business_rules['cluster_mapping'])}")
    
    print("\n🌐 APPLICATION URLS:")
    print("   Homepage:        http://127.0.0.1:5000")
    print("   Predict Form:    http://127.0.0.1:5000/predict-form")
    print("   About Segments:  http://127.0.0.1:5000/about")
    print("   Test Examples:   http://127.0.0.1:5000/test-example")
    print("   Health Check:    http://127.0.0.1:5000/health")
    print("\n📝 Try these test cases:")
    print("   • Regular: 45 days, 8 purchases, $1,200")
    print("   • Premium: 30 days, 15 purchases, $8,500")
    print("   • Elite:   10 days, 25 purchases, $15,000")
    print("="*60 + "\n")
    
    # Start the Flask server
    app.run(debug=True, host='0.0.0.0', port=5000)