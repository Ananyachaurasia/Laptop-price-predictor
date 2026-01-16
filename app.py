from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('laptop.pkl', 'rb'))

# Mock data for demonstration
companies =df['Company'].unique()
types = df['TypeName'].unique()
ram = [2,4,6,8,12,16,24,32,64]
touchscreen = ['No','Yes']
ips = ['No','Yes']
screen_size = [ 10.0, 18.0, 13.0]
resolutions = ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440',
               '2304x1440']
cpu=df['CPU brand'].unique()
hdd = [0,128,256,512,1024,2048]
ssd = [0,8,128,256,512,1024]
gpu = df['Gpu brand'].unique()
os = df['os'].unique()



@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')


@app.route('/predict')
def predict_page():
    """Prediction form page"""
    return render_template('predict.html',
                           companies=companies,
                           types=types,
                           ram=ram,
                           touchscreen=touchscreen,
                           ips=ips,
                           screen_size=screen_size,
                           resolutions=resolutions,
                           cpus=cpu,
                           hdd=hdd,
                           ssd=ssd,
                           gpus=gpu,
                           operating_systems=os)


@app.route('/predict_price', methods=['POST'])
def predict_price():
    """Handle prediction request with data cleaning and validation"""
    try:
        # 1. Get and Clean Data (Removing 'GB' and 'kg' strings to avoid conversion errors)
        company = request.form.get('company')
        type_name = request.form.get('type')

        # Strip text from numeric fields to avoid ValueError
        ram = int(request.form.get('ram').replace('GB', '').strip())
        weight = float(request.form.get('weight').replace('kg', '').strip())

        touchscreen = 1 if request.form.get('touchscreen') == 'Yes' else 0
        ips = 1 if request.form.get('ips') == 'Yes' else 0

        screen_size = float(request.form.get('screen_size'))
        resolution = request.form.get('resolution')

        cpu = request.form.get('cpu')

        # More stripping for storage fields
        hdd = int(request.form.get('hdd').replace('GB', '').strip())
        ssd = int(request.form.get('ssd').replace('GB', '').strip())

        gpu = request.form.get('gpu')
        os = request.form.get('os')

        # 2. Calculate PPI (Pixels Per Inch)
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

        # 3. Create DataFrame (Solves the "Feature Names" Warning)
        # IMPORTANT: Use the EXACT column names from your Jupyter Notebook training
        data_dict = {
            'Company': [company],
            'TypeName': [type_name],
            'Ram': [ram],
            'Weight': [weight],
            'Touchscreen': [touchscreen],
            'IPS': [ips],
            'ppi': [ppi],
            'CPU brand': [cpu],  # Double check if your notebook used 'Cpu' or 'Cpu brand'
            'HDD': [hdd],
            'SSD': [ssd],
            'Gpu brand': [gpu],  # Double check if your notebook used 'Gpu' or 'Gpu brand'
            'os': [os]
        }
        query_df = pd.DataFrame(data_dict)

        # 4. Predict (Applying np.exp because the model was trained on log prices)
        prediction_log = pipe.predict(query_df)[0]
        prediction = int(np.exp(prediction_log))

        # 5. Return success to your JavaScript
        return jsonify({
            'success': True,
            'prediction': f"The predicted price is ₹{prediction:,}"
        })

    except Exception as e:
        # This prints the REAL error in your PyCharm terminal so you can see it
        print(f"DEBUG ERROR: {e}")
        return jsonify({'success': False, 'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True)
