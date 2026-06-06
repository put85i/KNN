from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model dan scaler
with open('model_knn.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Nilai rata-rata dataset IBM HR untuk pembanding di grafik (Benchmark)
# Index: [Age, MonthlyIncome, DistanceFromHome, TotalWorkingYears, JobSatisfaction, YearsAtCompany]
avg_bertahan = [37.5, 6800, 8.9, 11.8, 2.8, 7.3]
avg_resign = [33.6, 4700, 10.6, 8.2, 2.4, 5.1]

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    user_input = None
    
    if request.method == 'POST':
        age = float(request.form['Age'])
        monthly_income = float(request.form['MonthlyIncome'])
        distance_from_home = float(request.form['DistanceFromHome'])
        total_working_years = float(request.form['TotalWorkingYears'])
        job_satisfaction = float(request.form['JobSatisfaction'])
        years_at_company = float(request.form['YearsAtCompany'])
        
        user_input = {
            'Age': age,
            'MonthlyIncome': monthly_income,
            'DistanceFromHome': distance_from_home,
            'TotalWorkingYears': total_working_years,
            'JobSatisfaction': job_satisfaction,
            'YearsAtCompany': years_at_company
        }
        
        input_data = np.array([[age, monthly_income, distance_from_home, total_working_years, job_satisfaction, years_at_company]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        if prediction == 1:
            prediction_result = "Karyawan Diprediksi akan RESIGN (Keluar)"
        else:
            prediction_result = "Karyawan Diprediksi akan BERTAHAN"
            
    return render_template('index.html', 
                           result=prediction_result, 
                           inputs=user_input,
                           avg_stay=avg_bertahan,
                           avg_leave=avg_resign)

if __name__ == '__main__':
    app.run(debug=True)