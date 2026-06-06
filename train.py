import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load Dataset IBM HR
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

# 2. Pilih Fitur Utama
features = ['Age', 'MonthlyIncome', 'DistanceFromHome', 'TotalWorkingYears', 'JobSatisfaction', 'YearsAtCompany']
X = df[features]

# 3. Preprocessing Target (Mengubah Attrition: Yes/No menjadi 1/0)
# Karena KNN hanya menerima angka numerik
y = df['Attrition'].map({'Yes': 1, 'No': 0})

# 4. Bagi data menjadi Training dan Testing (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Normalisasi Data (Sangat krusial karena Gaji ribuan sedangkan Usia puluhan)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Buat dan latih Model KNN (Menggunakan k=5)
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train_scaled, y_train)

# 7. Evaluasi Akurasi
y_pred = knn.predict(X_test_scaled)
print(f"Akurasi Model KNN: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 8. Simpan Model dan Scaler ke file .pkl
with open('model_knn.pkl', 'wb') as model_file:
    pickle.dump(knn, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model dan Scaler berhasil disimpan!")