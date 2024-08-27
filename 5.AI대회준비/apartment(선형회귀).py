import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# 1. 데이터 준비 (가상의 데이터 사용)
data = {
    'size': [85, 120, 75, 60, 95, 110],
    'floor': [3, 15, 7, 10, 12, 8],
    'year_built': [2001, 2015, 1998, 2010, 2005, 2000],
    'location': ['A', 'B', 'A', 'C', 'B', 'C'],
    'price': [300000, 500000, 250000, 350000, 450000, 400000]
}

df = pd.DataFrame(data)

# 2. 데이터 전처리
encoder = OneHotEncoder(sparse_output=False)
encoded_location = encoder.fit_transform(df[['location']])
encoded_df = pd.DataFrame(encoded_location, columns=encoder.get_feature_names_out(['location']))

df = df.drop(['location'], axis=1)
df = pd.concat([df, encoded_df], axis=1)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('price', axis=1))
X = pd.DataFrame(scaled_features, columns=df.drop('price', axis=1).columns)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 모델 구축
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),  # 층 추가
    Dense(1)
])

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# 4. 모델 학습
model.fit(X_train, y_train, epochs=300, batch_size=2, validation_split=0.2)

# 5. 가상의 샘플 데이터 생성
sample_data = {
    'size': [100],
    'floor': [10],
    'year_built': [2010],
    'location': ['B']
}

sample_df = pd.DataFrame(sample_data)

# 6. 가상의 샘플 데이터 전처리
encoded_sample_location = encoder.transform(sample_df[['location']])
encoded_sample_df = pd.DataFrame(encoded_sample_location, columns=encoder.get_feature_names_out(['location']))

sample_df = sample_df.drop(['location'], axis=1)
sample_df = pd.concat([sample_df, encoded_sample_df], axis=1)

scaled_sample = scaler.transform(sample_df)

# 7. 모델을 사용하여 예측
predicted_price = model.predict(scaled_sample)
print(f"Predicted price for the sample data: {predicted_price[0][0]:,.2f} 만 원")

# 8. 테스트 데이터에 대한 예측 및 정확도 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print(f"Test MSE: {mse:.2f}")
print(f"Test R² score: {r2:.2f}")
print(f"Test MAPE: {mape:.2f}%")
print(f"Test Accuracy: {100 - mape:.2f}%")
