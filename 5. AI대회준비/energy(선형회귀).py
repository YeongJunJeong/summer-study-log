import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터 준비 (가상의 데이터 사용)
data = {
    'temperature': [25, 30, 22, 28, 32, 35, 31, 27, 29, 24],  # 기온 (도)
    'humidity': [60, 55, 70, 65, 50, 45, 55, 60, 50, 65],  # 습도 (%)
    'solar_radiation': [200, 300, 250, 280, 320, 350, 310, 270, 290, 240],  # 일사량 (W/m^2)
    'wind_speed': [3, 2.5, 3.2, 2.8, 3.5, 3.6, 3.1, 2.9, 3.3, 2.7],  # 풍속 (m/s)
    'power_output': [400, 600, 500, 550, 650, 700, 620, 560, 580, 460]  # 태양광 발전량 (kW)
}

df = pd.DataFrame(data)

# 2. 데이터 전처리
# 특성 스케일링
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('power_output', axis=1))
X = pd.DataFrame(scaled_features, columns=df.drop('power_output', axis=1).columns)
y = df['power_output']

# 데이터 분리 (훈련 세트와 테스트 세트로 나누기)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 모델 구축
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # 출력층: 태양광 발전량을 예측하는 회귀 모델이므로 활성화 함수 없음
])

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# 4. 모델 학습
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

# 5. 예측 및 평가
y_pred = model.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'예측된 태양광 발전량: {y_pred}')
print(f'RMSE (Root Mean Squared Error): {rmse}')
print(f'R² Score (예측 정확도): {r2}')
