import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# 1. 데이터 준비
data = {
    'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'director': ['Director 1', 'Director 2', 'Director 1', 'Director 3', 'Director 2'],
    'genre': ['Action', 'Comedy', 'Action', 'Drama', 'Comedy'],
    'budget': [100000000, 50000000, 150000000, 30000000, 80000000],
    'runtime': [120, 90, 140, 110, 100],
    'audience': [1000000, 500000, 1500000, 300000, 800000]  # 실제 관객 수
}

df = pd.DataFrame(data)

# 2. 데이터 전처리
# 원-핫 인코딩으로 범주형 변수 변환
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[['director', 'genre']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['director', 'genre']))

# 원본 데이터프레임에서 범주형 열을 제거하고 인코딩된 열을 추가
df = df.drop(['title', 'director', 'genre'], axis=1)
df = pd.concat([df, encoded_df], axis=1)

# 특성 스케일링
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('audience', axis=1))
X = pd.DataFrame(scaled_features, columns=df.drop('audience', axis=1).columns)
y = df['audience']

# 데이터 분리 (훈련 세트와 테스트 세트로 나누기)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 모델 구축
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # 출력층: 관객 수를 예측하는 회귀 모델이므로 활성화 함수 없음
])

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

# 4. 모델 학습
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)

# 5. 예측 및 평가
y_pred = model.predict(X_test)
mse = MeanSquaredError()(y_test, y_pred).numpy()
rmse = np.sqrt(mse)

print(f'예측된 관객 수: {y_pred.flatten()}')
print(f'RMSE (Root Mean Squared Error): {rmse}')
