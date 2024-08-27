import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping

data = {
    'GPA': [3.5, 3.8, 3.0, 3.9, 2.8, 3.7, 3.2, 3.6],
    'SAT_Score': [1200, 1350, 1100, 1400, 1050, 1300, 1150, 1250],
    'Recommendation_Score': [4, 5, 3, 5, 2, 4, 3, 4],
    'Extracurricular_Activities': [2, 3, 1, 4, 2, 3, 2, 3],
    'Admission_Status': [1, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# 입력 변수(X)와 출력 변수(y) 분리
X = df.drop('Admission_Status', axis=1)
y = df['Admission_Status']

# 학습용 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 생성 (더 많은 층과 드롭아웃 적용)
model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일 (조정된 학습률)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 모델 학습
history = model.fit(X_train, y_train, epochs=200, validation_split=0.2, verbose=1, callbacks=[early_stopping])

# 모델 평가
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'테스트 정확도: {test_acc:.2f}')

# 새로운 데이터에 대한 예측
new_data = [[3.3, 1250, 4, 3]]  # 새로운 지원자의 데이터 (GPA, SAT_Score, Recommendation_Score, Extracurricular_Activities)
new_data = scaler.transform(new_data)  # 데이터 정규화
prediction = model.predict(new_data)

# 예측 결과 출력
prediction_class = (prediction >= 0.5).astype(int)
prediction_percentage = prediction[0][0] * 100
print(f'합격 예측 확률: {prediction_percentage:.2f}%')
print(f'예측 결과 (합격 여부): {"합격" if prediction_class[0][0] == 1 else "불합격"}')
