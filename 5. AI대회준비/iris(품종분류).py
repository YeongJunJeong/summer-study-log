import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터 로드 및 전처리
iris = load_iris()
X = iris.data
y = iris.target

# OneHotEncoder를 사용하여 타겟 변수를 원-핫 인코딩
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 표준화 (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, input_shape=(4,), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. 모델 훈련
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# 4. 모델 평가
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Train accuracy: {train_acc:.4f}')
print(f'Test accuracy: {test_acc:.4f}')

# 5. 모델 예측 및 정확도 계산
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 원-핫 인코딩을 다시 정수형 클래스로 변환
y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)
y_train_pred_labels = np.argmax(y_train_pred, axis=1)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)

# 훈련 데이터 및 테스트 데이터 정확도
train_accuracy = accuracy_score(y_train_labels, y_train_pred_labels)
test_accuracy = accuracy_score(y_test_labels, y_test_pred_labels)

print(f'Training Accuracy: {train_accuracy:.4f}')
print(f'Testing Accuracy: {test_accuracy:.4f}')

# 6. 샘플 데이터 예측
sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # 임의의 샘플 데이터
sample_data = scaler.transform(sample_data)  # 동일하게 스케일링
prediction = model.predict(sample_data)
predicted_class = np.argmax(prediction, axis=1)

# 품종 이름 출력
class_names = iris.target_names
print(f'Predicted class: {class_names[predicted_class[0]]}')
