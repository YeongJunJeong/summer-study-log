import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #cikit-learn에서 제공하는 데이터 전처리 도구로, 
#데이터의 평균을 0으로 하고 표준편차를 1로 맞추는 표준화 작업을 수행합니다. 
# 이를 통해 데이터의 스케일이 통일되어 모델 학습이 더 효과적이고 안정적이게 됩니다.
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping

# Excel 파일에서 데이터를 읽어오기
data = pd.read_excel("신용등급예측 데이터(1772개) 2016년도.xlsx")

# 필요한 열만 선택하여 새로운 DataFrame 생성
selected_columns = data[['신용평점', '우량불량분류(종속)', '유동비율', "당좌비율", "부채비율"]]

# 선택한 열로 새로운 DataFrame 생성
df = pd.DataFrame(selected_columns)

#모델 학습 시 데이터의 역할을 명확히 하기 위해서 
# 입력 변수(X)와 출력 변수(y) 분리
X = df.drop('우량불량분류(종속)', axis=1)
y = df['우량불량분류(종속)']

#테스트 세트를 20%로 지정하고 데이터 분할의 랜덤 시드를 분할함
#테스트 세트의 비율을 줄이면 훈련 데이터가 많아져서 모델이 더 잘 학습할 수 있지만, 
# 테스트 성능의 신뢰도가 떨어질 수 있습니다. 평가 데이터가 적어지기 때문에 
# 모델의 실제 성능을 정확히 측정하기 어려울 수 있습니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 정규화
scaler = StandardScaler() #데이터의 평균과 표준편차를 계산하여 데이터의 표준화를 수행하는 데 사용됩니다.
X_train = scaler.fit_transform(X_train) #훈련 데이터의 평균과 표준편차를 계산하고, 이를 사용해 데이터를 표준화합니다.
X_test = scaler.transform(X_test)#fit에서 계산된 평균과 표준편차를 사용해 새로운 데이터(테스트 데이터)를 표준화합니다.

model = models.Sequential([#models.Sequential: Keras에서 제공하는 신경망 모델의 기본 클래스입니다. 레이어를 순차적으로 쌓아서 모델을 구성합니다. 각 레이어는 이전 레이어의 출력을 입력으로 받습니다.
    #32개의 뉴런, 입력값이 0보다 크면 그대로 통과시키고, 
    # 0 이하의 값은 0으로 변환하는 ReLU(Rectified Linear Unit) 활성화 함수 사용
    
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일: 최적화 도구 및 손실 함수 설정
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습: 학습 데이터 중 20%를 검증용으로 사용
history = model.fit(X_train, y_train, epochs=200, validation_split=0.2, verbose=1)


# 새로운 데이터에 대한 예측
new_data = [[5, 110, 60, 40]] 
new_data = scaler.transform(new_data) 
prediction = model.predict(new_data)

# 예측 결과 출력
prediction_class = (prediction >= 0.5).astype(int)
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f'예측 결과 (우량,불량 여부): {"우량" if prediction_class[0][0] == 1 else "불량"}')
print(f'테스트 데이터셋에 대한 정확도 : {accuracy * 100:.2f}%')