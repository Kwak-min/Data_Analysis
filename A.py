import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------
# 1. 데이터 확인 (Load & Inspect)
# -----------------------------------------------------------
print("--- [1단계] 데이터 확인 ---")
df = pd.read_csv("card_gyeonggi_202503 - 복사본.csv")
print(df.head()) # 상위 5개 행 출력
print(df.info()) # 데이터 타입 및 결측치 확인

# 날짜 형식 변환
df['날짜'] = pd.to_datetime(df['날짜'], format='%Y%m%d')

# -----------------------------------------------------------
# 2. 데이터 분석 (Analyze)
# -----------------------------------------------------------
print("\n--- [2단계] 데이터 분석 ---")
# 요일별 평균 매출 확인 (0:월요일 ~ 6:일요일)
df['요일_코드'] = df['날짜'].dt.dayofweek
dow_mean = df.groupby('요일_코드')['매출금액'].mean()
print("요일별 평균 매출:")
print(dow_mean)
print(">> 분석 결과: 토요일(5) 매출이 가장 높으므로, '요일'이 중요한 예측 변수임을 확인했습니다.")

# 시각화: 일별 총 매출 추이
daily_sales = df.groupby('날짜')['매출금액'].sum()
plt.figure(figsize=(10, 5))
plt.plot(daily_sales.index, daily_sales.values, marker='o')
plt.title('3월 일별 총 매출 추이')
plt.grid(True)
plt.show()

# -----------------------------------------------------------
# 3. 모델 학습 및 생성 (Train & Build)
# -----------------------------------------------------------
print("\n--- [3단계] 모델 학습 및 생성 ---")

# 학습을 위해 데이터 집계 (날짜/업종/요일 -> 매출합계)
model_df = df.groupby(['날짜', '업종분류', '요일_코드'])['매출금액'].sum().reset_index()

# 업종명을 숫자로 변환 (Label Encoding)
le = LabelEncoder()
model_df['업종_숫자'] = le.fit_transform(model_df['업종분류'])

# 데이터 분리: 24일까지 학습(Train), 25일부터 평가(Test)
train_data = model_df[model_df['날짜'] <= '2025-03-24']
test_data = model_df[model_df['날짜'] > '2025-03-24']

# 입력 변수(X)와 정답 변수(y) 설정
X_train = train_data[['요일_코드', '업종_숫자']]
y_train = train_data['매출금액']

X_test = test_data[['요일_코드', '업종_숫자']]
y_test = test_data['매출금액']

# 랜덤 포레스트 모델 생성 및 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("모델 학습이 완료되었습니다.")

# -----------------------------------------------------------
# 4. 모델 평가 (Evaluate)
# -----------------------------------------------------------
print("\n--- [4단계] 모델 평가 ---")

# 테스트 데이터(25일~31일)에 대해 예측 수행
predictions = model.predict(X_test)

# 평가 지표 계산
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"모델 정확도 (R^2 Score): {r2:.4f} (1.0에 가까울수록 좋음)")
print(f"평균 오차 (MAE): {mae:,.0f}원")

# 실제값 vs 예측값 비교 시각화 (일부 구간)
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:50], label='실제 매출', marker='o')
plt.plot(predictions[:50], label='예측 매출', linestyle='--', marker='x')
plt.title('실제 매출 vs 예측 매출 비교 (부분 확대)')
plt.legend()
plt.show()