import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("KNN 회귀를 이용한 다음 날 매출금액 예측")
print("=" * 80)

# 1. 데이터 로드
print("\n[1단계] 데이터 로딩 중...")
df = pd.read_csv("electronics_final.csv", encoding='utf-8-sig')
print(f"완료: {df.shape[0]:,}개 행 로드됨")

# 날짜를 datetime으로 변환
col_date = df.columns[0]
col_amount = df.columns[1]
col_count = df.columns[2]

df[col_date] = pd.to_datetime(df[col_date], format='%Y%m%d')

# 일별 집계
daily_data = df.groupby(col_date).agg({
    col_amount: 'sum',
    col_count: 'sum'
}).reset_index()

daily_data = daily_data.sort_values(col_date).reset_index(drop=True)
print(f"일별 집계: {daily_data.shape[0]}일")
print(f"기간: {daily_data[col_date].min().strftime('%Y-%m-%d')} ~ {daily_data[col_date].max().strftime('%Y-%m-%d')}")

# 2. 시계열 데이터를 지도학습 형태로 변환
print("\n[2단계] 시계열 데이터 변환 중...")
lookback = 7  # 과거 7일 데이터로 다음 날 예측

sales_values = daily_data[col_amount].values
X_list = []
y_list = []

for i in range(lookback, len(sales_values)):
    X_list.append(sales_values[i-lookback:i])  # 과거 7일 매출
    y_list.append(sales_values[i])  # 다음 날 매출

X = np.array(X_list)
y = np.array(y_list)

print(f"생성된 데이터 크기: {X.shape[0]}개")
print(f"각 샘플: 과거 {lookback}일 매출 → 다음 날 매출 예측")

# 3. 학습/테스트 데이터 분리 (80:20)
print("\n[3단계] 학습/테스트 데이터 분리...")
train_size = int(len(X) * 0.8)

X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

print(f"학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")

# 4. 데이터 정규화 (KNN은 거리 기반이므로 필수)
print("\n[4단계] 데이터 정규화...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("완료: StandardScaler 적용")

# 5. 최적의 K 값 찾기
print("\n[5단계] 최적의 K 값 찾기...")
k_values = [1, 3, 5, 7, 10]
best_k = 5
best_score = -np.inf

for k in k_values:
    model = KNeighborsRegressor(n_neighbors=k, weights='distance')
    model.fit(X_train_scaled, y_train)
    score = model.score(X_train_scaled, y_train)
    print(f"  K={k:2d} | 학습 R² = {score:.4f}")
    if score > best_score:
        best_score = score
        best_k = k

print(f"\n✓ 최적의 K 값: {best_k}")

# 6. KNN 모델 학습
print(f"\n[6단계] KNN 모델 학습 (K={best_k})...")
knn_model = KNeighborsRegressor(n_neighbors=best_k, weights='distance')
knn_model.fit(X_train_scaled, y_train)
print("완료")

# 7. 예측
print("\n[7단계] 테스트 데이터 예측...")
predictions = knn_model.predict(X_test_scaled)
print("완료")

# 8. 성능 평가
print("\n[8단계] 모델 성능 평가")
print("=" * 60)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"RMSE (평균 제곱근 오차): {rmse:,.0f}원")
print(f"MAE  (평균 절대 오차)  : {mae:,.0f}원")
print(f"R²   (결정 계수)       : {r2:.4f}")
print("=" * 60)

# 9. 시각화
print("\n[9단계] 결과 시각화...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'KNN 회귀 (K={best_k})를 이용한 다음 날 매출금액 예측',
             fontsize=18, fontweight='bold', y=0.995)

# 날짜 정보
dates = daily_data[col_date].values
train_dates = dates[lookback:lookback+train_size]
test_dates = dates[lookback+train_size:]

# (1) 전체 데이터 추이
ax1 = axes[0, 0]
ax1.plot(train_dates, y_train, 'o-', color='blue', linewidth=2,
         markersize=6, label='학습 데이터 (실제값)', alpha=0.7)
ax1.plot(test_dates, y_test, 'o-', color='green', linewidth=2,
         markersize=6, label='테스트 데이터 (실제값)', alpha=0.7)
ax1.plot(test_dates, predictions, 's--', color='red', linewidth=2,
         markersize=6, label='테스트 데이터 (예측값)', alpha=0.7)
ax1.set_title('시계열 예측 결과', fontsize=14, fontweight='bold', pad=10)
ax1.set_xlabel('날짜', fontsize=12)
ax1.set_ylabel('매출금액 (원)', fontsize=12)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(axis='x', rotation=45, labelsize=10)

# (2) 실제값 vs 예측값 산점도
ax2 = axes[0, 1]
ax2.scatter(y_test, predictions, color='purple', s=100, alpha=0.6, edgecolor='black')
# 완벽한 예측선 (y=x)
min_val = min(y_test.min(), predictions.min())
max_val = max(y_test.max(), predictions.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='완벽한 예측')
ax2.set_title('실제값 vs 예측값', fontsize=14, fontweight='bold', pad=10)
ax2.set_xlabel('실제 매출금액 (원)', fontsize=12)
ax2.set_ylabel('예측 매출금액 (원)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')

# 성능 점수 표시
score_text = f'R² = {r2:.4f}\nRMSE = {rmse:,.0f}원\nMAE = {mae:,.0f}원'
ax2.text(0.05, 0.95, score_text, transform=ax2.transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.8',
         facecolor='lightblue', alpha=0.9), fontsize=11, fontweight='bold')

# (3) 예측 오차 분포
ax3 = axes[1, 0]
errors = y_test - predictions
ax3.hist(errors, bins=15, color='orange', edgecolor='black', alpha=0.75)
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='오차 = 0')
ax3.axvline(x=errors.mean(), color='blue', linestyle='--', linewidth=2,
            label=f'평균 오차 = {errors.mean():,.0f}원')
ax3.set_title('예측 오차 분포', fontsize=14, fontweight='bold', pad=10)
ax3.set_xlabel('오차 (실제값 - 예측값) (원)', fontsize=12)
ax3.set_ylabel('빈도', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

# (4) K 값에 따른 성능 비교
ax4 = axes[1, 1]
k_test_values = [1, 3, 5, 7, 10, 15]
train_scores = []
test_scores = []

for k in k_test_values:
    model = KNeighborsRegressor(n_neighbors=k, weights='distance')
    model.fit(X_train_scaled, y_train)
    train_scores.append(model.score(X_train_scaled, y_train))
    test_scores.append(model.score(X_test_scaled, y_test))

ax4.plot(k_test_values, train_scores, 'o-', linewidth=2.5, markersize=8,
         color='blue', label='학습 R²')
ax4.plot(k_test_values, test_scores, 's-', linewidth=2.5, markersize=8,
         color='green', label='테스트 R²')
ax4.axvline(x=best_k, color='red', linestyle='--', linewidth=2,
            label=f'최적 K = {best_k}')
ax4.set_title('K 값에 따른 모델 성능', fontsize=14, fontweight='bold', pad=10)
ax4.set_xlabel('K (이웃 개수)', fontsize=12)
ax4.set_ylabel('R² Score', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_xticks(k_test_values)

plt.tight_layout()
plt.savefig('KNN_시계열_예측_결과.png', dpi=300, bbox_inches='tight')
print("완료: 차트 저장 (KNN_시계열_예측_결과.png)")
plt.show()

# 10. 최종 요약
print("\n" + "=" * 80)
print("최종 분석 결과 요약")
print("=" * 80)
print(f"\n[모델 정보]")
print(f"  알고리즘: K-최근접 이웃 (KNN) 회귀")
print(f"  최적 K 값: {best_k}")
print(f"  입력: 과거 {lookback}일 매출 데이터")
print(f"  출력: 다음 날 매출금액 예측")

print(f"\n[데이터 정보]")
print(f"  전체 일수: {len(daily_data)}일")
print(f"  학습 데이터: {len(X_train)}일")
print(f"  테스트 데이터: {len(X_test)}일")

print(f"\n[성능 지표]")
print(f"  RMSE: {rmse:,.0f}원")
print(f"  MAE:  {mae:,.0f}원")
print(f"  R²:   {r2:.4f}")

print(f"\n[테스트 데이터 예측 결과]")
for i in range(min(5, len(y_test))):
    date = test_dates[i]
    actual = y_test[i]
    pred = predictions[i]
    error = actual - pred
    print(f"  {pd.Timestamp(date).strftime('%Y-%m-%d')} | 실제: {actual:>12,.0f}원 | 예측: {pred:>12,.0f}원 | 오차: {error:>10,.0f}원")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
print("\n생성된 파일: KNN_시계열_예측_결과.png")
print("\n이 프로그램은:")
print("  ✓ 과거 7일 매출 데이터로 다음 날 매출을 예측합니다")
print("  ✓ K-최근접 이웃(KNN) 회귀 알고리즘을 사용합니다")
print("  ✓ 비슷한 과거 패턴을 찾아서 예측하는 방식입니다")
print("\n" + "=" * 80)
