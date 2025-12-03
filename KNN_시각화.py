import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
print("=" * 80)
print("가전제품 판매 데이터 KNN 회귀 분석")
print("=" * 80)
print("\n[데이터 로딩 중...]")

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

print(f"일별 집계: {daily_data.shape[0]}일")

# 매출건수를 X, 매출금액을 y로 사용
X = daily_data[col_count].values.reshape(-1, 1)  # 매출건수
y = daily_data[col_amount].values  # 매출금액

print(f"X (매출건수) 범위: {X.min():.0f} ~ {X.max():.0f}건")
print(f"y (매출금액) 범위: {y.min():,.0f} ~ {y.max():,.0f}원")

# 학습 데이터와 테스트 데이터 분리 (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 예측을 위한 범위 생성
X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

# 시각화 (2x2 서브플롯)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('KNN 회귀 분석: 소득률에 따른 총 자산 예측', fontsize=18, fontweight='bold', y=0.995)

# K 값들
k_values = [1, 5, 10]

# 1번째 그래프: 전체 데이터 산점도
axes[0, 0].scatter(X_train, y_train, color='blue', s=80, alpha=0.7, edgecolor='black', label='학습 데이터')
axes[0, 0].scatter(X_test, y_test, color='red', s=80, alpha=0.7, edgecolor='black', label='테스트 데이터')
axes[0, 0].set_title('전체 데이터 분포 (학습/테스트 분리)', fontsize=14, fontweight='bold', pad=10)
axes[0, 0].set_xlabel('매출건수 (건)', fontsize=12)
axes[0, 0].set_ylabel('매출금액 (원)', fontsize=12)
axes[0, 0].legend(fontsize=11, loc='upper left')
axes[0, 0].grid(True, alpha=0.3, linestyle='--')

# 데이터 개수 표시
axes[0, 0].text(0.95, 0.05, f'학습: {len(X_train)}개\n테스트: {len(X_test)}개',
                transform=axes[0, 0].transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8),
                fontsize=10)

# 2~4번째 그래프: K=1, 5, 10에 대한 예측 결과
subplot_positions = [(0, 1), (1, 0), (1, 1)]
colors_pred = ['green', 'purple', 'orange']

for idx, k in enumerate(k_values):
    ax = axes[subplot_positions[idx][0], subplot_positions[idx][1]]

    # KNN 모델 학습
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)

    # 예측
    y_pred_range = knn.predict(X_range)
    y_pred_test = knn.predict(X_test)

    # 성능 평가
    train_score = knn.score(X_train, y_train)
    test_score = knn.score(X_test, y_test)

    # 데이터와 예측선 그리기
    ax.scatter(X_train, y_train, color='blue', s=70, alpha=0.6, edgecolor='black', label='학습 데이터')
    ax.scatter(X_test, y_test, color='red', s=70, alpha=0.6, edgecolor='black', label='테스트 데이터')
    ax.plot(X_range, y_pred_range, color=colors_pred[idx], linewidth=3,
            label=f'예측선 (K={k})', alpha=0.8)

    # 테스트 데이터 예측값 표시
    ax.scatter(X_test, y_pred_test, color=colors_pred[idx], s=100, marker='x',
               linewidth=2.5, label='테스트 예측값', zorder=5)

    ax.set_title(f'K={k} 이웃 KNN 회귀', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('매출건수 (건)', fontsize=12)
    ax.set_ylabel('매출금액 (원)', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')

    # 성능 점수 표시
    score_text = f'학습 R² = {train_score:.4f}\n테스트 R² = {test_score:.4f}'
    ax.text(0.95, 0.05, score_text,
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9),
            fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('KNN_회귀_분석결과.png', dpi=300, bbox_inches='tight')
print("=" * 80)
print("KNN 회귀 분석 완료!")
print("=" * 80)
print(f"\n생성된 파일: KNN_회귀_분석결과.png")
print("\n차트 설명:")
print("  좌상단: 학습/테스트 데이터 분포")
print("  우상단: K=1 이웃 KNN 예측 (과적합 경향)")
print("  좌하단: K=5 이웃 KNN 예측 (균형)")
print("  우하단: K=10 이웃 KNN 예측 (평활화)")
print("\n" + "=" * 80)

# 상세 성능 비교
print("\n【 K 값에 따른 성능 비교 】\n")
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_score = knn.score(X_train, y_train)
    test_score = knn.score(X_test, y_test)
    print(f"K={k:2d} | 학습 R²: {train_score:.4f} | 테스트 R²: {test_score:.4f}")

print("\n" + "=" * 80)
plt.show()
