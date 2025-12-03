"""
가전제품 판매 예측 - 머신러닝 모델 (K-최근접 이웃)
누구나 이해하기 쉽게 만든 분석 프로그램
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def print_header(title):
    """예쁘게 제목 출력"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def load_data(csv_file):
    """
    1단계: 데이터 불러오기
    """
    print_header("1단계: 데이터 불러오기")

    # CSV 파일 읽기
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    print(f"✓ 파일 로드 완료: {df.shape[0]:,}개 행")

    # 날짜 형식 변환
    col_date = df.columns[0]
    col_amount = df.columns[1]
    col_count = df.columns[2]

    df[col_date] = pd.to_datetime(df[col_date], format='%Y%m%d')

    # 일별로 집계
    daily_data = df.groupby(col_date).agg({
        col_amount: 'sum',
        col_count: 'sum'
    }).reset_index()

    daily_data.columns = ['날짜', '매출금액', '매출건수']

    print(f"✓ 일별 데이터: {len(daily_data)}일")
    print(f"✓ 기간: {daily_data['날짜'].min().strftime('%Y-%m-%d')} ~ {daily_data['날짜'].max().strftime('%Y-%m-%d')}")
    print(f"✓ 총 매출: {daily_data['매출금액'].sum():,.0f}원")

    return daily_data


def prepare_ml_data(daily_data, lookback=7):
    """
    2단계: 머신러닝용 데이터 준비

    설명:
    - 과거 7일의 매출 데이터를 보고 다음 날 매출을 예측합니다
    - 예) 3/1~3/7 매출로 3/8 매출 예측
    """
    print_header("2단계: 머신러닝용 데이터 준비")

    print(f"✓ 과거 {lookback}일 데이터로 다음 날 예측")
    print(f"  예시: 과거 7일 매출 → 다음 날 매출 예측")

    # 매출금액 배열
    sales = daily_data['매출금액'].values

    # 입력(X)과 출력(y) 데이터 만들기
    X = []  # 과거 7일 매출
    y = []  # 예측할 다음 날 매출

    for i in range(lookback, len(sales)):
        X.append(sales[i-lookback:i])  # 과거 7일
        y.append(sales[i])              # 다음 날

    X = np.array(X)
    y = np.array(y)

    print(f"✓ 생성된 데이터: {len(X)}개 샘플")
    print(f"  - 입력(X) 형태: {X.shape} → (샘플수, 과거 {lookback}일)")
    print(f"  - 출력(y) 형태: {y.shape} → (샘플수,)")

    return X, y


def split_train_test(X, y, train_ratio=0.8):
    """
    3단계: 학습/테스트 데이터 분리

    설명:
    - 학습 데이터(80%): 모델을 학습시킬 데이터
    - 테스트 데이터(20%): 모델 성능을 평가할 데이터
    """
    print_header("3단계: 학습/테스트 데이터 분리")

    split_point = int(len(X) * train_ratio)

    X_train = X[:split_point]
    y_train = y[:split_point]
    X_test = X[split_point:]
    y_test = y[split_point:]

    print(f"✓ 학습 데이터: {len(X_train)}개 ({train_ratio*100:.0f}%)")
    print(f"✓ 테스트 데이터: {len(X_test)}개 ({(1-train_ratio)*100:.0f}%)")
    print(f"\n💡 왜 나누나요?")
    print(f"   - 학습 데이터로 모델을 만들고")
    print(f"   - 테스트 데이터로 실제 성능을 확인합니다")

    return X_train, y_train, X_test, y_test


def normalize_data(X_train, X_test):
    """
    4단계: 데이터 정규화 (스케일링)

    설명:
    - KNN은 거리를 계산하므로 데이터 스케일을 맞춰야 합니다
    - 모든 데이터를 같은 범위로 변환합니다
    """
    print_header("4단계: 데이터 정규화")

    print("✓ StandardScaler 사용")
    print("  - 평균=0, 표준편차=1로 변환")
    print("  - 모든 특성을 동일한 스케일로 맞춤")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"✓ 정규화 완료")

    return X_train_scaled, X_test_scaled


def find_best_k(X_train_scaled, y_train):
    """
    5단계: 최적의 K 값 찾기

    설명:
    - K-최근접 이웃에서 K는 '가장 가까운 이웃 몇 개를 볼지' 결정
    - K=5면 가장 비슷한 5개 데이터를 참고해서 예측
    """
    print_header("5단계: 최적의 K 값 찾기")

    print("💡 K-최근접 이웃(KNN)이란?")
    print("   - 비슷한 과거 데이터를 찾아서 예측하는 방법")
    print("   - K=5 → 가장 비슷한 5개 데이터를 참고")
    print("   - K가 너무 작으면: 민감하게 반응 (과적합)")
    print("   - K가 너무 크면: 둔감하게 반응 (과소적합)")

    k_values = [3, 5, 7, 9, 11]
    best_k = 5
    best_score = -np.inf

    print(f"\n다양한 K 값 테스트:")
    for k in k_values:
        model = KNeighborsRegressor(n_neighbors=k, weights='distance')
        model.fit(X_train_scaled, y_train)
        score = model.score(X_train_scaled, y_train)

        print(f"  K={k:2d} → 점수(R²): {score:.4f}", end="")
        if score > best_score:
            best_score = score
            best_k = k
            print("  ⭐ (최고!)")
        else:
            print()

    print(f"\n✓ 최적의 K 값: {best_k}")

    return best_k


def train_knn_model(X_train_scaled, y_train, best_k):
    """
    6단계: KNN 모델 학습
    """
    print_header("6단계: KNN 모델 학습")

    print(f"✓ K={best_k}로 모델 학습 중...")

    model = KNeighborsRegressor(
        n_neighbors=best_k,
        weights='distance',  # 가까울수록 더 많이 반영
        algorithm='auto',
        metric='euclidean'   # 유클리드 거리 사용
    )

    model.fit(X_train_scaled, y_train)

    print(f"✓ 학습 완료!")
    print(f"\n💡 모델 설정:")
    print(f"   - K={best_k} (가장 가까운 {best_k}개 이웃 참고)")
    print(f"   - weights='distance' (가까운 이웃에 더 높은 가중치)")
    print(f"   - metric='euclidean' (유클리드 거리로 유사도 계산)")

    return model


def make_predictions(model, X_test_scaled):
    """
    7단계: 테스트 데이터로 예측
    """
    print_header("7단계: 테스트 데이터 예측")

    print("✓ 예측 중...")
    predictions = model.predict(X_test_scaled)

    print(f"✓ {len(predictions)}개 예측 완료")

    return predictions


def evaluate_model(y_test, predictions):
    """
    8단계: 모델 성능 평가

    설명:
    - RMSE: 평균적으로 얼마나 틀렸는지 (작을수록 좋음)
    - MAE: 절대 오차 평균 (작을수록 좋음)
    - R²: 설명력 (1에 가까울수록 좋음, 0~1)
    - MAPE: 퍼센트 오차 (작을수록 좋음)
    """
    print_header("8단계: 모델 성능 평가")

    # 성능 지표 계산
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions) * 100

    print("\n📊 성능 지표 (오차가 작을수록 좋음)")
    print("-" * 60)
    print(f"RMSE (Root Mean Squared Error)")
    print(f"  → 평균 오차: {rmse:,.0f}원")
    print(f"  → 의미: 평균적으로 실제값과 {rmse:,.0f}원 차이")

    print(f"\nMAE (Mean Absolute Error)")
    print(f"  → 절대 오차: {mae:,.0f}원")
    print(f"  → 의미: 절대값 기준 평균 {mae:,.0f}원 차이")

    print(f"\nR² Score (결정계수)")
    print(f"  → 점수: {r2:.4f}")
    print(f"  → 의미: 모델이 데이터의 {r2*100:.2f}% 설명")
    if r2 >= 0.8:
        print(f"  → 평가: 매우 좋음! ⭐⭐⭐")
    elif r2 >= 0.6:
        print(f"  → 평가: 좋음 ⭐⭐")
    elif r2 >= 0.4:
        print(f"  → 평가: 보통 ⭐")
    else:
        print(f"  → 평가: 개선 필요")

    print(f"\nMAPE (Mean Absolute Percentage Error)")
    print(f"  → 퍼센트 오차: {mape:.2f}%")
    print(f"  → 의미: 평균 {mape:.2f}% 정도 틀림")

    print("-" * 60)

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}


def visualize_results(daily_data, y_test, predictions, train_size, lookback):
    """
    9단계: 결과 시각화
    """
    print_header("9단계: 결과 시각화")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('K-최근접 이웃(KNN) 머신러닝 모델 분석 결과', fontsize=16, fontweight='bold')

    # 1. 전체 데이터와 예측 결과
    ax1 = axes[0, 0]
    dates = daily_data['날짜'].values
    all_sales = daily_data['매출금액'].values

    # 학습 데이터 영역
    train_dates = dates[:train_size]
    train_sales = all_sales[:train_size]

    # 테스트 데이터 영역
    test_start_idx = train_size
    test_dates = dates[test_start_idx:test_start_idx+len(y_test)]

    ax1.plot(train_dates, train_sales, 'o-', color='#2E86AB',
             linewidth=2, markersize=6, label='학습 데이터', alpha=0.7)
    ax1.plot(test_dates, y_test, 'o-', color='#2A9D8F',
             linewidth=2, markersize=6, label='실제값 (테스트)')
    ax1.plot(test_dates, predictions, 's--', color='#E63946',
             linewidth=2, markersize=6, label='예측값 (KNN)')

    ax1.set_title('일별 매출 예측 결과', fontsize=12, fontweight='bold')
    ax1.set_xlabel('날짜', fontsize=10)
    ax1.set_ylabel('매출금액 (원)', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # 2. 실제값 vs 예측값 산점도
    ax2 = axes[0, 1]
    ax2.scatter(y_test, predictions, alpha=0.6, s=100, color='#F18F01', edgecolor='black')

    # 완벽한 예측선 (y=x)
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='완벽한 예측')

    ax2.set_title('실제값 vs 예측값', fontsize=12, fontweight='bold')
    ax2.set_xlabel('실제 매출금액 (원)', fontsize=10)
    ax2.set_ylabel('예측 매출금액 (원)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. 예측 오차 분포
    ax3 = axes[1, 0]
    errors = y_test - predictions

    ax3.hist(errors, bins=15, color='#6A994E', edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='#E63946', linestyle='--', linewidth=2, label='오차=0')
    ax3.axvline(x=errors.mean(), color='#F18F01', linestyle='--', linewidth=2,
                label=f'평균 오차={errors.mean():,.0f}원')

    ax3.set_title('예측 오차 분포', fontsize=12, fontweight='bold')
    ax3.set_xlabel('오차 (실제 - 예측)', fontsize=10)
    ax3.set_ylabel('빈도', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 성능 지표 요약
    ax4 = axes[1, 1]
    ax4.axis('off')

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions) * 100

    # 텍스트 박스 내용
    total_days = len(daily_data)
    test_days = len(y_test)
    r2_pct = r2*100
    mape_val = mape

    summary_lines = [
        "【 KNN 모델 성능 요약 】",
        "",
        "📊 데이터 정보",
        f"  • 전체 데이터: {total_days}일",
        f"  • 학습 데이터: {train_size}일 (80%)",
        f"  • 테스트 데이터: {test_days}일 (20%)",
        f"  • 과거 참고 일수: {lookback}일",
        "",
        "📈 성능 지표",
        f"  • RMSE: {rmse:,.0f} 원",
        f"  • MAE: {mae:,.0f} 원",
        f"  • R² Score: {r2:.4f}",
        f"  • MAPE: {mape:.2f}%",
        "",
        "📌 해석",
        f"  • 평균 {mae:,.0f}원 정도 오차",
        f"  • 데이터의 {r2_pct:.1f}% 설명",
        f"  • 퍼센트 오차 {mape_val:.1f}%",
        "",
        "✅ KNN 모델 특징",
        "  • 거리 기반 예측",
        "  • 직관적이고 이해하기 쉬움",
        "  • 학습 속도 빠름",
        "  • 과거 유사 패턴 활용"
    ]

    summary_text = "\n".join(summary_lines)

    ax4.text(0.5, 0.5, summary_text,
             transform=ax4.transAxes,
             fontsize=10, verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round,pad=1', facecolor='#f0f0f0',
                      edgecolor='#2E86AB', linewidth=2, alpha=0.9),
             family='Malgun Gothic', linespacing=1.6)

    plt.tight_layout()
    plt.savefig('머신러닝_KNN_분석결과.png', dpi=300, bbox_inches='tight')
    print("✓ 차트 저장 완료: 머신러닝_KNN_분석결과.png")
    plt.show()


def main():
    """
    메인 실행 함수
    """
    print("\n" + "="*80)
    print("  " + "가전제품 판매 예측 - KNN 머신러닝 모델")
    print("="*80)

    # CSV 파일 경로
    csv_file = "electronics_final.csv"

    # === 1단계: 데이터 불러오기 ===
    daily_data = load_data(csv_file)

    # === 2단계: 머신러닝용 데이터 준비 ===
    lookback = 7  # 과거 7일 데이터 사용
    X, y = prepare_ml_data(daily_data, lookback)

    # === 3단계: 학습/테스트 데이터 분리 ===
    X_train, y_train, X_test, y_test = split_train_test(X, y, train_ratio=0.8)

    # === 4단계: 데이터 정규화 ===
    X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)

    # === 5단계: 최적의 K 값 찾기 ===
    best_k = find_best_k(X_train_scaled, y_train)

    # === 6단계: KNN 모델 학습 ===
    model = train_knn_model(X_train_scaled, y_train, best_k)

    # === 7단계: 테스트 데이터 예측 ===
    predictions = make_predictions(model, X_test_scaled)

    # === 8단계: 모델 성능 평가 ===
    metrics = evaluate_model(y_test, predictions)

    # === 9단계: 결과 시각화 ===
    train_size = len(X_train) + lookback
    visualize_results(daily_data, y_test, predictions, train_size, lookback)

    # 완료 메시지
    print("\n" + "="*80)
    print("  분석 완료!")
    print("="*80)
    print("\n생성된 파일:")
    print("  - 머신러닝_KNN_분석결과.png")
    print("\n💡 이 프로그램은:")
    print("  ✓ 과거 7일 매출 데이터로 다음 날 매출을 예측합니다")
    print("  ✓ K-최근접 이웃(KNN) 알고리즘을 사용합니다")
    print("  ✓ 비슷한 과거 패턴을 찾아서 예측하는 방식입니다")
    print("  ✓ 직관적이고 이해하기 쉬운 머신러닝 모델입니다")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
