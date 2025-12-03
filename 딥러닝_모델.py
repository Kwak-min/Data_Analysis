"""
가전제품 판매 예측 - 딥러닝 모델 (RNN)
누구나 이해하기 쉽게 만든 분석 프로그램
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# 딥러닝 라이브러리
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("오류: TensorFlow가 설치되지 않았습니다.")
    print("설치 방법: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False
    exit()

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
    print(f"[OK] 파일 로드 완료: {df.shape[0]:,}개 행")

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

    print(f"[OK] 일별 데이터: {len(daily_data)}일")
    print(f"[OK] 기간: {daily_data['날짜'].min().strftime('%Y-%m-%d')} ~ {daily_data['날짜'].max().strftime('%Y-%m-%d')}")
    print(f"[OK] 총 매출: {daily_data['매출금액'].sum():,.0f}원")

    return daily_data


def prepare_rnn_data(daily_data, lookback=7):
    """
    2단계: 딥러닝(RNN)용 데이터 준비

    설명:
    - RNN은 시퀀스 데이터를 처리하는 신경망입니다
    - 과거 7일의 매출 패턴을 학습하여 다음 날을 예측합니다
    """
    print_header("2단계: 딥러닝(RNN)용 데이터 준비")

    print(f"[RNN 특징]")
    print(f"  - Recurrent Neural Network (순환 신경망)")
    print(f"  - 시계열 데이터의 시간적 패턴을 학습합니다")
    print(f"  - 과거 정보를 '기억'하면서 순차적으로 처리합니다")
    print(f"  - 과거 {lookback}일 데이터로 다음 날 예측")

    # 매출금액 배열
    sales = daily_data['매출금액'].values.reshape(-1, 1)

    # 데이터 정규화 (0~1 범위로 변환)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    sales_scaled = scaler.fit_transform(sales)

    print(f"\n[데이터 정규화]")
    print(f"  - 원본 범위: {sales.min():,.0f}원 ~ {sales.max():,.0f}원")
    print(f"  - 정규화 범위: 0.0 ~ 1.0")
    print(f"  - 이유: 신경망 학습을 안정적으로 만들기 위해")

    # 시퀀스 데이터 생성
    X = []
    y = []

    for i in range(lookback, len(sales_scaled)):
        X.append(sales_scaled[i-lookback:i])  # 과거 7일
        y.append(sales_scaled[i])              # 다음 날

    X = np.array(X)
    y = np.array(y)

    print(f"\n[OK] 생성된 데이터: {len(X)}개 시퀀스")
    print(f"  - 입력(X) 형태: {X.shape} → (샘플수, 시퀀스길이, 특성수)")
    print(f"  - 출력(y) 형태: {y.shape} → (샘플수, 특성수)")

    return X, y, scaler


def split_train_test(X, y, train_ratio=0.8):
    """
    3단계: 학습/테스트 데이터 분리
    """
    print_header("3단계: 학습/테스트 데이터 분리")

    split_point = int(len(X) * train_ratio)

    X_train = X[:split_point]
    y_train = y[:split_point]
    X_test = X[split_point:]
    y_test = y[split_point:]

    print(f"[OK] 학습 데이터: {len(X_train)}개 ({train_ratio*100:.0f}%)")
    print(f"[OK] 테스트 데이터: {len(X_test)}개 ({(1-train_ratio)*100:.0f}%)")
    print(f"\n>> 왜 나누나요?")
    print(f"   - 학습 데이터로 신경망을 훈련시키고")
    print(f"   - 테스트 데이터로 실제 성능을 검증합니다")

    return X_train, y_train, X_test, y_test


def build_rnn_model(seq_length):
    """
    4단계: RNN 모델 구축

    설명:
    - SimpleRNN: 기본적인 순환 신경망 레이어
    - Dropout: 과적합 방지 (일부 뉴런을 무작위로 끔)
    - Dense: 완전연결 레이어 (최종 예측값 출력)
    """
    print_header("4단계: RNN 신경망 모델 구축")

    print("[RNN 아키텍처 설계]")
    print("  레이어 1: SimpleRNN(128) - 128개 뉴런, 시퀀스 유지")
    print("           └> Dropout(0.3) - 30% 뉴런 무작위 제거")
    print("  레이어 2: SimpleRNN(64)  - 64개 뉴런, 시퀀스 유지")
    print("           └> Dropout(0.3) - 30% 뉴런 무작위 제거")
    print("  레이어 3: SimpleRNN(32)  - 32개 뉴런")
    print("           └> Dropout(0.2) - 20% 뉴런 무작위 제거")
    print("  레이어 4: Dense(16)      - 16개 뉴런")
    print("  출력층:   Dense(1)       - 1개 출력 (매출 예측값)")

    model = Sequential([
        # 첫 번째 RNN 레이어
        SimpleRNN(128, activation='tanh', return_sequences=True,
                 input_shape=(seq_length, 1)),
        Dropout(0.3),

        # 두 번째 RNN 레이어
        SimpleRNN(64, activation='tanh', return_sequences=True),
        Dropout(0.3),

        # 세 번째 RNN 레이어
        SimpleRNN(32, activation='tanh'),
        Dropout(0.2),

        # 완전 연결 레이어
        Dense(16, activation='relu'),
        Dense(1)
    ])

    # 모델 컴파일
    model.compile(
        optimizer='adam',      # Adam 최적화 알고리즘
        loss='mse',            # 평균제곱오차 (Mean Squared Error)
        metrics=['mae']        # 평균절대오차 (Mean Absolute Error)
    )

    print(f"\n[OK] 신경망 모델 구축 완료")
    print(f"  - 총 레이어: 7개")
    print(f"  - 최적화: Adam")
    print(f"  - 손실함수: MSE (Mean Squared Error)")

    # 모델 요약
    print(f"\n[모델 구조 상세]")
    model.summary()

    return model


def train_rnn_model(model, X_train, y_train, epochs=150, batch_size=8):
    """
    5단계: RNN 모델 학습

    설명:
    - Epoch: 전체 학습 데이터를 한 번 학습하는 것
    - Batch Size: 한 번에 처리하는 데이터 개수
    - Early Stopping: 성능 개선이 없으면 조기 종료
    """
    print_header("5단계: RNN 모델 학습 (훈련)")

    print(f"[학습 설정]")
    print(f"  - Epochs: {epochs}회 (최대)")
    print(f"  - Batch Size: {batch_size}개")
    print(f"  - Early Stopping: 20회 개선 없으면 조기 종료")
    print(f"\n>> Epoch란?")
    print(f"   - 전체 학습 데이터를 1번 학습하는 것")
    print(f"   - {epochs}회 반복하면서 패턴을 학습합니다")

    # 조기 종료 설정
    early_stop = EarlyStopping(
        monitor='loss',           # 손실(loss)을 모니터링
        patience=20,              # 20회 개선 없으면 중단
        restore_best_weights=True,# 최적의 가중치로 복원
        verbose=1
    )

    print(f"\n[학습 시작...]")
    print(f"=" * 80)

    # 모델 학습
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1  # 학습 과정 출력
    )

    print(f"\n[OK] 학습 완료!")
    print(f"  - 실제 학습 Epochs: {len(history.history['loss'])}회")
    print(f"  - 최종 Loss: {history.history['loss'][-1]:.6f}")
    print(f"  - 최종 MAE: {history.history['mae'][-1]:.6f}")

    return history


def make_predictions(model, X_test, scaler):
    """
    6단계: 테스트 데이터로 예측
    """
    print_header("6단계: 테스트 데이터 예측")

    print("[예측 중...]")
    predictions_scaled = model.predict(X_test, verbose=0)

    # 정규화 해제 (원래 스케일로 복원)
    predictions = scaler.inverse_transform(predictions_scaled).flatten()

    print(f"[OK] {len(predictions)}개 예측 완료")
    print(f"  - 예측값 범위: {predictions.min():,.0f}원 ~ {predictions.max():,.0f}원")

    return predictions


def evaluate_model(y_test, predictions, scaler):
    """
    7단계: 모델 성능 평가
    """
    print_header("7단계: 모델 성능 평가")

    # 정규화 해제
    y_test_original = scaler.inverse_transform(y_test).flatten()

    # 성능 지표 계산
    mse = mean_squared_error(y_test_original, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, predictions)
    r2 = r2_score(y_test_original, predictions)
    mape = mean_absolute_percentage_error(y_test_original, predictions) * 100

    print("\n[성능 지표] (오차가 작을수록 좋음)")
    print("-" * 60)
    print(f"RMSE (Root Mean Squared Error)")
    print(f"  >> 평균 오차: {rmse:,.0f}원")
    print(f"  >> 의미: 평균적으로 실제값과 {rmse:,.0f}원 차이")

    print(f"\nMAE (Mean Absolute Error)")
    print(f"  >> 절대 오차: {mae:,.0f}원")
    print(f"  >> 의미: 절대값 기준 평균 {mae:,.0f}원 차이")

    print(f"\nR² Score (결정계수)")
    print(f"  >> 점수: {r2:.4f}")
    print(f"  >> 의미: 모델이 데이터의 {r2*100:.2f}% 설명")
    if r2 >= 0.8:
        print(f"  >> 평가: 매우 좋음! [***]")
    elif r2 >= 0.6:
        print(f"  >> 평가: 좋음 [**]")
    elif r2 >= 0.4:
        print(f"  >> 평가: 보통 [*]")
    else:
        print(f"  >> 평가: 개선 필요")

    print(f"\nMAPE (Mean Absolute Percentage Error)")
    print(f"  >> 퍼센트 오차: {mape:.2f}%")
    print(f"  >> 의미: 평균 {mape:.2f}% 정도 틀림")

    print("-" * 60)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'y_test_original': y_test_original
    }


def visualize_training_history(history):
    """
    8단계: 학습 과정 시각화
    """
    print_header("8단계: 학습 과정 시각화")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 그래프
    epochs_range = range(1, len(history.history['loss']) + 1)
    axes[0].plot(epochs_range, history.history['loss'], 'o-',
                 linewidth=2, markersize=4, color='#E63946')
    axes[0].set_title('학습 손실(Loss) 변화', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('Loss (MSE)', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # MAE 그래프
    axes[1].plot(epochs_range, history.history['mae'], 's-',
                 linewidth=2, markersize=4, color='#2A9D8F')
    axes[1].set_title('학습 MAE 변화', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=10)
    axes[1].set_ylabel('MAE', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('딥러닝_RNN_학습과정.png', dpi=300, bbox_inches='tight')
    print("[OK] 학습 과정 차트 저장: 딥러닝_RNN_학습과정.png")
    plt.show()


def visualize_results(daily_data, y_test_original, predictions, train_size, lookback, metrics, history):
    """
    9단계: 최종 결과 시각화
    """
    print_header("9단계: 최종 결과 시각화")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. 전체 데이터와 예측 결과
    ax1 = fig.add_subplot(gs[0, :])
    dates = daily_data['날짜'].values
    all_sales = daily_data['매출금액'].values

    # 학습 데이터 영역
    train_dates = dates[:train_size]
    train_sales = all_sales[:train_size]

    # 테스트 데이터 영역
    test_start_idx = train_size
    test_dates = dates[test_start_idx:test_start_idx+len(y_test_original)]

    ax1.plot(train_dates, train_sales, 'o-', color='#2E86AB',
             linewidth=2, markersize=6, label='학습 데이터', alpha=0.7)
    ax1.plot(test_dates, y_test_original, 'o-', color='#2A9D8F',
             linewidth=2, markersize=6, label='실제값 (테스트)')
    ax1.plot(test_dates, predictions, 's--', color='#E63946',
             linewidth=2, markersize=6, label='예측값 (RNN)')

    ax1.set_title('일별 매출 예측 결과 (RNN 딥러닝)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('날짜', fontsize=10)
    ax1.set_ylabel('매출금액 (원)', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # 2. 실제값 vs 예측값 산점도
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(y_test_original, predictions, alpha=0.6, s=100,
                color='#F18F01', edgecolor='black')

    # 완벽한 예측선
    min_val = min(y_test_original.min(), predictions.min())
    max_val = max(y_test_original.max(), predictions.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--',
             linewidth=2, label='완벽한 예측')

    ax2.set_title('실제값 vs 예측값', fontsize=12, fontweight='bold')
    ax2.set_xlabel('실제 매출금액 (원)', fontsize=10)
    ax2.set_ylabel('예측 매출금액 (원)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. 예측 오차 분포
    ax3 = fig.add_subplot(gs[1, 1])
    errors = y_test_original - predictions

    ax3.hist(errors, bins=15, color='#6A994E', edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='#E63946', linestyle='--', linewidth=2, label='오차=0')
    ax3.axvline(x=errors.mean(), color='#F18F01', linestyle='--', linewidth=2,
                label=f'평균 오차={errors.mean():,.0f}원')

    ax3.set_title('예측 오차 분포', fontsize=12, fontweight='bold')
    ax3.set_xlabel('오차 (실제 - 예측)', fontsize=10)
    ax3.set_ylabel('빈도', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 학습 과정 (Loss)
    ax4 = fig.add_subplot(gs[2, 0])
    epochs_range = range(1, len(history.history['loss']) + 1)
    ax4.plot(epochs_range, history.history['loss'], 'o-',
             linewidth=2, markersize=4, color='#E63946')
    ax4.set_title('학습 손실(Loss) 변화', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epoch', fontsize=10)
    ax4.set_ylabel('Loss (MSE)', fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 5. 성능 지표 요약
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    total_days = len(daily_data)
    test_days = len(y_test_original)
    actual_epochs = len(history.history['loss'])

    summary_lines = [
        "【 RNN 모델 성능 요약 】",
        "",
        "📊 데이터 정보",
        f"  • 전체 데이터: {total_days}일",
        f"  • 학습 데이터: {train_size}일 (80%)",
        f"  • 테스트 데이터: {test_days}일 (20%)",
        f"  • 시퀀스 길이: {lookback}일",
        "",
        "🧠 신경망 구조",
        f"  • RNN 레이어: 3층 (128→64→32)",
        f"  • Dropout: 과적합 방지",
        f"  • 학습 Epochs: {actual_epochs}회",
        "",
        "📈 성능 지표",
        f"  • RMSE: {metrics['RMSE']:,.0f} 원",
        f"  • MAE: {metrics['MAE']:,.0f} 원",
        f"  • R² Score: {metrics['R2']:.4f}",
        f"  • MAPE: {metrics['MAPE']:.2f}%",
        "",
        "📌 해석",
        f"  • 평균 {metrics['MAE']:,.0f}원 정도 오차",
        f"  • 데이터의 {metrics['R2']*100:.1f}% 설명",
        f"  • 퍼센트 오차 {metrics['MAPE']:.1f}%",
        "",
        "✅ RNN 모델 특징",
        "  • 시계열 패턴 학습",
        "  • 순환 구조로 기억",
        "  • 복잡한 패턴 포착",
        "  • 자동 특성 추출"
    ]

    summary_text = "\n".join(summary_lines)

    ax5.text(0.5, 0.5, summary_text,
             transform=ax5.transAxes,
             fontsize=9.5, verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round,pad=1', facecolor='#f0f0f0',
                      edgecolor='#2E86AB', linewidth=2, alpha=0.9),
             family='Malgun Gothic', linespacing=1.6)

    plt.suptitle('RNN (Recurrent Neural Network) 딥러닝 모델 분석 결과',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('딥러닝_RNN_분석결과.png', dpi=300, bbox_inches='tight')
    print("[OK] 최종 결과 차트 저장: 딥러닝_RNN_분석결과.png")
    plt.show()


def main():
    """
    메인 실행 함수
    """
    print("\n" + "="*80)
    print("  " + "가전제품 판매 예측 - RNN 딥러닝 모델")
    print("="*80)

    if not TENSORFLOW_AVAILABLE:
        print("\n[오류] TensorFlow가 설치되지 않아 실행할 수 없습니다.")
        return

    # CSV 파일 경로
    csv_file = "electronics_final.csv"

    # === 1단계: 데이터 불러오기 ===
    daily_data = load_data(csv_file)

    # === 2단계: RNN용 데이터 준비 ===
    lookback = 7  # 과거 7일 데이터 사용
    X, y, scaler = prepare_rnn_data(daily_data, lookback)

    # === 3단계: 학습/테스트 데이터 분리 ===
    X_train, y_train, X_test, y_test = split_train_test(X, y, train_ratio=0.8)

    # === 4단계: RNN 모델 구축 ===
    model = build_rnn_model(seq_length=lookback)

    # === 5단계: RNN 모델 학습 ===
    history = train_rnn_model(model, X_train, y_train, epochs=150, batch_size=8)

    # === 6단계: 테스트 데이터 예측 ===
    predictions = make_predictions(model, X_test, scaler)

    # === 7단계: 모델 성능 평가 ===
    metrics = evaluate_model(y_test, predictions, scaler)

    # === 8단계: 학습 과정 시각화 ===
    visualize_training_history(history)

    # === 9단계: 최종 결과 시각화 ===
    train_size = len(X_train) + lookback
    visualize_results(daily_data, metrics['y_test_original'], predictions,
                     train_size, lookback, metrics, history)

    # 완료 메시지
    print("\n" + "="*80)
    print("  분석 완료!")
    print("="*80)
    print("\n생성된 파일:")
    print("  - 딥러닝_RNN_학습과정.png")
    print("  - 딥러닝_RNN_분석결과.png")
    print("\n>> 이 프로그램은:")
    print("  [OK] 과거 7일 매출 패턴으로 다음 날 매출을 예측합니다")
    print("  [OK] RNN (순환신경망) 딥러닝 알고리즘을 사용합니다")
    print("  [OK] 시계열 데이터의 시간적 의존성을 학습합니다")
    print("  [OK] 복잡한 패턴도 자동으로 찾아내는 강력한 모델입니다")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
