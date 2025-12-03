"""
가전제품 판매 시계열 예측 분석 프로그램
- 머신러닝 모델: K-최근접 이웃(KNN) 회귀
- 딥러닝 모델: RNN (Recurrent Neural Network)
- 성능 비교 및 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 머신러닝 라이브러리
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# 딥러닝 라이브러리
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    print("경고: TensorFlow가 설치되지 않았습니다. 딥러닝 모델은 사용할 수 없습니다.")
    print("설치 방법: pip install tensorflow")
    DEEP_LEARNING_AVAILABLE = False


class ApplianceSalesPredictor:
    """가전제품 판매 예측 시스템"""

    def __init__(self, csv_path):
        """
        초기화 함수

        Parameters:
        -----------
        csv_path : str
            데이터 CSV 파일 경로
        """
        self.csv_path = csv_path
        self.df = None
        self.daily_sales = None
        self.scaler = MinMaxScaler()
        self.results = {}

    def load_and_prepare_data(self):
        """데이터 로드 및 전처리"""
        print("="*80)
        print("[1단계] 데이터 로드 및 전처리")
        print("="*80)

        # 데이터 로드
        self.df = pd.read_csv(self.csv_path)
        print(f"\n원본 데이터 크기: {self.df.shape}")
        print(f"컬럼: {list(self.df.columns)}")

        # 날짜 형식 변환
        self.df['날짜'] = pd.to_datetime(self.df['날짜'], format='%Y%m%d')

        # 일별 총 매출 집계
        self.daily_sales = self.df.groupby('날짜')['매출금액'].sum().reset_index()
        self.daily_sales.columns = ['날짜', '매출금액']
        self.daily_sales = self.daily_sales.sort_values('날짜').reset_index(drop=True)

        print(f"\n일별 집계 데이터 크기: {self.daily_sales.shape}")
        print(f"기간: {self.daily_sales['날짜'].min()} ~ {self.daily_sales['날짜'].max()}")
        print(f"평균 일 매출: {self.daily_sales['매출금액'].mean():,.0f}원")
        print(f"최대 일 매출: {self.daily_sales['매출금액'].max():,.0f}원")
        print(f"최소 일 매출: {self.daily_sales['매출금액'].min():,.0f}원")

        # 추가 피처 생성
        self.daily_sales['요일'] = self.daily_sales['날짜'].dt.dayofweek
        self.daily_sales['일'] = self.daily_sales['날짜'].dt.day
        self.daily_sales['주말여부'] = (self.daily_sales['요일'] >= 5).astype(int)

        return self.daily_sales

    def visualize_data(self):
        """데이터 시각화"""
        print("\n[데이터 시각화]")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 1. 일별 매출 추이
        axes[0, 0].plot(self.daily_sales['날짜'], self.daily_sales['매출금액'],
                       marker='o', linewidth=2, markersize=6, color='#2E86AB')
        axes[0, 0].set_title('가전제품 일별 매출 추이', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('날짜', fontsize=11)
        axes[0, 0].set_ylabel('매출금액 (원)', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. 요일별 평균 매출
        dow_sales = self.daily_sales.groupby('요일')['매출금액'].mean()
        dow_names = ['월', '화', '수', '목', '금', '토', '일']
        axes[0, 1].bar(range(7), dow_sales.values, color='#A23B72')
        axes[0, 1].set_title('요일별 평균 매출', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('요일', fontsize=11)
        axes[0, 1].set_ylabel('평균 매출금액 (원)', fontsize=11)
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(dow_names)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # 3. 매출 분포
        axes[1, 0].hist(self.daily_sales['매출금액'], bins=15, color='#F18F01', edgecolor='black')
        axes[1, 0].set_title('매출금액 분포', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('매출금액 (원)', fontsize=11)
        axes[1, 0].set_ylabel('빈도', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 4. 주말 vs 평일 박스플롯
        weekend_data = [
            self.daily_sales[self.daily_sales['주말여부'] == 0]['매출금액'].values,
            self.daily_sales[self.daily_sales['주말여부'] == 1]['매출금액'].values
        ]
        bp = axes[1, 1].boxplot(weekend_data, labels=['평일', '주말'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#6A994E', '#BC4749']):
            patch.set_facecolor(color)
        axes[1, 1].set_title('평일 vs 주말 매출 비교', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('매출금액 (원)', fontsize=11)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('01_데이터_시각화.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✓ 시각화 완료: 01_데이터_시각화.png 저장됨")

    def split_data(self, train_ratio=0.8):
        """학습/테스트 데이터 분리"""
        split_idx = int(len(self.daily_sales) * train_ratio)

        self.train_data = self.daily_sales[:split_idx].copy()
        self.test_data = self.daily_sales[split_idx:].copy()

        print(f"\n[데이터 분리]")
        print(f"학습 데이터: {len(self.train_data)}일 ({self.train_data['날짜'].min()} ~ {self.train_data['날짜'].max()})")
        print(f"테스트 데이터: {len(self.test_data)}일 ({self.test_data['날짜'].min()} ~ {self.test_data['날짜'].max()})")

        return self.train_data, self.test_data

    def calculate_metrics(self, y_true, y_pred, model_name):
        """성능 지표 계산"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # MAPE 계산 (0으로 나누기 방지)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }

        self.results[model_name] = {
            'predictions': y_pred,
            'metrics': metrics
        }

        return metrics

    def print_metrics(self, metrics, model_name):
        """성능 지표 출력"""
        print(f"\n{'='*60}")
        print(f"[{model_name} 성능 지표]")
        print(f"{'='*60}")
        print(f"MSE (Mean Squared Error)      : {metrics['MSE']:,.0f}")
        print(f"RMSE (Root Mean Squared Error): {metrics['RMSE']:,.0f}원")
        print(f"MAE (Mean Absolute Error)     : {metrics['MAE']:,.0f}원")
        print(f"R² Score                      : {metrics['R2']:.4f}")
        print(f"MAPE (Mean Absolute % Error)  : {metrics['MAPE']:.2f}%")
        print(f"{'='*60}")

    # ========================================================================
    # 머신러닝 모델: K-최근접 이웃(KNN) 회귀
    # ========================================================================

    def train_knn(self, n_neighbors=5):
        """K-최근접 이웃(KNN) 회귀 모델 학습 및 예측"""
        print("\n" + "="*80)
        print("[머신러닝 모델] K-최근접 이웃(KNN) 회귀")
        print("="*80)

        # 피처 준비 - 과거 N일의 매출 데이터를 피처로 사용
        lookback = 7  # 과거 7일 데이터 사용

        # 시계열 데이터를 지도학습 형태로 변환
        X_train_list = []
        y_train_list = []

        # 학습 데이터 생성
        sales_values = self.daily_sales['매출금액'].values
        for i in range(lookback, len(self.train_data)):
            X_train_list.append(sales_values[i-lookback:i])
            y_train_list.append(sales_values[i])

        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)

        # 테스트 데이터 생성
        X_test_list = []
        y_test_list = []
        test_start_idx = len(self.train_data)

        for i in range(test_start_idx, len(self.daily_sales)):
            if i >= lookback:
                X_test_list.append(sales_values[i-lookback:i])
                y_test_list.append(sales_values[i])

        X_test = np.array(X_test_list)
        y_test = np.array(y_test_list)

        print(f"\n학습 데이터 크기: {X_train.shape}")
        print(f"테스트 데이터 크기: {X_test.shape}")

        # 데이터 정규화 (KNN은 거리 기반이므로 스케일링 필수)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 최적의 K 값 찾기
        print(f"\n[다양한 K 값으로 모델 성능 테스트]")
        best_k = n_neighbors
        best_score = -np.inf
        k_values = [3, 5, 7, 9, 11]

        for k in k_values:
            temp_model = KNeighborsRegressor(n_neighbors=k)
            temp_model.fit(X_train_scaled, y_train)
            score = temp_model.score(X_train_scaled, y_train)
            print(f"K={k}: 학습 R² = {score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k

        print(f"\n✓ 최적의 K 값: {best_k}")

        # 최적 K로 최종 모델 학습
        model = KNeighborsRegressor(
            n_neighbors=best_k,
            weights='distance',  # 거리에 반비례하는 가중치 사용
            algorithm='auto',
            metric='euclidean'
        )
        model.fit(X_train_scaled, y_train)

        # 예측
        predictions = model.predict(X_test_scaled)

        # 성능 평가
        metrics = self.calculate_metrics(y_test, predictions, 'KNN')
        self.print_metrics(metrics, f'K-최근접 이웃 (K={best_k})')

        return predictions, metrics

    # ========================================================================
    # 딥러닝 모델: RNN (Recurrent Neural Network)
    # ========================================================================

    def create_sequences(self, data, seq_length):
        """시계열 시퀀스 생성 (딥러닝용)"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    def train_rnn(self, seq_length=7, epochs=150, batch_size=8):
        """RNN(Recurrent Neural Network) 모델 학습 및 예측"""
        if not DEEP_LEARNING_AVAILABLE:
            print("\n딥러닝 모델을 사용할 수 없습니다.")
            return None, None

        print("\n" + "="*80)
        print("[딥러닝 모델] RNN (Recurrent Neural Network)")
        print("="*80)

        # 데이터 정규화
        sales_data = self.daily_sales['매출금액'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(sales_data)

        # 시퀀스 생성
        X, y = self.create_sequences(scaled_data, seq_length)

        # 학습/테스트 분리
        split_idx = len(self.train_data) - seq_length
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]

        print(f"\n학습 데이터 shape: {X_train.shape}")
        print(f"테스트 데이터 shape: {X_test.shape}")

        # RNN 모델 구축 (2층 구조)
        print("\n[RNN 모델 아키텍처]")
        model = Sequential([
            # 첫 번째 RNN 레이어 (return_sequences=True로 다음 레이어로 시퀀스 전달)
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
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        # 모델 구조 출력
        print("\n[모델 요약]")
        model.summary()

        # 조기 종료 설정
        early_stop = EarlyStopping(
            monitor='loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )

        # 모델 학습
        print("\n[모델 학습 중...]")
        print(f"Epochs: {epochs}, Batch Size: {batch_size}")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )

        print(f"\n✓ 학습 완료 (실제 학습 Epochs: {len(history.history['loss'])})")

        # 예측
        print("\n[테스트 데이터 예측 중...]")
        predictions_scaled = model.predict(X_test, verbose=0)
        predictions = self.scaler.inverse_transform(predictions_scaled).flatten()
        y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # 성능 평가
        metrics = self.calculate_metrics(y_test_original, predictions, 'RNN')
        self.print_metrics(metrics, f'RNN (seq_length={seq_length})')

        # 결과 저장
        self.results['RNN']['history'] = history.history

        # 학습 곡선 시각화
        self.plot_training_history(history, 'RNN')

        return predictions, metrics

    def plot_training_history(self, history, model_name):
        """학습 과정 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss 그래프
        axes[0].plot(history.history['loss'], linewidth=2, color='#E63946')
        axes[0].set_title(f'{model_name} 모델 학습 손실(Loss)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=10)
        axes[0].set_ylabel('Loss (MSE)', fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # MAE 그래프
        axes[1].plot(history.history['mae'], linewidth=2, color='#2A9D8F')
        axes[1].set_title(f'{model_name} 모델 학습 MAE', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=10)
        axes[1].set_ylabel('MAE', fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'00_{model_name}_학습_곡선.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✓ 학습 곡선 저장: 00_{model_name}_학습_곡선.png")

    # ========================================================================
    # 결과 비교 및 시각화
    # ========================================================================

    def compare_all_models(self):
        """모든 모델 성능 비교"""
        print("\n" + "="*80)
        print("[최종 모델 성능 비교]")
        print("="*80)

        # 결과 테이블 생성
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                '모델': model_name,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R²': metrics['R2'],
                'MAPE(%)': metrics['MAPE']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('RMSE')

        print("\n")
        print(comparison_df.to_string(index=False))

        # 최고 성능 모델
        best_model = comparison_df.iloc[0]['모델']
        print(f"\n{'='*80}")
        print(f"🏆 최고 성능 모델: {best_model}")
        print(f"{'='*80}")

        return comparison_df

    def visualize_predictions(self):
        """예측 결과 시각화"""
        print("\n[예측 결과 시각화]")

        n_models = len(self.results)
        fig, axes = plt.subplots(n_models, 1, figsize=(16, 5*n_models))

        if n_models == 1:
            axes = [axes]

        y_true = self.test_data['매출금액'].values
        test_dates = self.test_data['날짜'].values

        for idx, (model_name, result) in enumerate(self.results.items()):
            y_pred = result['predictions']

            # 길이 맞추기 (딥러닝 모델의 경우)
            if len(y_pred) < len(y_true):
                y_true_plot = y_true[-len(y_pred):]
                test_dates_plot = test_dates[-len(y_pred):]
            else:
                y_true_plot = y_true
                test_dates_plot = test_dates

            axes[idx].plot(test_dates_plot, y_true_plot,
                          marker='o', linewidth=2, markersize=8,
                          label='실제 매출', color='#2E86AB')
            axes[idx].plot(test_dates_plot, y_pred[:len(y_true_plot)],
                          marker='s', linewidth=2, markersize=8,
                          label='예측 매출', color='#F18F01', linestyle='--')

            axes[idx].set_title(f'{model_name} 모델 - 실제 vs 예측',
                               fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('날짜', fontsize=11)
            axes[idx].set_ylabel('매출금액 (원)', fontsize=11)
            axes[idx].legend(fontsize=11)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].tick_params(axis='x', rotation=45)

            # R² 점수 표시
            r2 = result['metrics']['R2']
            mae = result['metrics']['MAE']
            axes[idx].text(0.02, 0.98, f"R² = {r2:.4f}\nMAE = {mae:,.0f}원",
                          transform=axes[idx].transAxes,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                          fontsize=10)

        plt.tight_layout()
        plt.savefig('02_모델_예측_비교.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✓ 시각화 완료: 02_모델_예측_비교.png 저장됨")

    def visualize_metrics_comparison(self, comparison_df):
        """성능 지표 비교 시각화"""
        print("\n[성능 지표 비교 시각화]")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        models = comparison_df['모델'].values

        # 1. RMSE 비교
        axes[0, 0].barh(models, comparison_df['RMSE'].values, color='#E63946')
        axes[0, 0].set_title('RMSE 비교 (낮을수록 좋음)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('RMSE', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3, axis='x')

        # 2. MAE 비교
        axes[0, 1].barh(models, comparison_df['MAE'].values, color='#F18F01')
        axes[0, 1].set_title('MAE 비교 (낮을수록 좋음)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('MAE', fontsize=11)
        axes[0, 1].grid(True, alpha=0.3, axis='x')

        # 3. R² 비교
        axes[1, 0].barh(models, comparison_df['R²'].values, color='#2A9D8F')
        axes[1, 0].set_title('R² Score 비교 (높을수록 좋음)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('R² Score', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        axes[1, 0].set_xlim([0, 1])

        # 4. MAPE 비교
        axes[1, 1].barh(models, comparison_df['MAPE(%)'].values, color='#A23B72')
        axes[1, 1].set_title('MAPE 비교 (낮을수록 좋음)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('MAPE (%)', fontsize=11)
        axes[1, 1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig('03_성능_지표_비교.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✓ 시각화 완료: 03_성능_지표_비교.png 저장됨")

    def generate_report(self, comparison_df):
        """최종 분석 리포트 생성"""
        print("\n" + "="*80)
        print("[최종 분석 리포트]")
        print("="*80)

        report = []
        report.append("="*80)
        report.append("가전제품 판매 시계열 예측 분석 리포트")
        report.append("="*80)
        report.append(f"\n분석 기간: {self.daily_sales['날짜'].min()} ~ {self.daily_sales['날짜'].max()}")
        report.append(f"총 데이터: {len(self.daily_sales)}일")
        report.append(f"학습 데이터: {len(self.train_data)}일")
        report.append(f"테스트 데이터: {len(self.test_data)}일")

        report.append("\n" + "-"*80)
        report.append("1. 데이터 기초 통계")
        report.append("-"*80)
        report.append(f"평균 일 매출: {self.daily_sales['매출금액'].mean():,.0f}원")
        report.append(f"표준편차: {self.daily_sales['매출금액'].std():,.0f}원")
        report.append(f"최대 매출: {self.daily_sales['매출금액'].max():,.0f}원")
        report.append(f"최소 매출: {self.daily_sales['매출금액'].min():,.0f}원")

        report.append("\n" + "-"*80)
        report.append("2. 모델 성능 비교")
        report.append("-"*80)
        report.append("\n" + comparison_df.to_string(index=False))

        report.append("\n" + "-"*80)
        report.append("3. 결론 및 제안")
        report.append("-"*80)

        best_model = comparison_df.iloc[0]
        worst_model = comparison_df.iloc[-1]

        report.append(f"\n🏆 최고 성능 모델: {best_model['모델']}")
        report.append(f"   - RMSE: {best_model['RMSE']:,.0f}원")
        report.append(f"   - MAE: {best_model['MAE']:,.0f}원")
        report.append(f"   - R²: {best_model['R²']:.4f}")
        report.append(f"   - MAPE: {best_model['MAPE(%)']:.2f}%")

        report.append(f"\n📊 최저 성능 모델: {worst_model['모델']}")
        report.append(f"   - RMSE: {worst_model['RMSE']:,.0f}원")
        report.append(f"   - MAE: {worst_model['MAE']:,.0f}원")
        report.append(f"   - R²: {worst_model['R²']:.4f}")
        report.append(f"   - MAPE: {worst_model['MAPE(%)']:.2f}%")

        # 모델별 특징 분석
        report.append("\n" + "-"*80)
        report.append("4. 모델별 특징 및 분석")
        report.append("-"*80)

        if 'KNN' in self.results:
            report.append("\n[K-최근접 이웃(KNN) 회귀]")
            report.append("- 거리 기반 비모수적 머신러닝 기법")
            report.append("- 과거 유사 패턴을 찾아 예측 수행")
            report.append("- 특징:")
            report.append("  * 직관적이고 이해하기 쉬운 알고리즘")
            report.append("  * 학습 시간이 짧고 구현이 간단함")
            report.append("  * 거리 기반이므로 정규화(스케일링) 필수")
            report.append("  * K값 선택이 성능에 큰 영향")
            report.append("- 장점: 비선형 패턴 포착, 빠른 학습")
            report.append("- 단점: 예측 시간이 상대적으로 느림, 차원의 저주")

        if 'RNN' in self.results:
            report.append("\n[RNN (Recurrent Neural Network - 순환 신경망)]")
            report.append("- 시퀀스 데이터 처리에 특화된 딥러닝 기법")
            report.append("- 이전 시점의 정보를 순환적으로 전달하여 시계열 패턴 학습")
            report.append("- 특징:")
            report.append("  * 시간적 의존성(temporal dependency) 학습 가능")
            report.append("  * 가변 길이 시퀀스 처리 가능")
            report.append("  * 은닉 상태(hidden state)로 과거 정보 기억")
            report.append("  * tanh 활성화 함수로 비선형성 확보")
            report.append("- 장점: 복잡한 시계열 패턴 포착, 자동 피처 추출")
            report.append("- 단점: 학습 시간이 길고, 하이퍼파라미터 튜닝 필요")
            report.append("- 구조: 3층 RNN (128→64→32 유닛) + Dropout + Dense 레이어")

        report.append("\n" + "-"*80)
        report.append("5. 머신러닝 vs 딥러닝 비교 분석")
        report.append("-"*80)
        report.append("\n[머신러닝: KNN]")
        report.append("✓ 학습 속도: 매우 빠름")
        report.append("✓ 해석 가능성: 높음 (최근접 이웃 확인 가능)")
        report.append("✓ 데이터 요구량: 상대적으로 적음")
        report.append("✓ 적용 난이도: 쉬움")

        report.append("\n[딥러닝: RNN]")
        report.append("✓ 학습 속도: 느림 (에폭 반복 필요)")
        report.append("✓ 해석 가능성: 낮음 (블랙박스 모델)")
        report.append("✓ 데이터 요구량: 많음")
        report.append("✓ 복잡도: 높음 (신경망 구조 설계 필요)")
        report.append("✓ 성능: 복잡한 패턴에서 우수")

        report.append("\n" + "-"*80)
        report.append("6. 권장사항 및 결론")
        report.append("-"*80)
        report.append(f"\n🏆 최고 성능 모델: {best_model['모델']}")

        if len(self.results) == 2:
            report.append("\n[모델 선택 가이드]")
            if best_model['모델'] == 'KNN':
                report.append("✓ KNN이 더 우수한 성능 → 빠른 학습과 해석성이 중요한 경우 추천")
                report.append("✓ 실시간 예측이 필요하거나 간단한 패턴의 경우 KNN 활용")
            else:
                report.append("✓ RNN이 더 우수한 성능 → 복잡한 시계열 패턴 존재")
                report.append("✓ 장기 예측이나 정확도가 최우선인 경우 RNN 활용")

        report.append("\n[향후 개선 방향]")
        report.append("✓ 정기적인 모델 재학습 (신규 데이터 반영)")
        report.append("✓ 외부 변수 추가 (프로모션, 공휴일, 날씨 등)")
        report.append("✓ 하이퍼파라미터 튜닝으로 성능 최적화")
        report.append("✓ 앙상블 기법 적용 (KNN + RNN 결합)")
        report.append("✓ 다양한 시계열 길이(lookback) 테스트")

        report.append("\n" + "="*80)
        report.append(f"리포트 생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)

        # 출력 및 저장
        report_text = "\n".join(report)
        print(report_text)

        with open('04_분석_리포트.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        print("\n✓ 리포트 저장 완료: 04_분석_리포트.txt")

        return report_text

    def run_full_analysis(self):
        """전체 분석 파이프라인 실행"""
        print("\n")
        print("█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + " " * 20 + "가전제품 판매 예측 분석 시스템" + " " * 20 + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80)
        print("\n")

        # 1. 데이터 로드 및 전처리
        self.load_and_prepare_data()
        self.visualize_data()

        # 2. 데이터 분리
        self.split_data(train_ratio=0.8)

        # 3. 머신러닝 모델 학습 (K-최근접 이웃)
        print("\n" + "█"*80)
        print("█" + " "*20 + "머신러닝 모델 학습: K-최근접 이웃" + " "*21 + "█")
        print("█"*80)

        self.train_knn()

        # 4. 딥러닝 모델 학습 (RNN)
        if DEEP_LEARNING_AVAILABLE:
            print("\n" + "█"*80)
            print("█" + " "*22 + "딥러닝 모델 학습: RNN (순환신경망)" + " "*21 + "█")
            print("█"*80)

            self.train_rnn(seq_length=7, epochs=150)
        else:
            print("\n" + "⚠"*40)
            print("경고: TensorFlow가 설치되지 않아 딥러닝 모델을 사용할 수 없습니다.")
            print("머신러닝 모델(KNN)만으로 분석을 진행합니다.")
            print("⚠"*40)

        # 5. 결과 비교
        comparison_df = self.compare_all_models()

        # 6. 시각화
        self.visualize_predictions()
        self.visualize_metrics_comparison(comparison_df)

        # 7. 리포트 생성
        self.generate_report(comparison_df)

        print("\n" + "█"*80)
        print("█" + " "*28 + "분석 완료!" + " "*35 + "█")
        print("█"*80)
        print("\n생성된 파일:")
        if DEEP_LEARNING_AVAILABLE:
            print("  - 00_RNN_학습_곡선.png")
        print("  - 01_데이터_시각화.png")
        print("  - 02_모델_예측_비교.png")
        print("  - 03_성능_지표_비교.png")
        print("  - 04_분석_리포트.txt")
        print("\n" + "█"*80 + "\n")


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    # CSV 파일 경로 설정
    csv_file = "card_gyeonggi_202503.csv"

    # 분석 시스템 초기화 및 실행
    predictor = ApplianceSalesPredictor(csv_file)
    predictor.run_full_analysis()
