import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def visualize_electronics_data(csv_file):
    """
    가전제품 판매 데이터 시각화

    Parameters:
    -----------
    csv_file : str
        CSV 파일 경로 (electronics_final.csv)
    """
    print("="*80)
    print("가전제품 판매 데이터 시각화")
    print("="*80)

    # 데이터 로드
    print("\n[데이터 로딩 중...]")
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    print(f"완료: {df.shape[0]:,}개 행 로드됨")

    # 날짜 변환
    col_date = df.columns[0]
    col_amount = df.columns[1]
    col_count = df.columns[2]

    df[col_date] = pd.to_datetime(df[col_date], format='%Y%m%d')

    # 일별 집계
    daily_data = df.groupby(col_date).agg({
        col_amount: 'sum',
        col_count: 'sum'
    }).reset_index()

    print(f"\n일별 집계 데이터: {daily_data.shape[0]}일")
    print(f"기간: {daily_data[col_date].min().strftime('%Y년 %m월 %d일')} ~ {daily_data[col_date].max().strftime('%Y년 %m월 %d일')}")

    # 차트 생성
    print("\n[차트 생성 중...]")

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle('가전제품 판매 데이터 분석', fontsize=20, fontweight='bold', y=0.995)

    # 통계 계산
    avg_amount = daily_data[col_amount].mean()
    max_amount = daily_data[col_amount].max()
    min_amount = daily_data[col_amount].min()
    avg_count = daily_data[col_count].mean()

    # 1. 일별 매출금액 추이
    axes[0, 0].plot(daily_data[col_date], daily_data[col_amount],
                    marker='o', linewidth=2.5, markersize=7, color='#2E86AB', label='일별 매출')
    axes[0, 0].axhline(y=avg_amount, color='#E63946', linestyle='--',
                       label=f'평균: {avg_amount:,.0f}원', linewidth=2.5)
    axes[0, 0].axhline(y=max_amount, color='#2A9D8F', linestyle=':',
                       label=f'최대: {max_amount:,.0f}원', linewidth=2)
    axes[0, 0].axhline(y=min_amount, color='#F18F01', linestyle=':',
                       label=f'최소: {min_amount:,.0f}원', linewidth=2)
    axes[0, 0].set_title('일별 매출금액 추이', fontsize=14, fontweight='bold', pad=10)
    axes[0, 0].set_xlabel('날짜', fontsize=12)
    axes[0, 0].set_ylabel('매출금액 (원)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    axes[0, 0].tick_params(axis='x', rotation=45, labelsize=10)
    axes[0, 0].tick_params(axis='y', labelsize=10)
    axes[0, 0].legend(fontsize=10, loc='upper left')

    # 2. 요일별 평균 매출 (가장 중요한 차트)
    daily_data['요일'] = daily_data[col_date].dt.dayofweek
    dow_sales = daily_data.groupby('요일')[col_amount].mean()
    dow_names = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']

    colors = ['#E63946', '#F18F01', '#F4A261', '#2A9D8F', '#264653', '#E76F51', '#BC4749']
    bars = axes[0, 1].bar(range(7), dow_sales.values, color=colors, edgecolor='black',
                          alpha=0.85, linewidth=1.5)
    axes[0, 1].set_title('요일별 평균 매출', fontsize=14, fontweight='bold', pad=10)
    axes[0, 1].set_xlabel('요일', fontsize=12)
    axes[0, 1].set_ylabel('평균 매출금액 (원)', fontsize=12)
    axes[0, 1].set_xticks(range(7))
    axes[0, 1].set_xticklabels(dow_names, fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y', linestyle='--')

    # 막대 위에 값 표시
    for i, (bar, v) in enumerate(zip(bars, dow_sales.values)):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{v/1000000:.1f}M', ha='center', va='bottom',
                       fontsize=10, fontweight='bold')

    # 3. 주요 통계 정보 (텍스트 박스)
    axes[1, 0].axis('off')

    stats_text = f"""
    【 주요 통계 정보 】

    📊 매출금액
      • 총 매출: {daily_data[col_amount].sum():,.0f} 원
      • 일 평균: {avg_amount:,.0f} 원
      • 최대값: {max_amount:,.0f} 원
      • 최소값: {min_amount:,.0f} 원
      • 표준편차: {daily_data[col_amount].std():,.0f} 원

    📈 매출건수
      • 총 건수: {daily_data[col_count].sum():,} 건
      • 일 평균: {avg_count:.1f} 건
      • 최대값: {daily_data[col_count].max():,} 건
      • 최소값: {daily_data[col_count].min():,} 건

    📅 분석 기간
      • 시작일: {daily_data[col_date].min().strftime('%Y년 %m월 %d일')}
      • 종료일: {daily_data[col_date].max().strftime('%Y년 %m월 %d일')}
      • 총 일수: {len(daily_data)} 일

    🏆 요일별 분석
      • 최고: {dow_names[dow_sales.argmax()]} ({dow_sales.max():,.0f}원)
      • 최저: {dow_names[dow_sales.argmin()]} ({dow_sales.min():,.0f}원)
    """

    axes[1, 0].text(0.5, 0.5, stats_text,
                    transform=axes[1, 0].transAxes,
                    fontsize=11, verticalalignment='center', horizontalalignment='center',
                    bbox=dict(boxstyle='round,pad=1', facecolor='#f0f0f0',
                             edgecolor='#2E86AB', linewidth=3, alpha=0.9),
                    family='Malgun Gothic', linespacing=1.8)

    # 4. 일별 매출건수 추이
    axes[1, 1].plot(daily_data[col_date], daily_data[col_count],
                    marker='s', linewidth=2.5, markersize=7, color='#A23B72', label='일별 건수')
    axes[1, 1].axhline(y=avg_count, color='#E63946', linestyle='--',
                       label=f'평균: {avg_count:.1f}건', linewidth=2.5)
    axes[1, 1].set_title('일별 매출건수 추이', fontsize=14, fontweight='bold', pad=10)
    axes[1, 1].set_xlabel('날짜', fontsize=12)
    axes[1, 1].set_ylabel('매출건수 (건)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    axes[1, 1].tick_params(axis='x', rotation=45, labelsize=10)
    axes[1, 1].tick_params(axis='y', labelsize=10)
    axes[1, 1].legend(fontsize=10, loc='upper left')

    # 5. 매출금액 분포 히스토그램
    n, bins, patches = axes[2, 0].hist(daily_data[col_amount], bins=12, color='#6A994E',
                                       edgecolor='black', alpha=0.75, linewidth=1.5)
    axes[2, 0].axvline(x=avg_amount, color='#E63946', linestyle='--',
                       label=f'평균: {avg_amount:,.0f}원', linewidth=2.5)
    axes[2, 0].axvline(x=max_amount, color='#2A9D8F', linestyle=':',
                       label=f'최대: {max_amount:,.0f}원', linewidth=2)
    axes[2, 0].axvline(x=min_amount, color='#F18F01', linestyle=':',
                       label=f'최소: {min_amount:,.0f}원', linewidth=2)
    axes[2, 0].set_title('매출금액 분포', fontsize=14, fontweight='bold', pad=10)
    axes[2, 0].set_xlabel('매출금액 (원)', fontsize=12)
    axes[2, 0].set_ylabel('빈도 (일수)', fontsize=12)
    axes[2, 0].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[2, 0].tick_params(labelsize=10)
    axes[2, 0].legend(fontsize=10)

    # 6. 누적 매출 추이
    daily_data['누적매출'] = daily_data[col_amount].cumsum()
    axes[2, 1].plot(daily_data[col_date], daily_data['누적매출'],
                    linewidth=3, color='#2A9D8F', marker='o', markersize=5)
    axes[2, 1].fill_between(daily_data[col_date], daily_data['누적매출'],
                            alpha=0.3, color='#2A9D8F')
    axes[2, 1].set_title('누적 매출금액 추이', fontsize=14, fontweight='bold', pad=10)
    axes[2, 1].set_xlabel('날짜', fontsize=12)
    axes[2, 1].set_ylabel('누적 매출금액 (원)', fontsize=12)
    axes[2, 1].grid(True, alpha=0.3, linestyle='--')
    axes[2, 1].tick_params(axis='x', rotation=45, labelsize=10)
    axes[2, 1].tick_params(axis='y', labelsize=10)

    # 최종 누적값 표시
    final_value = daily_data['누적매출'].iloc[-1]
    axes[2, 1].text(0.5, 0.95, f'총 누적 매출: {final_value:,.0f} 원',
                   transform=axes[2, 1].transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFE66D',
                            edgecolor='#E63946', linewidth=2.5, alpha=0.9),
                   fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('electronics_data_visualization.png', dpi=300, bbox_inches='tight')
    print("Done: Chart saved as 'electronics_data_visualization.png'")
    plt.show()

    # Print statistics
    print("\n" + "="*80)
    print("Data Statistics Summary")
    print("="*80)
    print(f"\nTotal Period: {daily_data.shape[0]} days")
    print(f"Date Range: {daily_data[col_date].min().strftime('%Y-%m-%d')} ~ {daily_data[col_date].max().strftime('%Y-%m-%d')}")

    print(f"\n[Sales Amount]")
    print(f"  Total: {daily_data[col_amount].sum():,.0f} KRW")
    print(f"  Average: {daily_data[col_amount].mean():,.0f} KRW")
    print(f"  Median: {daily_data[col_amount].median():,.0f} KRW")
    print(f"  Std Dev: {daily_data[col_amount].std():,.0f} KRW")
    print(f"  Max: {daily_data[col_amount].max():,.0f} KRW (on {daily_data.loc[daily_data[col_amount].idxmax(), col_date].strftime('%Y-%m-%d')})")
    print(f"  Min: {daily_data[col_amount].min():,.0f} KRW (on {daily_data.loc[daily_data[col_amount].idxmin(), col_date].strftime('%Y-%m-%d')})")

    print(f"\n[Sales Count]")
    print(f"  Total: {daily_data[col_count].sum():,}")
    print(f"  Average: {daily_data[col_count].mean():.1f}")
    print(f"  Median: {daily_data[col_count].median():.1f}")
    print(f"  Max: {daily_data[col_count].max():,} (on {daily_data.loc[daily_data[col_count].idxmax(), col_date].strftime('%Y-%m-%d')})")
    print(f"  Min: {daily_data[col_count].min():,} (on {daily_data.loc[daily_data[col_count].idxmin(), col_date].strftime('%Y-%m-%d')})")

    print(f"\n[Day of Week Analysis]")
    for i, day_name in enumerate(dow_names):
        avg_sales = dow_sales.values[i]
        print(f"  {day_name}: {avg_sales:,.0f} KRW (average)")

    best_day = dow_names[dow_sales.argmax()]
    worst_day = dow_names[dow_sales.argmin()]
    print(f"\n  Best day: {best_day} ({dow_sales.max():,.0f} KRW)")
    print(f"  Worst day: {worst_day} ({dow_sales.min():,.0f} KRW)")

    print("\n" + "="*80)
    print("Visualization completed!")
    print("="*80)

    return daily_data


if __name__ == "__main__":
    # CSV file path
    csv_file = "electronics_final.csv"

    # Visualize data
    daily_data = visualize_electronics_data(csv_file)

    print("\nGenerated file:")
    print("  - electronics_data_visualization.png")
    print("\nThis chart shows:")
    print("  1. Daily sales amount trend with average line")
    print("  2. Daily sales count trend with average line")
    print("  3. Sales amount distribution histogram")
    print("  4. Sales count distribution histogram")
    print("  5. Average sales by day of week (bar chart)")
    print("  6. Cumulative sales trend")
