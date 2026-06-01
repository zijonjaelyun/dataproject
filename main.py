import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# 페이지 기본 설정
st.set_page_config(
    page_title="서울 1950년대 기온 복원 프로젝트",
    page_icon="🌡️",
    layout="wide"
)

# ------------------------------------------------------------------
# [공통 데이터 로드 및 전처리 함수]
# ------------------------------------------------------------------
@st.cache_data
def load_and_preprocess_data(file_path_or_buffer):
    # CSV 로드 (UTF-8 인코딩)
    df = pd.read_csv(file_path_or_buffer, encoding='utf-8')
    
    # 칼럼명 및 날짜 정제
    df['날짜'] = df['날짜'].str.strip()
    df['Date'] = pd.to_datetime(df['날짜'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    # 분석 편의를 위한 영문 칼럼 매핑
    df = df.rename(columns={
        '평균기온(℃)': 'Temp_Avg',
        '최저기온(℃)': 'Temp_Min',
        '최고기온(℃)': 'Temp_Max'
    })
    return df

# 데이터 소스 확보 (로컬 파일 우선, 없으면 업로더)
df_raw = None
try:
    df_raw = load_and_preprocess_data('ta_20260601093156.csv')
except Exception:
    pass

# ------------------------------------------------------------------
# [Page 1: 데이터 개요 및 누락 현황 탐색]
# ------------------------------------------------------------------
def show_page_overview():
    st.title("📊 서울 기온 데이터 개요 및 누락 현황")
    st.markdown("---")
    
    if df_raw is None:
        st.warning("⚠️ 디렉토리에 `ta_20260601093156.csv` 파일이 없습니다. 아래에 파일을 업로드해주세요.")
        uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
        if uploaded_file is not None:
            df = load_and_preprocess_data(uploaded_file)
        else:
            st.stop()
    else:
        df = df_raw
        st.success("✅ `ta_20260601093156.csv` 데이터를 성공적으로 불러왔습니다!")

    # 기본 통계 요약
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("전체 데이터 기간", f"{df['Year'].min()}년 ~ {df['Year'].max()}년")
    with col2:
        st.metric("총 관측 일수", f"{len(df):,} 일")
    with col3:
        st.metric("평균기온 누락 건수", f"{df['Temp_Avg'].isna().sum()} 건")

    st.subheader("📌 1950년대(1950~1959) 데이터 집중 분석")
    df_50s = df[(df['Year'] >= 1950) & (df['Year'] <= 1959)].copy()
    
    total_50s = len(df_50s)
    missing_50s = df_50s['Temp_Avg'].isna().sum()
    missing_ratio = (missing_50s / total_50s) * 100
    
    st.write(f"1950년대 총 {total_50s:,}일 중 **{missing_50s:,}일({missing_ratio:.2f}%)**의 평균기온 데이터가 결측 상태입니다.")
    
    # 연도별 결측치 시각화
    missing_by_year = df_50s.groupby('Year')['Temp_Avg'].apply(lambda x: x.isna().sum()).reset_index()
    missing_by_year.columns = ['Year', 'Missing_Days']
    
    fig = px.bar(missing_by_year, x='Year', y='Missing_Days', 
                 title="1950년대 연도별 평균기온 결측 일수 (한국전쟁 시기 집중)",
                 labels={'Missing_Days': '결측 일수', 'Year': '연도'},
                 text_auto=True, color='Missing_Days', color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)

    # 데이터 샘플 확인
    st.subheader("👀 1950년대 원본 데이터 샘플 (결측치 포함)")
    st.dataframe(df_50s[['Date', 'Temp_Avg', 'Temp_Min', 'Temp_Max']].reset_index(drop=True), use_container_width=True)


# ------------------------------------------------------------------
# [Page 2: 1950년대 기온 다양한 알고리즘으로 복원하기]
# ------------------------------------------------------------------
def show_page_imputation():
    st.title("🔮 다양한 예측 기법을 통한 기온 데이터 복원")
    st.markdown("---")
    
    if df_raw is None:
        st.info("첫 번째 페이지에서 데이터를 먼저 업로드하거나 프로젝트 폴더에 CSV를 배치해주세요.")
        st.stop()
    else:
        df = df_raw.copy()

    st.sidebar.header("🛠️ 복원 방법 설정")
    method = st.sidebar.selectbox(
        "복원 알고리즘 선택",
        ["선형 보간법 (Linear Interpolation)", 
         "역대 일별 평균값 대입법 (Historical Day-of-Year Mean)", 
         "머신러닝 예측 (Random Forest Regressor)"]
    )
    
    target_year = st.sidebar.slider("분석 및 시각화 타겟 연도", 1950, 1959, 1953)

    # 복원 수행
    if method == "선형 보간법 (Linear Interpolation)":
        st.markdown("> **선형 보간법:** 결측치의 앞값과 뒷값을 직선으로 연결하여 중간값을 채우는 방식입니다. 연속적으로 짧게 빠진 구간에 유리합니다.")
        df['Imputed_Temp'] = df['Temp_Avg'].interpolate(method='linear')
        
    elif method == "역대 일별 평균값 대입법 (Historical Day-of-Year Mean)":
        st.markdown("> **역대 일별 평균값 대입법:** 110여 년의 역사 동안 해당 일자(예: 모든 해의 6월 25일)에 관측된 평균 기온을 계산하여 대입하는 기후학적 방식입니다.")
        # 일차(Day of Year)별 평균 계산
        doy_mean = df.groupby('DayOfYear')['Temp_Avg'].mean()
        df['Imputed_Temp'] = df['Temp_Avg'].fillna(df['DayOfYear'].map(doy_mean))
        
    elif method == "머신러닝 예측 (Random Forest Regressor)":
        st.markdown("> **머신러닝 예측 (Random Forest):** 결측치가 없는 정상 기간의 데이터(연, 월, 일, 일차 정보)를 학습하여 결측 기간의 기온을 예측·복원합니다.")
        
        with st.spinner("머신러닝 모델 학습 및 예측 중... (약 3~5초 소요)"):
            # 결측치 유무로 트레인/테스트 분리
            train_mask = df['Temp_Avg'].notna()
            train_df = df[train_mask]
            
            features = ['Year', 'Month', 'Day', 'DayOfYear']
            
            X_train = train_df[features]
            y_train = train_df['Temp_Avg']
            
            # 모델 정의 (속도를 위해 나무 개수 조절)
            rf = RandomForestRegressor(n_estimators=30, max_depth=12, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            
            # 전체 데이터 예측 후 결측치만 치환
            df['Pred_Temp'] = rf.predict(df[features])
            df['Imputed_Temp'] = df['Temp_Avg'].fillna(df['Pred_Temp'])

    # 복원 여부 구분 플래그 추가
    df['Is_Imputed'] = df['Temp_Avg'].isna()

    # 타겟 연도 필터링
    df_target = df[df['Year'] == target_year].copy()
    
    # 시각화 데이터 구성
    st.subheader(f"📈 {target_year}년 기온 복원 결과 시각화")
    
    if df_target['Is_Imputed'].sum() == 0:
        st.info(f"{target_year}년은 이미 원본 데이터가 온전하게 존재합니다! 복원된 가상의 값이 아닌 실제 관측치 그래프를 보여줍니다.")
    
    # 라인 차트 생성 (실제 데이터와 복원 데이터를 다른 색상이나 힌트로 파악 가능하게 구성)
    fig_line = px.line(df_target, x='Date', y='Imputed_Temp', 
                       title=f"{target_year}년 일별 평균 기온 추이 ({method})",
                       labels={'Imputed_Temp': '기온 (℃)', 'Date': '날짜'})
    
    # 복원된 구역을 음영이나 산점도로 표시
    df_imputed_points = df_target[df_target['Is_Imputed'] == True]
    if not df_imputed_points.empty:
        fig_line.add_scatter(x=df_imputed_points['Date'], y=df_imputed_points['Imputed_Temp'], 
                             mode='markers', name='복원된 데이터 포인트', 
                             marker=dict(color='red', size=6, symbol='circle'))
        
    st.plotly_chart(fig_line, use_container_width=True)

    # 데이터 요약 비교 및 다운로드
    st.subheader("📥 복원 완료된 데이터 확인 및 다운로드")
    show_cols = ['Date', 'Temp_Avg', 'Imputed_Temp', 'Is_Imputed']
    st.dataframe(df_target[show_cols].reset_index(drop=True), use_container_width=True)
    
    # CSV 다운로드 버튼
    csv_data = df_target[show_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"💾 {target_year}년 복원 데이터 CSV 다운로드",
        data=csv_data,
        file_name=f"seoul_temperature_imputed_{target_year}.csv",
        mime="text/csv"
    )

# ------------------------------------------------------------------
# [멀티 페이지 라우팅 시스템 정의 - st.navigation]
# ------------------------------------------------------------------
page_overview = st.Page(show_page_overview, title="데이터 현황 탐색", icon="📊")
page_imputation = st.Page(show_page_imputation, title="기온 복원 시뮬레이터", icon="🔮")

# 네비게이션 구동
pg = st.navigation([page_overview, page_imputation])
pg.run()
