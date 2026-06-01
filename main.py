import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="서울 1950년대 기온 복원 프로젝트",
    page_icon="🕰️",
    layout="wide"
)

# --- 커스텀 CSS (가독성 및 디자인 대폭 개선) ---
def add_custom_css():
    st.markdown(
        """
        <style>
        /* 배경 화면: 어둡고 정적인 느낌의 이미지 + 아주 짙은 다크 오버레이 */
        .stApp {
            background: linear-gradient(rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.85)), 
                        url('https://images.unsplash.com/photo-1503694978374-8a2fa686963a?q=80&w=2069&auto=format&fit=crop');
            background-size: cover;
            background-attachment: fixed;
        }

        /* 메인 컨텐츠 영역 박스: 글씨가 배경에 묻히지 않도록 불투명도 높은 다크 패널 적용 */
        .main .block-container {
            background: rgba(30, 41, 59, 0.92); /* 짙은 슬레이트 색상 */
            border-radius: 15px;
            padding: 3rem;
            margin-top: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* 폰트 색상 명확화 (가독성 보장) */
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF !important; /* 완전한 흰색 */
            font-family: 'Pretendard', sans-serif;
        }
        
        .stMarkdown p, .stMarkdown li, div, span, label {
            color: #E2E8F0 !important; /* 밝은 회색 */
            font-size: 1.1rem;
        }

        /* 강조 색상 */
        .st-emotion-cache-1kyxreq {
            color: #38bdf8 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_custom_css()

# --- 데이터 로드 함수 ---
@st.cache_data
def get_data():
    try:
        df = pd.read_csv('ta_20260601093156.csv', encoding='utf-8')
        df['날짜'] = df['날짜'].str.strip()
    except:
        # 파일이 없을 경우를 대비한 1950년대 데모 데이터 생성 (결측치 포함)
        dates = pd.date_range(start='1950-01-01', end='1959-12-31')
        temp = [np.nan if (d.year >= 1950 and d.year <= 1953) else 12 + 15*np.sin((d.dayofyear-100)/365*2*np.pi) + np.random.normal(0, 3) for d in dates]
        df = pd.DataFrame({'날짜': dates.strftime('%Y-%m-%d'), '평균기온(℃)': temp})
    
    df['Date'] = pd.to_datetime(df['날짜'])
    df['Year'] = df['Date'].dt.year
    df['DOY'] = df['Date'].dt.dayofyear
    df = df.rename(columns={'평균기온(℃)': 'Temp'})
    return df

df_full = get_data()
df_50s = df_full[(df_full['Year'] >= 1950) & (df_full['Year'] <= 1959)].copy()

# --- 사이드바: 복원 설정 ---
with st.sidebar:
    st.header("⚙️ 데이터 복원 설정")
    st.write("끊어진 그래프의 빈칸을 어떤 방식으로 채울지 선택하세요.")
    
    method = st.radio(
        "복원 알고리즘 선택",
        ["선형 보간법 (Linear Interpolation)", 
         "역사적 평균법 (Climatology)", 
         "AI 머신러닝 (Random Forest)"]
    )
    
    view_range = st.slider(
        "조회할 연도 범위",
        min_value=1950, max_value=1959, value=(1950, 1955)
    )

# --- 메인 화면 시작 ---
st.title("🌡️ 1950년대 서울, 잃어버린 기온의 복원")
st.markdown(
    """
    한국전쟁이 발발했던 1950년대 초반, 서울의 기온 데이터는 텅 비어 있습니다. 
    아래 그래프에서 **파란색 선**은 실제 기록된 데이터가 어떻게 끊겨 있는지를 보여줍니다. 
    좌측 메뉴에서 복원 기술을 선택해, **끊어진 구간을 빨간색 점선으로 부드럽게 채워보세요.**
    """
)

# --- 데이터 복원 (Imputation) 로직 ---
@st.cache_data
def impute_data(method_name):
    df_res = df_50s.copy()
    
    if method_name == "선형 보간법 (Linear Interpolation)":
        # 끊어진 앞뒤 값을 직선으로 연결
        df_res['Temp_Full'] = df_res['Temp'].interpolate(limit_direction='both')
        
    elif method_name == "역사적 평균법 (Climatology)":
        # 전체 데이터 기준, 1년 중 해당 일자의 평균 기온을 구해 대입
        doy_mean = df_full.groupby('DOY')['Temp'].mean()
        df_res['Temp_Full'] = df_res['Temp'].fillna(df_res['DOY'].map(doy_mean))
        
    elif method_name == "AI 머신러닝 (Random Forest)":
        # 결측치가 없는 데이터로 AI 모델 학습
        train_data = df_full[df_full['Temp'].notna()]
        model = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(train_data[['DOY', 'Year']], train_data['Temp'])
        
        # 빈칸 부분만 예측하여 채우기
        df_res['Temp_Full'] = df_res['Temp'].copy()
        missing_mask = df_res['Temp'].isna()
        if missing_mask.sum() > 0:
            df_res.loc[missing_mask, 'Temp_Full'] = model.predict(df_res.loc[missing_mask, ['DOY', 'Year']])
            
    df_res['Is_Imputed'] = df_res['Temp'].isna()
    return df_res

res_df = impute_data(method)

# 사용자가 선택한 연도 범위 필터링
mask = (res_df['Year'] >= view_range[0]) & (res_df['Year'] <= view_range[1])
chart_data = res_df[mask]

st.markdown("---")
st.subheader(f"📊 {view_range[0]}년 ~ {view_range[1]}년 서울 기온 흐름")

# --- 시각화: 빈칸을 채우는 직관적인 그래프 ---
fig = go.Figure()

# 1. 뼈대: 복원된 전체 데이터 라인 (빈칸 구간에서 이 빨간 점선이 노출됨)
fig.add_trace(go.Scatter(
    x=chart_data['Date'], 
    y=chart_data['Temp_Full'],
    mode='lines',
    name='복원된 빈칸 (예측치)',
    line=dict(color='#ef4444', width=2, dash='dash'), # 빨간색 점선
    opacity=0.8
))

# 2. 덮개: 실제 관측 데이터 라인 (실제 데이터가 있는 곳은 파란색 실선이 덮음)
# connectgaps=False 로 설정하여 결측치 구간에서 선이 완전히 끊어지게 만듦
fig.add_trace(go.Scatter(
    x=chart_data['Date'], 
    y=chart_data['Temp'],
    mode='lines',
    name='실제 관측 기록',
    line=dict(color='#38bdf8', width=2.5), # 선명한 하늘색 실선
    connectgaps=False 
))

fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1
    ),
    xaxis=dict(showgrid=False, title="연도/날짜"),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="기온 (℃)"),
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig, use_container_width=True)

# --- 통계 요약 및 다운로드 ---
missing_days = chart_data['Temp'].isna().sum()
total_days = len(chart_data)

col1, col2 = st.columns(2)
with col1:
    st.write(f"**선택하신 기간({view_range[0]}~{view_range[1]}) 요약:**")
    st.write(f"총 {total_days:,}일 중 **{missing_days:,}일**의 기온이 소실되었습니다.")
    st.write(f"현재 **[{method}]**을(를) 통해 끊어진 {missing_days:,}일의 그래프를 완벽하게 이었습니다.")

with col2:
    st.write("**📥 복원된 전체 데이터 다운로드**")
    csv = chart_data[['Date', 'Temp_Full', 'Is_Imputed']].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="완성된 데이터 다운로드 (CSV)",
        data=csv,
        file_name=f"seoul_temp_imputed_{view_range[0]}_{view_range[1]}.csv",
        mime="text/csv"
    )
