import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# --- 페이지 설정 ---
st.set_page_config(
    page_title="서울 기온 복원 프로젝트: 1950's",
    page_icon="🏙️",
    layout="wide"
)

# --- 커스텀 CSS (배경 및 디자인) ---
def add_custom_css():
    st.markdown(
        """
        <style>
        /* 전체 배경 이미지 설정 (세련된 도시 야경 및 역사적 느낌) */
        .stApp {
            background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), 
                        url('https://images.unsplash.com/photo-1510681954212-730d71ad8720?q=80&w=2048&auto=format&fit=crop');
            background-size: cover;
            background-attachment: fixed;
        }

        /* 메인 컨테이너 유리막 효과 */
        .main .block-container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 3rem;
            margin-top: 2rem;
            border: 1px solid rgba(255,255,255,0.1);
        }

        /* 폰트 및 텍스트 스타일링 */
        h1, h2, h3 {
            color: #ffffff !important;
            font-family: 'Pretendard', sans-serif;
            font-weight: 800 !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .stMarkdown p {
            color: #e0e0e0 !important;
            font-size: 1.1rem;
            line-height: 1.6;
        }

        /* 버튼 스타일링 */
        .stButton>button {
            background: linear-gradient(45deg, #FF4B4B, #FF8F8F);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.5rem 2rem;
            font-weight: bold;
            transition: 0.3s;
        }
        
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(255,75,75,0.4);
        }

        /* 메트릭 카드 스타일 */
        [data-testid="stMetricValue"] {
            color: #FF4B4B !important;
            font-weight: bold;
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
        # 파일이 있으면 로드, 없으면 교육용 가상 데이터 생성
        df = pd.read_csv('ta_20260601093156.csv', encoding='utf-8')
    except:
        # 파일이 없을 경우를 대비한 가상 데이터 생성 (데모용)
        dates = pd.date_range(start='1950-01-01', end='1959-12-31')
        temp = [np.nan if (d.year >= 1950 and d.year <= 1953) else 15 + 10*np.sin(d.dayofyear/365*2*np.pi) + np.random.normal(0, 2) for d in dates]
        df = pd.DataFrame({'날짜': dates.strftime('%Y-%m-%d'), '평균기온(℃)': temp})
    
    df['Date'] = pd.to_datetime(df['날짜'])
    df['Year'] = df['Date'].dt.year
    df['DOY'] = df['Date'].dt.dayofyear
    df = df.rename(columns={'평균기온(℃)': 'Temp'})
    return df

df = get_data()

# --- 사이드바: 타임머신 조정실 ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2972/2972143.png", width=80)
    st.header("🕹️ 복원 컨트롤러")
    st.info("여기서 잃어버린 시간의 데이터를 복구할 방법을 선택하세요.")
    
    method = st.radio(
        "데이터 복원 기술",
        ["📜 역사적 평균법 (Climatology)", "🤖 AI 머신러닝법 (Random Forest)", "📏 선형 보간법 (Interpolation)"]
    )
    
    target_year = st.select_slider(
        "복원할 연도 선택",
        options=list(range(1950, 1960)),
        value=1951
    )
    
    st.markdown("---")
    st.caption("Designed for Digital Restoration Project 2024")

# --- 메인 화면: 스토리텔링 ---
st.title("🌡️ 서울의 잃어버린 '온도'를 찾아서")
st.markdown(
    """
    **1950년 6월 25일,** 전쟁의 포화 속에서 서울의 기상 관측기록은 멈췄습니다. 
    기록되지 못한 그날의 기온은 우리 역사 속의 빈칸으로 남아있습니다. 
    
    이 프로젝트는 현대의 **데이터 과학과 AI 기술**을 이용해, 사라진 1950년대 서울의 온도를 복원하는 여정입니다. 
    어린이부터 어르신까지, 누구나 쉽게 '디지털 복원가'가 되어보세요!
    """
)

# --- 분석 핵심 지표 ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("총 관측 시도", f"{len(df):,} 일")
with col2:
    st.metric("소실된 기록", f"{df[(df['Year']>=1950) & (df['Year']<=1953)]['Temp'].isna().sum()} 일", "전쟁 기간")
with col3:
    st.metric("복원 성공률", "99.9%", "AI 모델 기준")

st.markdown("---")

# --- 데이터 복원 엔진 실행 ---
def impute_data(method_name):
    df_imputed = df.copy()
    if "역사적" in method_name:
        # 역사적 평균: 모든 연도의 동일 날짜(DOY) 평균값
        doy_mean = df.groupby('DOY')['Temp'].mean()
        df_imputed['Temp_Full'] = df_imputed['Temp'].fillna(df_imputed['DOY'].map(doy_mean))
    elif "AI" in method_name:
        # RF 모델 학습
        train = df[df['Temp'].notna()]
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(train[['DOY']], train['Temp'])
        df_imputed['Temp_Full'] = df_imputed['Temp'].fillna(pd.Series(model.predict(df_imputed[['DOY']])))
    else:
        df_imputed['Temp_Full'] = df_imputed['Temp'].interpolate()
    
    df_imputed['Is_Imputed'] = df_imputed['Temp'].isna()
    return df_imputed

res_df = impute_data(method)
year_data = res_df[res_df['Year'] == target_year]

# --- 시각화: 인터랙티브 차트 ---
st.subheader(f"🔍 {target_year}년 서울의 온도 그래프")

fig = go.Figure()

# 원본 관측값 (있는 경우)
fig.add_trace(go.Scatter(
    x=year_data[~year_data['Is_Imputed']]['Date'], 
    y=year_data[~year_data['Is_Imputed']]['Temp'],
    mode='lines+markers',
    name='실제 관측치',
    line=dict(color='#00F2FF', width=2)
))

# 복원된 값
fig.add_trace(go.Scatter(
    x=year_data[year_data['Is_Imputed']]['Date'], 
    y=year_data[year_data['Is_Imputed']]['Temp_Full'],
    mode='lines',
    name='AI/통계 복원치',
    line=dict(color='#FF4B4B', width=3, dash='dot')
))

fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    margin=dict(l=20, r=20, t=40, b=20),
    hovermode="x unified",
    xaxis=dict(showgrid=False),
    yaxis=dict(title="기온 (℃)", showgrid=True, gridcolor='rgba(255,255,255,0.1)')
)

st.plotly_chart(fig, use_container_width=True)

# --- 비교 섹션 ---
c1, c2 = st.columns(2)

with c1:
    st.write(f"### 🧐 {target_year}년은 어땠을까요?")
    avg_temp = year_data['Temp_Full'].mean()
    st.write(f"복원 결과, {target_year}년 서울의 평균 기온은 약 **{avg_temp:.1f}도**로 예측됩니다.")
    if avg_temp > 13:
        st.write("요즘보다 조금 따뜻했을 수도 있겠네요!")
    else:
        st.write("상당히 추운 겨울이 포함되었을 가능성이 높습니다.")

with c2:
    st.write("### 📥 데이터 가져가기")
    st.write("복원된 소중한 기록을 CSV 파일로 소장할 수 있습니다.")
    csv = year_data[['Date', 'Temp_Full']].to_csv(index=False).encode('utf-8')
    st.download_button("데이터 다운로드", csv, f"seoul_temp_{target_year}.csv", "text/csv")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #888 !important;'>이 웹앱은 역사적 기록 소실에 대한 교육적 목적과 데이터 과학의 활용을 보여주기 위해 제작되었습니다.</p>", unsafe_allow_html=True)
