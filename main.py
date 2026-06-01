import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="서울 기온 복원 및 디지털 게임",
    page_icon="🌡️",
    layout="wide"
)

# --- 100% 가독성 보장 고대비 CSS 스타일링 ---
def apply_high_contrast_theme():
    st.markdown(
        """
        <style>
        /* 기본 배경: 깔끔하고 밝은 라이트 그레이 */
        .stApp {
            background-color: #F8FAFC;
        }
        /* 메인 컨텐츠 카드: 완전 흰색 패널로 글씨 시인성 100% 보장 */
        .main .block-container {
            background-color: #FFFFFF;
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(15, 23, 42, 0.08);
            border: 1px solid #E2E8F0;
            margin-top: 1.5rem;
        }
        /* 제목 및 본문 글씨 색상 강제 지정 (절대 묻히지 않음) */
        h1, h2, h3, h4, .stSubheader {
            color: #0F172A !important; /* 깊은 흑네이비 색상 */
            font-weight: 800 !important;
        }
        p, li, span, label, div {
            color: #334155 !important; /* 선명한 다크 그레이 */
            font-weight: 500;
        }
        /* 라디오 버튼 및 슬라이더 라벨 글씨 강화 */
        .stRadio label, .stSlider label {
            color: #0F172A !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
        }
        /* 사이드바 영역 스타일 */
        [data-testid="stSidebar"] {
            background-color: #0F172A !important;
        }
        [data-testid="stSidebar"] * {
            color: #F1F5F9 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

apply_high_contrast_theme()

# --- 공통 데이터 처리 기술 엔진 ---
@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv('ta_20260601093156.csv', encoding='utf-8')
        df['날짜'] = df['날짜'].str.strip()
    except:
        # 파일 경로 예외 발생 시 자동 시뮬레이션 데이터 생성
        dates = pd.date_range(start='1950-01-01', end='1959-12-31')
        temp = [np.nan if (d.year >= 1950 and d.year <= 1952) else 12 + 14*np.sin((d.dayofyear-100)/365*2*np.pi) + np.random.normal(0, 2) for d in dates]
        df = pd.DataFrame({'날짜': dates.strftime('%Y-%m-%d'), '평균기온(℃)': temp})
    
    df['Date'] = pd.to_datetime(df['날짜'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DOY'] = df['Date'].dt.dayofyear
    df = df.rename(columns={'평균기온(℃)': 'Temp'})
    return df

df_master = load_historical_data()

# --- PAGE 1: 인터랙티브 복원 시뮬레이터 ---
def show_page_simulator():
    st.title("📊 1950년대 서울 기온 복원 실험실")
    st.write("한국전쟁으로 끊겨버린 서울의 기온 그래프를 유저님이 직접 상호작용하여 채워보는 공간입니다.")
    
    st.markdown("---")
    
    # 사이드바 설정 연동 대신 화면 상단 배치로 직관성 향상
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("🛠️ 복원 옵션")
        method = st.selectbox(
            "사용할 수학/AI 알고리즘",
            ["선형 보간법 (Linear Interpolation)", "기후학적 역사 평균법 (Climatology)", "AI 예측 모델 (Random Forest)"]
        )
        
        target_years = st.slider("조회 연도 범위", 1950, 1959, (1950, 1954))
        
        st.write("")
        # ✨ 핵심 상호작용 버튼
        show_imputed = st.checkbox("🔍 잃어버린 빈칸에 복원 데이터 채워넣기", value=False)
        if show_imputed:
            st.success("💡 복원 데이터가 그래프에 활성화되었습니다! 빨간 점선을 확인하세요.")
        else:
            st.info("👆 위 체크박스를 선택하면 끊어진 빈 우주 같은 그래프가 채워집니다.")

    # 데이터 연산
    df_50s = df_master[(df_master['Year'] >= target_years[0]) & (df_master['Year'] <= target_years[1])].copy()
    
    if "선형" in method:
        df_50s['Temp_Imputed'] = df_50s['Temp'].interpolate(limit_direction='both')
    elif "역사" in method:
        doy_mean = df_master.groupby('DOY')['Temp'].mean()
        df_50s['Temp_Imputed'] = df_50s['Temp'].fillna(df_50s['DOY'].map(doy_mean))
    else:
        train = df_master[df_master['Temp'].notna()]
        rf = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(train[['DOY', 'Year']], train['Temp'])
        df_50s['Temp_Imputed'] = df_50s['Temp'].copy()
        m_mask = df_50s['Temp'].isna()
        if m_mask.sum() > 0:
            df_50s.loc[m_mask, 'Temp_Imputed'] = rf.predict(df_50s.loc[m_mask, ['DOY', 'Year']])

    with c2:
        st.subheader("📈 서울 기온 데이터 시각화")
        fig = go.Figure()

        # 1. 상호작용 상태에 따라 복원 점선 그래프 노출 여부 결정
        if show_imputed:
            fig.add_trace(go.Scatter(
                x=df_50s['Date'], y=df_50s['Temp_Imputed'],
                mode='lines', name='복원된 기온 (예측치)',
                line=dict(color='#EF4444', width=2, dash='dash')
            ))

        # 2. 실제 기록된 데이터 선 (언제나 노출, 결측 구간은 단절됨)
        fig.add_trace(go.Scatter(
            x=df_50s['Date'], y=df_50s['Temp'],
            mode='lines+markers', name='실제 관측 기록',
            line=dict(color='#0284C7', width=2.5),
            connectgaps=False
        ))

        fig.update_layout(
            paper_bgcolor='white', plot_bgcolor='#F8FAFC',
            hovermode="x unified",
            xaxis=dict(gridcolor='#E2E8F0', title="날짜"),
            yaxis=dict(gridcolor='#E2E8F0', title="기온 (℃)"),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)


# --- PAGE 2: 내가 그리는 기온 그래프 (미니 게임) ---
def show_page_game():
    st.title("🎮 [미니 게임] 사라진 1952년 여름을 그려라!")
    st.write("1952년 5월부터 9월까지 서울의 실제 기온 기록은 완전히 소실되었습니다. 데이터 분석가가 되어 당시의 월별 평균 기온을 예측해보세요!")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("✏️ 유저 기온 입력 패널")
        st.write("각 월의 예상 평균 기온을 슬라이더로 조절하여 나만의 그래프를 그려보세요.")
        
        u_may = st.slider("5월 평균 기온 예측 (℃)", 10.0, 30.0, 14.0, step=0.5)
        u_jun = st.slider("6월 평균 기온 예측 (℃)", 15.0, 35.0, 18.0, step=0.5)
        u_jul = st.slider("7월 평균 기온 예측 (℃)", 15.0, 40.0, 20.0, step=0.5)
        u_aug = st.slider("8월 평균 기온 예측 (℃)", 15.0, 40.0, 21.0, step=0.5)
        u_sep = st.slider("9월 평균 기온 예측 (℃)", 10.0, 30.0, 16.0, step=0.5)
        
        check_btn = st.button("🎯 정답 확인 및 스코어 계산하기", use_container_width=True)

    # 수학적 정답 세팅 (역사적 기후 평균값 추출)
    ans_months = [5, 6, 7, 8, 9]
    real_ans = []
    for m in ans_months:
        val = df_master[(df_master['Month'] == m) & (df_master['Temp'].notna())]['Temp'].mean()
        real_ans.append(round(val, 1))

    user_guesses = [u_may, u_jun, u_jul, u_aug, u_sep]

    with col2:
        st.subheader("📉 내가 그린 그래프 vs 실제 예측 모델")
        
        fig_game = go.Figure()
        months_label = ['5월', '6월', '7월', '8월', '9월']
        
        # 유저 예측선
        fig_game.add_trace(go.Scatter(
            x=months_label, y=user_guesses,
            mode='lines+markers', name='내가 그린 기온선',
            line=dict(color='#F59E0B', width=4),
            marker=dict(size=10)
        ))
        
        if check_btn:
            # 정답 확인을 누르면 실제 수학적 복원선 등장
            fig_game.add_trace(go.Scatter(
                x=months_label, y=real_ans,
                mode='lines+markers', name='실제 기후 모델 정답선',
                line=dict(color='#10B981', width=4, dash='dash'),
                marker=dict(size=10)
            ))
            
            # 점수 계산 (Mean Absolute Error 기반 점수화)
            mae = np.mean(np.abs(np.array(user_guesses) - np.array(real_ans)))
            score = max(0, int(100 - (mae * 7)))
            
            st.metric(label="Restoration Accuracy Score", value=f"{score}점 / 100점")
            
            if score >= 90:
                st.balloons()
                st.success("🥇 완벽합니다! 당신은 전설적인 디지털 기후 역사 복원가입니다!")
            elif score >= 75:
                st.info("🥈 훌륭합니다! 과거 서울의 계절적 특징을 완벽하게 파악하고 계시네요.")
            else:
                st.warning("🥉 아쉽습니다! 정답선(녹색 점선)을 참고하여 기온을 다시 조정해 보세요.")

        fig_game.update_layout(
            paper_bgcolor='white', plot_bgcolor='#F8FAFC',
            xaxis=dict(title="월"), yaxis=dict(title="평균 기온 (℃)", range=[5, 35]),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_game, use_container_width=True)


# --- 멀티 페이지 내비게이션 라우팅 ---
page_1 = st.Page(show_page_simulator, title="데이터 복원 실험실", icon="📊")
page_2 = st.Page(show_page_game, title="기온 맞추기 미니게임", icon="🎮")

pg = st.navigation([page_1, page_2])
pg.run()
