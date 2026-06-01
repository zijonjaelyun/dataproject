# dataproject
```python?code_reference&code_event_index=5
readme_content = """# 🌡️ 서울 1950년대 기온 복원 프로젝트 (Seoul Temperature Imputation)

이 프로젝트는 1907년부터 최근까지의 서울 기상 관측 데이터(`ta_20260601093156.csv`)를 바탕으로, 한국전쟁 전후(1950년대)에 발생한 **기온 결측치를 분석하고 다양한 데이터 분석/머신러닝 기법으로 복원해보는 Streamlit 웹 애플리케이션**입니다.

## ✨ 주요 기능

스트림릿(Streamlit)의 최신 네비게이션 기능을 활용한 멀티 페이지(Multi-page) 앱으로 구성되어 있습니다.

1. **📊 데이터 현황 탐색 (Page 1)**
   - 서울 기온 데이터의 전체 개요 및 통계 확인
   - 1950년대 집중 결측치 현황 시각화 및 원본 데이터 탐색
   - 파일 업로드 기능 지원 (기본 파일이 없을 경우)

2. **🔮 기온 복원 시뮬레이터 (Page 2)**
   - 다양한 알고리즘을 통한 결측치 복원 시뮬레이션
     - `선형 보간법 (Linear Interpolation)`
     - `역대 일별 평균값 대입법 (Historical Day-of-Year Mean)`
     - `머신러닝 예측 (Random Forest Regressor)`
   - 연도별 복원 결과 인터랙티브 시각화 (Plotly)
   - 복원된 데이터 CSV 다운로드 지원

## 📂 프로젝트 구조
