import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. 페이지 제목 및 레이아웃 설정
st.set_page_config(
    page_title="콘크리트 안전성 AI 계측 시스템",
    page_icon="🚧",
    layout="centered"
)

# 2. 공사장 + 공학 느낌의 커스텀 CSS 스타일 정의
st.markdown("""
    <style>
    /* 전체 배경 및 폰트 스타일 조정 */
    .stApp {
        background-color: #F8F9FA;
    }
    
    /* 공사장 표지판 스타일의 타이틀 대형 배너 */
    .construction-banner {
        background-color: #FFC107; /* 안전 황색 */
        border: 4px solid #212529;
        padding: 20px;
        border-radius: 5px;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
        margin-bottom: 25px;
        position: relative;
        overflow: hidden;
    }
    /* 공사장 스트라이프 느낌의 데코 레이어 (상하단 테두리 효과 대체) */
    .construction-banner::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 8px;
        background: repeating-linear-gradient(-45deg, #212529, #212529 10px, #FFC107 10px, #FFC107 20px);
    }
    
    .main-title {
        font-size: 26px;
        font-weight: 900;
        color: #212529;
        margin: 5px 0 2px 0;
        font-family: 'Courier New', monospace;
    }
    .sub-title {
        font-size: 13px;
        font-weight: bold;
        color: #495057;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* 섹션 헤더 스타일 (공학 계측 패널 느낌) */
    .panel-header {
        font-size: 16px;
        font-weight: bold;
        color: #343A40;
        border-left: 5px solid #212529;
        padding-left: 10px;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    
    /* 결과 출력 박스 (공사장 안내판 커스텀) */
    .result-box {
        padding: 22px;
        border-radius: 6px;
        text-align: center;
        font-weight: bold;
        font-size: 22px;
        margin-top: 25px;
        border: 3px solid;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
    }
    
    /* 에러/알림 스타일 커스텀 */
    .stButton>button {
        background-color: #343A40 !important; /* 인더스트리얼 다크 그레이 */
        color: #FFC107 !important; /* 황색 글씨 */
        font-weight: bold !important;
        border: 2px solid #212529 !important;
        height: 45px;
        font-size: 16px !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #212529 !important;
        color: #FFFFFF !important;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# 상단 공사장 표지판 배너 렌더링
st.markdown('''
    <div class="construction-banner">
        <div class="main-title">🚧 CONCRETE SAFETY SYSTEM 🚧</div>
        <div class="sub-title">[ 구조물 콘크리트 배합 안전성 AI 실시간 계측기 ]</div>
    </div>
''', unsafe_allow_html=True)

# 3. 모델 파일 로드
model_path = 'concrete_model.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

rf_model = load_model()

if rf_model is None:
    st.error(f"❌ 시스템 오류: 폴더 내에 '{model_path}' 파일이 존재하지 않습니다. 인프라 경로를 확인하세요.")
else:
    # 4. 입력 패널 구성
    st.markdown('<div class="panel-header">⚙️ DATA INPUT PANEL (배합 성분 입력)</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cement = st.number_input("🏗️ 시멘트량 (Cement)", min_value=0.0, max_value=1000.0, value=250.0, step=10.0, help="단위: kg/m³")
    with col2:
        water = st.number_input("💧 배합수량 (Water)", min_value=0.0, max_value=500.0, value=180.0, step=5.0, help="단위: kg/m³")
    with col3:
        age = st.number_input("📅 양생기간 (Age)", min_value=1, max_value=365, value=28, step=1, help="단위: 일(day)")

    st.write("")
    
    # 5. 계측 프로세스 작동 버튼
    if st.button("⚙️ AI 정밀 안전성 계측 개시", use_container_width=True):
        
        # 모델의 피처 형태와 순서(3개 변수)에 맞춘 데이터프레임 구조 생성
        input_data = pd.DataFrame(
            [[cement, water, age]],
            columns=['시멘트', '물', '양생기간']
        )
        
        try:
            # 모델 예측 및 확률 추출
            probabilities = rf_model.predict_proba(input_data)[0]
            
            prob_unsafe = probabilities[0] * 100  # 0 클래스(불안전) 확률
            prob_safe = probabilities[1] * 100    # 1 클래스(안전) 확률
            
            # 확률 분포에 따라 공사장 위험/안전 색상 매칭
            if prob_safe >= prob_unsafe:
                status = "안전 (SAFE)"
                confidence = prob_safe
                box_color = "#E8F8F5"  # 은은한 산업용 그린
                text_color = "#0E6251"
                border_color = "#117A65"
            else:
                status = "위험/불안전 (DANGER)"
                confidence = prob_unsafe
                box_color = "#FADBD8"  # 신호등 정지/위험 적색
                text_color = "#78281F"
                border_color = "#943126"
                
            # 6. 결과 출력 인터페이스
            st.markdown(f'''
                <div class="result-box" style="background-color: {box_color}; color: {text_color}; border-color: {border_color};">
                    👷 AI 분석 판정: {status}<br>
                    <span style="font-size: 16px; font-weight: normal; color: #5D6D7E;">(해당 클래스 판정 신뢰도: {confidence:.2f}%)</span>
                </div>
            ''', unsafe_allow_html=True)
            
            # 하단에 시스템 게이지 스타일로 프로그레스 바 배치
            st.write("")
            st.markdown('<div class="panel-header">📊 ANALYSIS REPORT (계측 분석 로그)</div>', unsafe_allow_html=True)
            
            st.text(f"🟢 안전성 확보 지수 (Safe Probability): {prob_safe:.1f}%")
            st.progress(int(prob_safe))
            
            st.text(f"🔴 균열 및 위험 지수 (Unsafe Probability): {prob_unsafe:.1f}%")
            st.progress(int(prob_unsafe))
            
        except Exception as e:
            st.error(f"⚠️ 계측 장치 모듈 작동 오류: {e}")
