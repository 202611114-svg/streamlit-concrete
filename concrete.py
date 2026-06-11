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

# 2. 인더스트리얼 대시보드 전용 고급 CSS 스타일 정의
st.markdown("""
    <style>
    /* 배경이 허전하지 않도록 은은한 엔지니어링 모눈/그리드 배경 효과 적용 */
    .stApp {
        background-color: #F4F6F7;
        background-image: linear-gradient(#E5E8E8 1px, transparent 1px), linear-gradient(90deg, #E5E8E8 1px, transparent 1px);
        background-size: 20px 20px;
    }
    
    /* 공사장 표지판 스타일 배너 */
    .construction-banner {
        background-color: #FFC107;
        border: 4px solid #212529;
        padding: 18px;
        border-radius: 6px;
        text-align: center;
        box-shadow: 0px 6px 12px rgba(0,0,0,0.15);
        margin-bottom: 25px;
        position: relative;
    }
    .construction-banner::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 6px;
        background: repeating-linear-gradient(-45deg, #212529, #212529 10px, #FFC107 10px, #FFC107 20px);
    }
    
    .main-title {
        font-size: 25px;
        font-weight: 900;
        color: #212529;
        margin: 5px 0 2px 0;
        font-family: 'Courier New', monospace;
    }
    .sub-title {
        font-size: 13px;
        font-weight: bold;
        color: #495057;
        letter-spacing: 0.5px;
    }
    
    /* 입력 컴포넌트가 배치되는 제어 패널 구역 정의 (허전함 채우기 스킨) */
    .control-panel {
        background-color: #FFFFFF;
        border: 2px solid #BDC3C7;
        border-top: 5px solid #34495E;
        padding: 25px;
        border-radius: 8px;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05), 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    .panel-header {
        font-size: 15px;
        font-weight: bold;
        color: #2C3E50;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* 판정 결과 박스 */
    .result-box {
        padding: 20px;
        border-radius: 6px;
        text-align: center;
        font-weight: bold;
        font-size: 22px;
        margin-top: 20px;
        border: 3px solid;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    }
    
    /* 계측 개시 버튼 커스텀 */
    .stButton>button {
        background-color: #2C3E50 !important; /* 강인한 다크 네이비 테마 */
        color: #FFC107 !important; 
        font-weight: bold !important;
        border: 2px solid #1A252F !important;
        height: 48px;
        font-size: 16px !important;
        border-radius: 6px !important;
        box-shadow: 0px 3px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #1A252F !important;
        color: #FFFFFF !important;
        box-shadow: 0px 5px 12px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# 상단 배너 표시
st.markdown('''
    <div class="construction-banner">
        <div class="main-title">🚧 CONCRETE SAFETY SYSTEM V2 🚧</div>
        <div class="sub-title">[ 구조물 콘크리트 배합 안전성 AI 실시간 계측 모니터 ]</div>
    </div>
''', unsafe_allow_html=True)

# 모델 로드 디렉토리 체크
model_path = 'concrete_model.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

rf_model = load_model()

if rf_model is None:
    st.error(f"❌ 인프라 연동 실패: 폴더 내에 '{model_path}' 파일이 누락되었습니다. 경로를 재정비하세요.")
else:
    # 3. 데이터 입력 패널 내부 구성을 전용 컨테이너와 슬라이더(막대)로 배치
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-header">⚙️ CONTROL CONSOLE (막대를 조절하여 배합비를 맞추세요)</div>', unsafe_allow_html=True)
    
    # 편의성을 위해 슬라이더(st.slider) 장착 및 현실적인 범위 지정
    cement = st.slider("🏗️ 시멘트 배합량 (Cement, kg/m³)", min_value=100.0, max_value=600.0, value=250.0, step=5.0)
    water = st.slider("💧 배합수량 (Water, kg/m³)", min_value=120.0, max_value=250.0, value=180.0, step=2.0)
    age = st.slider("📅 구조물 양생기간 (Age, day)", min_value=1, max_value=100, value=28, step=1)
    
    st.markdown('</div>', unsafe_allow_html=True) # 패널 닫기
    
    st.write("")
    
    # 4. 분석 연산 가동 버튼
    if st.button("⚙️ 시스템 가동: AI 정밀 안전 계측", use_container_width=True):
        
        # 모델의 피처 컬럼 규격과 정확히 일치시킴
        input_data = pd.DataFrame(
            [[cement, water, age]],
            columns=['시멘트', '물', '양생기간']
        )
        
        try:
            # 안전/불안전 확률 집계
            probabilities = rf_model.predict_proba(input_data)[0]
            
            prob_unsafe = probabilities[0] * 100
            prob_safe = probabilities[1] * 100
            
            # 위험도 분기처리 스킨 컬러 맵
            if prob_safe >= prob_unsafe:
                status = "안전 보증 (SAFE)"
                confidence = prob_safe
                box_color = "#EAFAF1"  
                text_color = "#145A32"
                border_color = "#27AE60"
            else:
                status = "구조 위험 (DANGER)"
                confidence = prob_unsafe
                box_color = "#FDEDEC"  
                text_color = "#78281F"
                border_color = "#E74C3C"
                
            # 5. 시각 결과 패널 출력
            st.markdown(f'''
                <div class="result-box" style="background-color: {box_color}; color: {text_color}; border-color: {border_color};">
                    👷 AI 구조 판정: {status}<br>
                    <span style="font-size: 15px; font-weight: normal; color: #7F8C8D;">(종합 판정 신뢰 확률: {confidence:.2f}%)</span>
                </div>
            ''', unsafe_allow_html=True)
            
            # 6. 하단 게이지 차트 분석 영역
            st.write("")
            st.markdown('<strong>📊 REAL-TIME CORE LOG (실시간 계측 로그 수치)</strong>', unsafe_allow_html=True)
            
            st.text(f"🟢 안전성 확보 비율 (Safe Index): {prob_safe:.1f}%")
            st.progress(int(prob_safe))
            
            st.text(f"🔴 균열 및 위험 우려도 (Unsafe Index): {prob_unsafe:.1f}%")
            st.progress(int(prob_unsafe))
            
        except Exception as e:
            st.error(f"⚠️ 시스템 경보 - 계측 연산 에러: {e}")
