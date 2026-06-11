import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. 페이지 제목 및 디자인 레이아웃 설정
st.set_page_config(
    page_title="콘크리트 안전성 예측기",
    page_icon="👷",
    layout="centered"
)

# UI/UX 가독성을 위한 커스텀 CSS 스타일 정의
st.markdown("""
    <style>
    .main-title {
        font-size: 28px;
        font-weight: bold;
        color: #2E4053;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 14px;
        color: #7FB3D5;
        text-align: center;
        margin-bottom: 25px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">👷 콘크리트 배합 안전성 예측 AI 시스템</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">시멘트, 물, 양생기간 데이터를 입력하여 실시간으로 콘크리트 안전 여부를 판별합니다.</div>', unsafe_allow_html=True)

# 2. 업로드했던 pkl 파일 자동 불러오기
model_path = 'concrete_model.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

rf_model = load_model()

if rf_model is None:
    st.error(f"❌ 폴더 내에 '{model_path}' 파일이 발견되지 않았습니다. 파일명이 정확한지 확인해 주세요.")
else:
    # 3. 사용자 입력란 레이아웃 (3개 변수 입력)
    st.subheader("📋 배합 정보 입력")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cement = st.number_input("시멘트량 (Cement, kg/m³)", min_value=0.0, max_value=1000.0, value=250.0, step=10.0)
    with col2:
        water = st.number_input("배합수량 (Water, kg/m³)", min_value=0.0, max_value=500.0, value=180.0, step=5.0)
    with col3:
        age = st.number_input("양생기간 (Age, day)", min_value=1, max_value=365, value=28, step=1)

    # 4. 정밀 검사 버튼 클릭 시 동작
    if st.button("🚀 안전성 정밀 검사 시작", use_container_width=True):
        
        # [핵심] 내부 전처리 로직: 변수는 3개만 받지만, 모델이 5개 변수용으로 학습되어 있으므로 실시간으로 채워줍니다.
        water_cement_ratio = water / cement if cement != 0 else 0
        
        if age <= 7:
            age_category = 1
        elif age <= 28:
            age_category = 2
        else:
            age_category = 3
            
        # 모델의 피처 형태와 순서(5개)에 맞춘 임시 데이터프레임 구조 생성
        input_data = pd.DataFrame(
            [[cement, water, age, water_cement_ratio, age_category]],
            columns=['시멘트', '물', '양생기간', '물_시멘트_비율', '양생기간_범주']
        )
        
        try:
            # 5. 각 클래스별 소속 확률 추출 및 퍼센트 계산
            probabilities = rf_model.predict_proba(input_data)[0]
            
            prob_unsafe = probabilities[0] * 100  # 0 클래스(불안전) 확률
            prob_safe = probabilities[1] * 100    # 1 클래스(안전) 확률
            
            # 확률 분포에 따라 테마 색상과 판단 문자열을 매칭
            if prob_safe >= prob_unsafe:
                status = "안전"
                confidence = prob_safe
                box_color = "#E8F8F5"  # 은은한 초록색 계열
                text_color = "#117A65"
            else:
                status = "불안전"
                confidence = prob_unsafe
                box_color = "#FADBD8"  # 은은한 붉은색 계열
                text_color = "#943126"
                
            # 6. 최종 예측 결과 카드 시각화
            st.markdown(f'''
                <div class="result-box" style="background-color: {box_color}; color: {text_color};">
                    👷 예측 결과: {status} 상태 ({confidence:.2f}% 신뢰)
                </div>
            ''', unsafe_allow_html=True)
            
            # 하단에 게이지(프로그레스 바) 형태로 확률을 뿌려 시각적 효과 부여
            st.write("")
            st.write("📊 **모델 내부 세부 확률 분포**")
            
            st.text(f"🟢 안전할 확률 (Safe): {prob_safe:.1f}%")
            st.progress(int(prob_safe))
            
            st.text(f"🔴 불안전할 확률 (Unsafe): {prob_unsafe:.1f}%")
            st.progress(int(prob_unsafe))
            
        except Exception as e:
            st.error(f"예측 작업 진행 중 오류가 발생했습니다: {e}")
