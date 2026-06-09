import streamlit as st
import joblib
import numpy as np

# 1. 페이지 기본 설정 및 공사장 컨셉 디자인 (CSS)
st.set_page_config(page_title="콘크리트 강도 예측 시스템", page_icon="🚧", layout="centered")

st.markdown("""
    <style>
    /* 공사장 안전띠 느낌의 메인 배경 및 타이틀 */
    .main-title {
        background-color: #f1c40f;
        color: #000000;
        padding: 20px;
        text-align: center;
        font-weight: bold;
        border-radius: 10px;
        border: 4px dashed #000000;
        font-size: 28px;
        margin-bottom: 30px;
    }
    /* 서브 타이틀 및 강조 텍스트 */
    .safety-text {
        color: #e67e22;
        font-weight: bold;
        text-align: center;
    }
    /* 예측 버튼 스타일 */
    .stButton>button {
        background-color: #2c3e50 !important;
        color: white !important;
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
        height: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# 타이틀 표시
st.markdown('<div class="main-title">🚧 안전제일: 콘크리트 강도 예측 시스템 🚧</div>', unsafe_allow_html=True)
st.markdown('<p class="safety-text">※ 정확한 배합 정보를 입력하여 콘크리트의 안전성(압축강도)을 진단하세요.</p>', unsafe_allow_html=True)
st.write("---")

# 2. 모델 로드 (에러 예외 처리)
@st.cache_resource
def load_model():
    try:
        # VS Code 내 같은 폴더에 pkl 파일이 있어야 합니다.
        return joblib.load("concrete_model.pkl")
    except FileNotFoundError:
        st.error("🚨 'concrete_model.pkl' 파일을 찾을 수 없습니다. 모델 파일을 app.py와 같은 폴더에 넣어주세요!")
        return None

model = load_model()

# 3. 사용자 입력 양식 구성 (7가지 배합 성분 + 재령 데이터 수집)
st.subheader("🏗️ 콘크리트 성분 배합비 입력")

col1, col2 = st.columns(2)

with col1:
    cement = st.number_input("시멘트량 (Cement - kg/m³)", min_value=0.0, value=250.0, step=10.0)
    blast_furnace_slag = st.number_input("고로슬래그 미분말 (Slag - kg/m³)", min_value=0.0, value=0.0, step=10.0)
    fly_ash = st.number_input("플라이애시 (Fly Ash - kg/m³)", min_value=0.0, value=0.0, step=10.0)
    water = st.number_input("배합수량 (Water - kg/m³)", min_value=0.0, value=180.0, step=5.0)

with col2:
    superplasticizer = st.number_input("고성능 감수제 (Superplasticizer - kg/m³)", min_value=0.0, value=0.0, step=1.0)
    coarse_aggregate = st.number_input("굵은 골재 (Coarse Aggregate - kg/m³)", min_value=0.0, value=900.0, step=10.0)
    fine_aggregate = st.number_input("잔골재 (Fine Aggregate - kg/m³)", min_value=0.0, value=800.0, step=10.0)
    age = st.number_input("재령 기간 (Age - 일 수)", min_value=1, value=28, step=1)

st.write("---")

# 4. 예측 수행 및 결과 출력
if st.button("📊 콘크리트 압축 강도 예측하기"):
    if model is not None:
        # 모델 입력 형태에 맞게 2차원 배열 데이터 생성
        input_data = np.array([[cement, blast_furnace_slag, fly_ash, water, 
                                superplasticizer, coarse_aggregate, fine_aggregate, age]])
        
        # 수행평가 파일 기준 '분류 모델(정확도 87.2%)'로 설계되어 있으므로 predict 결과를 받음
        prediction = model.predict(input_data)[0]
        
        st.subheader("👷 예측 결과 보고서")
        
        # 분류 모델 결과에 따른 공사장 스타일 피드백 안내
        if prediction == 1: # (또는 모델의 타겟 클래스 기준에 맞춰 변경 가능)
            st.success("✅ **[안전]** 본 배합 비율은 목표 압축강도 기준을 만족하며 안전성이 우수합니다.")
        else:
            st.error("🚨 **[위험/부실 위험]** 본 배합 비율은 강도가 부족할 위험이 있습니다. 배합을 재조정하십시오.")