import streamlit as st
import pandas as pd
import joblib

st.title("신체 정보를 이용한 몸무게 예측 머신러닝 모델")
st.write("신체 정보를 입력하면 몸무게를 예측합니다.")

# 모델 불러오기
model = joblib.load("weight_model_male.pkl")

st.sidebar.header("머신러닝 모델 설계 실습 (다중회귀)")

# 코랩 코드의 변수 순서대로 입력 받기
height = st.slider("키 (cm)", 140.0, 190.0, 170.0)
waist = st.slider("허리 둘레 (cm)", 50.0, 120.0, 80.0)
hip = st.slider("엉덩이 둘레 (cm)", 85.0, 120.0, 100.0)
chest = st.slider("윗가슴 둘레 (cm)", 70.0, 130.0, 95.0)       # 추가
foot_size = st.slider("발 크기 (mm)", 220.0, 300.0, 260.0)    # 추가

if st.button("몸무게 예측하기"):
    # 코랩에서 쓰신 것과 완벽히 일치하는 DataFrame 생성 (변수 순서 중요!)
    input_data = pd.DataFrame(
        [[height, waist, hip, chest, foot_size]],
        columns=['키', '허리둘레', '엉덩이둘레', '윗가슴둘레', '발크기']
    )
    
    # 예측 수행
    prediction = model.predict(input_data)
    
    # 결과 출력
    st.success(f"▶ 선형회귀 예측 몸무게: {prediction[0]:.1f} kg")