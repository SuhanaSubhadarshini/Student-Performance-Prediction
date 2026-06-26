import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Student Performance Predictor")

st.title("🎓 Student Performance Predictor")

# Numerical Features
hours_studied = st.slider("Hours Studied", 0, 50, 10)
attendance = st.slider("Attendance (%)", 0, 100, 75)
sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
previous_scores = st.slider("Previous Scores", 0, 100, 60)
tutoring_sessions = st.slider("Tutoring Sessions", 0, 10, 2)
physical_activity = st.slider("Physical Activity (hours/week)", 0, 20, 5)

# Categorical Features
parental_involvement = st.selectbox(
    "Parental Involvement",
    ["High", "Medium", "Low"]
)

access_resources = st.selectbox(
    "Access to Resources",
    ["High", "Medium", "Low"]
)

extracurricular = st.selectbox(
    "Extracurricular Activities",
    ["No", "Yes"]
)

motivation = st.selectbox(
    "Motivation Level",
    ["High", "Medium", "Low"]
)

internet = st.selectbox(
    "Internet Access",
    ["No", "Yes"]
)

family_income = st.selectbox(
    "Family Income",
    ["High", "Medium", "Low"]
)

teacher_quality = st.selectbox(
    "Teacher Quality",
    ["High", "Medium", "Low"]
)

school_type = st.selectbox(
    "School Type",
    ["Private", "Public"]
)

peer_influence = st.selectbox(
    "Peer Influence",
    ["Negative", "Neutral", "Positive"]
)

learning_disability = st.selectbox(
    "Learning Disabilities",
    ["No", "Yes"]
)

parent_education = st.selectbox(
    "Parental Education Level",
    ["College", "High School", "Postgraduate"]
)

distance = st.selectbox(
    "Distance From Home",
    ["Far", "Moderate", "Near"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

if st.button("Predict Score"):

    data = {
        'Hours_Studied': hours_studied,
        'Attendance': attendance,
        'Sleep_Hours': sleep_hours,
        'Previous_Scores': previous_scores,
        'Tutoring_Sessions': tutoring_sessions,
        'Physical_Activity': physical_activity,

        'Parental_Involvement_Low': 1 if parental_involvement == "Low" else 0,
        'Parental_Involvement_Medium': 1 if parental_involvement == "Medium" else 0,

        'Access_to_Resources_Low': 1 if access_resources == "Low" else 0,
        'Access_to_Resources_Medium': 1 if access_resources == "Medium" else 0,

        'Extracurricular_Activities_Yes': 1 if extracurricular == "Yes" else 0,

        'Motivation_Level_Low': 1 if motivation == "Low" else 0,
        'Motivation_Level_Medium': 1 if motivation == "Medium" else 0,

        'Internet_Access_Yes': 1 if internet == "Yes" else 0,

        'Family_Income_Low': 1 if family_income == "Low" else 0,
        'Family_Income_Medium': 1 if family_income == "Medium" else 0,

        'Teacher_Quality_Low': 1 if teacher_quality == "Low" else 0,
        'Teacher_Quality_Medium': 1 if teacher_quality == "Medium" else 0,

        'School_Type_Public': 1 if school_type == "Public" else 0,

        'Peer_Influence_Neutral': 1 if peer_influence == "Neutral" else 0,
        'Peer_Influence_Positive': 1 if peer_influence == "Positive" else 0,

        'Learning_Disabilities_Yes': 1 if learning_disability == "Yes" else 0,

        'Parental_Education_Level_High School': 1 if parent_education == "High School" else 0,
        'Parental_Education_Level_Postgraduate': 1 if parent_education == "Postgraduate" else 0,

        'Distance_from_Home_Moderate': 1 if distance == "Moderate" else 0,
        'Distance_from_Home_Near': 1 if distance == "Near" else 0,

        'Gender_Male': 1 if gender == "Male" else 0
    }

    input_df = pd.DataFrame([data])

    # Match training columns exactly
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    st.success(f"📈 Predicted Exam Score: {prediction[0]:.2f}")