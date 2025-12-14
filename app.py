import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)


# Page Config
st.set_page_config(page_title="Student Performance Analysis", layout="wide")

st.title("🎓 Student Academic Performance Analysis")
st.markdown("### Data Analysis & Machine Learning Dashboard")


# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("SAP-4000.csv")

    # column normalization
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # categorical normalization
    cat_cols = df.select_dtypes(include='object').columns
    for c in cat_cols:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    # missing values
    df.parent_education.fillna('none', inplace=True)
    return df

df = load_data()

categorical = ['gender','tutoring','region','parent_education']
numerical   = ['hoursstudied/week', 'attendance(%)']


# Sidebar
st.sidebar.title(" Navigation")
section = st.sidebar.radio(
    "Go to:",
    [
        "Dataset Overview",
        "EDA & Visualizations",
        "Correlation Analysis",
        "Regression Model (Random Forest)",
        "Classification (Pass / Fail)",
        " Student Prediction"
    ]
)



# Dataset Overview
if section == "Dataset Overview":
    st.subheader(" Dataset Overview")

    st.write("Shape of dataset:", df.shape)
    st.dataframe(df.head())

    st.subheader("Data Info")
    st.write(df.dtypes)

    st.subheader("Missing Values (%)")
    missing = df.isnull().sum() * 100 / len(df)
    st.write(missing)


# EDA & Visualizations
elif section == "EDA & Visualizations":
    st.subheader(" Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Exam Score Distribution**")
        fig, ax = plt.subplots()
        sns.histplot(df['exam_score'], bins=25, kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("**Exam Score by Gender**")
        fig, ax = plt.subplots()
        sns.boxplot(x='gender', y='exam_score', data=df, ax=ax)
        st.pyplot(fig)

    st.markdown("**Hours Studied vs Exam Score**")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x='hoursstudied/week',
        y='exam_score',
        data=df,
        hue='gender',
        alpha=0.6,
        ax=ax
    )
    sns.regplot(
        x='hoursstudied/week',
        y='exam_score',
        data=df,
        scatter=False,
        color='black',
        ax=ax
    )
    st.pyplot(fig)


# Correlation
elif section == "Correlation Analysis":
    st.subheader(" Correlation Analysis")

    numeric_cols = ['hoursstudied/week', 'attendance(%)', 'exam_score']
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)


# Train / Val / Test Split
X = df[categorical + numerical]
y = df.exam_score

x_train, x_, y_train, y_ = train_test_split(
    X, y, test_size=0.3, random_state=42
)

x_val, x_test, y_val, y_test = train_test_split(
    x_, y_, test_size=0.5, random_state=42
)

dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(x_train.to_dict(orient='records'))
X_val   = dv.transform(x_val.to_dict(orient='records'))
X_test  = dv.transform(x_test.to_dict(orient='records'))


# Regression Model
if section == "Regression Model (Random Forest)":
    st.subheader(" Random Forest Regressor")

    rf = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=40,
        random_state=42
    )
    rf.fit(X_train, y_train)

    y_val_pred = rf.predict(X_val)
    y_test_pred = rf.predict(X_test)

    col1, col2, col3 = st.columns(3)

    col1.metric("Validation RMSE", f"{np.sqrt(mean_squared_error(y_val, y_val_pred)):.2f}")
    col2.metric("Test RMSE", f"{np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}")
    col3.metric("Test R²", f"{r2_score(y_test, y_test_pred):.2f}")

    results = pd.DataFrame({
        "Actual Score": y_test.values,
        "Predicted Score": y_test_pred
    })

    st.markdown("### Sample Predictions")
    st.dataframe(results.head(10))

# Classification
elif section == "Classification (Pass / Fail)":
    st.subheader(" Pass / Fail Classification (Naive Bayes)")

    y_train_cls = (y_train >= 50).astype(int)
    y_val_cls   = (y_val >= 50).astype(int)
    y_test_cls  = (y_test >= 50).astype(int)

    nb = GaussianNB()
    nb.fit(X_train, y_train_cls)

    y_pred = nb.predict(X_test)

    st.metric("Test Accuracy", f"{accuracy_score(y_test_cls, y_pred):.2f}")

    st.markdown("### Classification Report")
    st.text(classification_report(y_test_cls, y_pred))

    cm = confusion_matrix(y_test_cls, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Fail','Pass'],
        yticklabels=['Fail','Pass'],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


# Student Prediction
elif section == " Student Prediction":
    st.subheader(" Predict Student Result")

    st.markdown("Enter student information to predict **Exam Score** and **Pass / Fail**")

    # -------- User Inputs --------
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["male", "female"])
        tutoring = st.selectbox("Private Tutoring", ["yes", "no"])
        region = st.selectbox("Region", ["urban", "rural"])
        parent_education = st.selectbox(
            "Parent Education",
            ["none", "primary", "secondary", "tertiary"]
        )

    with col2:
        hours = st.slider("Hours Studied per Week", 0, 60, 20)
        attendance = st.slider("Attendance (%)", 0, 100, 75)

    # Prepare Input 
    input_dict = [{
        "gender": gender,
        "tutoring": tutoring,
        "region": region,
        "parent_education": parent_education,
        "hoursstudied/week": hours,
        "attendance(%)": attendance
    }]

    X_user = dv.transform(input_dict)

    #  Train Models (reuse trained ones)
    rf = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=40,
        random_state=42
    )
    rf.fit(X_train, y_train)

    nb = GaussianNB()
    nb.fit(X_train, (y_train >= 50).astype(int))

    #  Prediction Button 
    if st.button(" Predict Result"):
        predicted_score = rf.predict(X_user)[0]
        predicted_class = nb.predict(X_user)[0]

        st.markdown("---")
        st.markdown("###  Prediction Result")

        st.metric("Predicted Exam Score", f"{predicted_score:.1f}")

        if predicted_class == 1:
            st.success(" Result: PASS")
        else:
            st.error(" Result: FAIL")

        # Confidence-style message
        if predicted_score >= 85:
            st.info(" Excellent performance expected!")
        elif predicted_score >= 65:
            st.info(" Good performance")
        elif predicted_score >= 50:
            st.warning(" Borderline - needs improvement")
        else:
            st.warning(" High risk of failure")
