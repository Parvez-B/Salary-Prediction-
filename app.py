import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

st.set_page_config(layout="wide")
st.title("Salary Analysis and Prediction App")

 
uploaded_file = st.file_uploader("Upload your 'salaries.csv' file", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.drop_duplicates(inplace=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    
    st.subheader("Salary Distribution by Experience Level")
    fig1 = plt.figure(figsize=(10, 5))
    sns.boxplot(x='experience_level', y='salary_in_usd', data=df, palette="Blues")
    st.pyplot(fig1)

    st.subheader("Employment Type Distribution")
    fig2 = plt.figure(figsize=(8, 4))
    sns.countplot(x='employment_type', data=df, palette="Set2")
    st.pyplot(fig2)

    st.subheader("Top 10 Most Common Job Titles")
    fig3 = plt.figure(figsize=(12, 6))
    top_jobs = df['job_title'].value_counts().nlargest(10)
    sns.barplot(x=top_jobs.values, y=top_jobs.index, palette="magma")
    st.pyplot(fig3)

    st.subheader("Salary Trends by Company Size")
    fig4 = plt.figure(figsize=(10, 5))
    sns.boxplot(x='company_size', y='salary_in_usd', data=df, palette="coolwarm")
    st.pyplot(fig4)

    st.subheader("Top Employee Residences vs Company Locations")
    fig5, ax = plt.subplots(1, 2, figsize=(14, 6))
    top_countries = df['employee_residence'].value_counts().nlargest(10)
    sns.barplot(x=top_countries.values, y=top_countries.index, ax=ax[0], palette="cividis")
    ax[0].set_title("Top 10 Employee Residence Locations")
    top_locations = df['company_location'].value_counts().nlargest(10)
    sns.barplot(x=top_locations.values, y=top_locations.index, ax=ax[1], palette="coolwarm")
    ax[1].set_title("Top 10 Company Locations")
    st.pyplot(fig5)

     
    df.dropna(inplace=True)
    categorical_cols = ['experience_level', 'employment_type', 'job_title', 
                        'salary_currency', 'employee_residence', 'company_location', 'company_size']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop(columns=['salary_in_usd'])
    y = df['salary_in_usd']
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.3, random_state=50)

    
    models = {
        "Linear Regression": LinearRegression(),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5)
    }

    st.subheader("Model Performance (R² Score)")
    r2_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred) * 100
        r2_results[name] = round(r2, 2)

    r2_df = pd.DataFrame(list(r2_results.items()), columns=["Model", "R² Score (%)"])
    fig6 = plt.figure(figsize=(10, 6))
    sns.barplot(x="R² Score (%)", y="Model", data=r2_df, palette="viridis")
    for index, row in r2_df.iterrows():
        plt.text(row["R² Score (%)"] + 1, index, f"{row['R² Score (%)']}%", va='center')
    st.pyplot(fig6)

   
    bins = [0, 50000, 100000, np.inf]
    y_test_cat = np.digitize(y_test, bins) - 1
    y_pred_cat = np.digitize(y_pred, bins) - 1

    cm = confusion_matrix(y_test_cat, y_pred_cat)
    acc = accuracy_score(y_test_cat, y_pred_cat)
    f1 = f1_score(y_test_cat, y_pred_cat, average='weighted')

    st.subheader("Confusion Matrix (Binned Salaries)")
    fig7, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues", ax=ax)
    st.pyplot(fig7)

    st.write(f"**Accuracy:** {acc:.2f}")
    st.write(f"**F1 Score (Weighted):** {f1:.2f}")
