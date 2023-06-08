#IMPORT LIBRARI
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree


st.title("KECERDASAN KOMPUTASIONAL C")
st.write("Nama  : Nuriyah Amelia Febrianti")
st.write("NIM   : 2104111000168")

# Tampilan Aturan Navbarnya 
upload_data, preprocessing, modeling , NaiveBayes, DecisionTree = st.tabs(["Upload Data", "Preprocessing", "Modeling","NaiveBayes","DecisionTree"])

df = pd.read_csv('https://raw.githubusercontent.com/NuriyahFebrianti/data/main/DATASET.csv')

#Uploud data
with upload_data:
    uploaded_file = st.file_uploader("Upload file CSV")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.header("Dataset")
        st.dataframe(df)

#Preprocessing
with preprocessing:
    st.subheader("Normalisasi Data")
    X = df.drop(columns=["Penyakit"])
    y = df["Penyakit"].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    feature_names = X.columns.copy()
    scaled_features = pd.DataFrame(scaled, columns=feature_names)

    st.subheader("Hasil Normalisasi Data")
    st.dataframe(scaled_features)

    st.subheader("Target Label")
    labels = pd.get_dummies(df["Penyakit"])
    st.dataframe(labels)

#Modelling
with modeling:
    training, test, training_label, test_label = train_test_split(scaled_features, y, test_size=0.2, random_state=100)

    st.subheader("Modeling")
    model = LogisticRegression()
    model.fit(training, training_label)

    y_pred = model.predict(test)
    accuracy = accuracy_score(test_label, y_pred)

    st.write("Akurasi Model: ", accuracy)

    # Plotting grafik akurasi
    plt.figure(figsize=(8, 6))
    plt.bar(['Akurasi'], [accuracy])
    plt.ylim([0, 1])
    plt.ylabel('Akurasi')
    plt.title('Grafik Akurasi Model')
    st.pyplot(plt)
with NaiveBayes:
    training, test, training_label, test_label = train_test_split(scaled_features, y, test_size=0.2, random_state=100)

    st.subheader("NaiveBayes")
    model1 = MultinomialNB()
    model1.fit(training, training_label)

    y_pred1 = model1.predict(test)
    accuracy1 = accuracy_score(test_label, y_pred)

    st.write("Akurasi Model: ", accuracy1)

    plt.figure(figsize=(8, 6))
    plt.bar(['Akurasi'], [accuracy1])
    plt.ylim([0, 1])
    plt.ylabel('Akurasi')
    plt.title('Grafik Akurasi Model')
    st.pyplot(plt)

with DecisionTree:
    training, test, training_label, test_label = train_test_split(scaled_features, y, test_size=0.2, random_state=100)

    st.subheader("DecisionTree")
    model2 = DecisionTreeClassifier()
    model2.fit(training, training_label)
    
    y_pred2 = model2.predict(test)
    accuracy2 = accuracy_score(test_label, y_pred2)

    st.write("Akurasi Model: ", accuracy2)
    
    tree.plot_tree(accuracy2)

    plt.figure(figsize=(8, 6))
    plt.bar(['Akurasi'], [accuracy2])
    plt.ylim([0, 1])
    plt.ylabel('Akurasi')
    plt.title('Grafik Akurasi Model')
    st.pyplot(plt)





