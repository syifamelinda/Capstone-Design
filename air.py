import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import base64
import io
import requests
from PIL import ImageDraw
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
import pandas as pd
from sklearn import svm, tree
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.linalg import pinv, inv
import time
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from scipy.linalg import pinv, inv
import time
import numpy as np
from numpy.linalg import pinv, inv
import time
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from numpy.linalg import pinv, inv
import time
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Create sidebar with dropdown menu
with st.sidebar:
    selected = option_menu("", ["Beranda", "Informasi", "Credits"])

# Display selected page
if selected == "Beranda":
    with st.sidebar:
        selected_method = st.selectbox("Pilih Metode", ["K-Nearest Neighbor", "Decision Tree", "Extreme Learning Machine"])
        input_option = st.selectbox("Masukkan Data", ["Manual", "Upload File"])

    # Tampilkan judul halaman
    st.markdown(
        """
        <style>
        .title {
            color: black;
            font-size: 36px;
            margin: 0;
            padding: 20px 0;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='custom-box'><h1 class='title'>Uji Kelayakan Air Minum</h1></div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<center>Masukkan data kandungan air minum anda, untuk mengetahui kelayakannya!</center>"
        + "<center>Sebelumnya silahkan membaca buku panduannya terlebih dahulu <a href='https://drive.google.com/drive/folders/1LvNsC55XPSoXlDhV0L36W8pM2YjZvEBb?usp=sharing' target='_blank'>disini</a> !</center>"
        + "<hr style='border: 1px solid black;'><br>",
        unsafe_allow_html=True
    )
    

    class elm():
        def __init__(self, hidden_units, activation_function, x, y, C, elm_type, one_hot=True, random_type='normal'):
            self.hidden_units = hidden_units
            self.activation_function = activation_function
            self.random_type = random_type
            self.x = x
            self.y = y
            self.C = C
            self.class_num = np.unique(self.y).shape[0]
            self.beta = np.zeros((self.hidden_units, self.class_num))
            self.elm_type = elm_type
            self.one_hot = one_hot

            if elm_type == 'clf' and self.one_hot:
                self.one_hot_label = np.zeros((self.y.shape[0], self.class_num))
                for i in range(self.y.shape[0]):
                    self.one_hot_label[i, int(self.y[i])] = 1

            if self.random_type == 'uniform':
                self.W = np.random.uniform(low=0, high=1, size=(self.hidden_units, self.x.shape[1]))
                self.b = np.random.uniform(low=0, high=1, size=(self.hidden_units, 1))
            if self.random_type == 'normal':
                self.W = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, self.x.shape[1]))
                self.b = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, 1))

        def __input2hidden(self, x):
            x = np.array(x, dtype=np.float64)  # Mengubah tipe data x menjadi float64
            self.temH = np.dot(self.W, x.T) + self.b

            if self.activation_function == 'sigmoid':
                self.H = 1 / (1 + np.exp(-self.temH))

            if self.activation_function == 'relu':
                self.H = self.temH * (self.temH > 0)

            if self.activation_function == 'sin':
                self.H = np.sin(self.temH)

            if self.activation_function == 'tanh':
                self.H = (np.exp(self.temH) - np.exp(-self.temH)) / (np.exp(self.temH) + np.exp(-self.temH))

            if self.activation_function == 'leaky_relu':
                self.H = np.maximum(0, self.temH) + 0.1 * np.minimum(0, self.temH)

            return self.H

        def __hidden2output(self, H):
            self.output = np.dot(H.T, self.beta)
            return self.output

        def fit(self, algorithm):
            self.time1 = time.perf_counter()
            self.H = self.__input2hidden(self.x)
            if self.elm_type == 'clf':
                if self.one_hot:
                    self.y_temp = self.one_hot_label
                else:
                    self.y_temp = self.y
            if self.elm_type == 'reg':
                self.y_temp = self.y

            if algorithm == 'no_re':
                self.beta = np.dot(pinv(self.H.T), self.y_temp)

            if algorithm == 'solution1':
                self.tmp1 = inv(np.eye(self.H.shape[0]) / self.C + np.dot(self.H, self.H.T))
                self.tmp2 = np.dot(self.tmp1, self.H)
                self.beta = np.dot(self.tmp2, self.y_temp)

            if algorithm == 'solution2':
                self.tmp1 = inv(np.eye(self.H.shape[0]) / self.C + np.dot(self.H, self.H.T))
                self.tmp2 = np.dot(self.H.T, self.tmp1)
                self.beta = np.dot(self.tmp2.T, self.y_temp)
            self.time2 = time.perf_counter()

            self.result = self.__hidden2output(self.H)

            if self.elm_type == 'clf':
                self.result = np.exp(self.result) / np.sum(np.exp(self.result), axis=1).reshape(-1, 1)

            if self.elm_type == 'clf':
                self.y_ = np.where(self.result == np.max(self.result, axis=1).reshape(-1, 1))[1]
                self.correct = 0
                for i in range(self.y.shape[0]):
                    if self.y_[i] == self.y[i]:
                        self.correct += 1
                self.train_score = self.correct / self.y.shape[0]
            if self.elm_type == 'reg':
                self.train_score = np.sqrt(np.sum((self.result - self.y) * (self.result - self.y)) / self.y.shape[0])
            train_time = str(self.time2 - self.time1)
            return self.beta, self.train_score, train_time

        def predict(self, x):
            x = np.array(x, dtype=np.float64)  # Mengubah tipe data x menjadi float64
            self.H = self.__input2hidden(x)
            self.y_ = self.__hidden2output(self.H)
            if self.elm_type == 'clf':
                self.y_ = np.where(self.y_ == np.max(self.y_, axis=1).reshape(-1, 1))[1]

            return self.y_

        def predict_proba(self, x):
            x = np.array(x, dtype=np.float64)  # Mengubah tipe data x menjadi float64
            self.H = self.__input2hidden(x)
            self.y_ = self.__hidden2output(self.H)
            if self.elm_type == 'clf':
                self.proba = np.exp(self.y_) / np.sum(np.exp(self.y_), axis=1).reshape(-1, 1)
            return self.proba

        def score(self, x, y):
            self.prediction = self.predict(x)
            if self.elm_type == 'clf':
                self.correct = 0
                for i in range(y.shape[0]):
                    if self.prediction[i] == y[i]:
                        self.correct += 1
                self.test_score = self.correct / y.shape[0]
            if self.elm_type == 'reg':
                self.test_score = np.sqrt(np.sum((self.result - self.y) * (self.result - self.y)) / self.y.shape[0])
            return self.test_score




    if input_option == "Manual":
        

        col1, col2, col3, col4, col5 = st.columns(5)
        

        with col1:
            EColli = st.text_input('E.Colli', key='E.Colli', value='', placeholder='0 jml/100 mL')
            st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)

        with col2:
            Coliform = st.text_input('Coliform', key='Coliform', value='', placeholder='0 jml/100 mL')
            st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)

        with col3:
            Arsen = st.text_input('Arsen', key='Arsen', value='', placeholder='0 jml/100 mL')
            st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)

        with col4:
            Kromium = st.text_input('Kromium', key='Kromium', value='', placeholder='0 jml/100 mL')
            st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)
            
        with col5:
            Kadmium = st.text_input('Kadmium', key='Kadmium', value='', placeholder='0 jml/100 mL')
            st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)

        with col1:
            Nitrit = st.text_input('Nitrit', key='Nitrit', value='', placeholder='0 jml/100 mL')
            st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)

        with col2:
            Nitrat = st.text_input('Nitrat', key='Nitrat', value='', placeholder='0 jml/100 mL')
            st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)

        with col3:
            Sianida = st.text_input('Sianida', key='Sianida', value='', placeholder='0 jml/100 mL')
            st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)

        with col4:
            Selenium = st.text_input('Selenium', key='Selenium', value='', placeholder='0 jml/100 mL')
            st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)
        
        with col5:
            Alumunium = st.text_input('Alumunium', key='Alumunium', value='', placeholder='0 jml/100 mL')
            st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)

        with col1:
            Besi = st.text_input('Besi', key='Besi', value='', placeholder='0 jml/100 mL')
            st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)

        with col2:
            Kesadahan = st.text_input('Kesadahan', key='Kesadahan', value='', placeholder='0 jml/100 mL')
            st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)

        with col3:
            Klorida = st.text_input('Klorida', key='Klorida', value='', placeholder='0 jml/100 mL')
            st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)

        with col4:
            Mangan = st.text_input('Mangan', key='Mangan',  value='', placeholder='0 jml/100 mL')
            st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)
        
        with col5:
            pH = st.text_input('pH', key='pH', value='', placeholder='0 jml/100 mL')
            st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)

        with col1:
            Seng = st.text_input('Seng', key='Seng', value='', placeholder='0 jml/100 mL')
           st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)

        with col2:
            Sulfat = st.text_input('Sulfat', key='Sulfat', value='', placeholder='0 jml/100 mL')
           st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)

        with col3:
            Tembaga = st.text_input('Tembaga', key='Tembaga', value='', placeholder='0 jml/100 mL')
           st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)

        with col4:
            Amonia = st.text_input('Amonia', key='Amonia', value='', placeholder='0 jml/100 mL')
           st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)
        
        with col5:
            Chlor = st.text_input('Chlor', key='Chlor', value='', placeholder='0 jml/100 mL')
            st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)

        with col1:
            Bau = st.text_input('Bau', key='Bau', value='', placeholder='0 jml/100 mL')
            st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)

        with col2:
            Warna = st.text_input('Warna', key='Warna', value='', placeholder='0 jml/100 mL')
           st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)

        with col3:
            Kekeruhan = st.text_input('Kekeruhan', key='Kekeruhan', value='', placeholder='0 jml/100 mL')
            st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)

        with col4:
            Rasa = st.text_input('Rasa ', key='Rasa ', value='', placeholder='0 jml/100 mL')
          st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)
        
        with col5:
            TDS = st.text_input('TDS', key='TDS', value='', placeholder='0 jml/100 mL')
          st.markdown('<style>.stTextInput input::placeholder { color: #7ab8bf; }</style>', unsafe_allow_html=True)
        
        if st.button('Cek Kelayakan'):
            # Perform prediction based on the selected method
            if selected_method == "Decision Tree":
                with open("Decisiontree.sav", 'rb') as file:
                    model_data = pickle.load(file)
        
                kelayakan_model = model_data
        
                # Preprocess the input data
                input_data = [EColli, Coliform, Arsen, Kromium, Kadmium, Nitrat, Nitrit, Sianida, Selenium, Alumunium, Besi, Kesadahan, Klorida, Mangan, pH, Seng, Sulfat, Tembaga, Amonia, Chlor, Bau, Warna, Kekeruhan, Rasa, TDS]
        
                # Convert non-numeric fields to numeric values
                input_data = [float(value) if isinstance(value, str) else value for value in input_data]
        
                # Perform scaling on the input data
                scaler = StandardScaler()
                input_data_scaled = scaler.fit_transform([input_data])
        
                # Make predictions
                air_prediction = model_data.predict(input_data_scaled)
        
                if air_prediction[0] == 0:
                    air_diagnosis = 'Air Layak Minum'
                else:
                    air_diagnosis = 'Air Tidak Layak Minum'
        
                #st.success(air_diagnosis)
                # ...
                #pass
            
            elif selected_method == "K-Nearest Neighbor":
                with open("K-Nearest Neighbors.sav", 'rb') as file:
                    model_data = pickle.load(file)
        
                kelayakan_model = model_data
        
                # Preprocess the input data
                input_data = [EColli, Coliform, Arsen, Kromium, Kadmium, Nitrat, Nitrit, Sianida, Selenium, Alumunium, Besi, Kesadahan, Klorida, Mangan, pH, Seng, Sulfat, Tembaga, Amonia, Chlor, Bau, Warna, Kekeruhan, Rasa, TDS]
        
                # Convert non-numeric fields to numeric values
                input_data = [float(value) if isinstance(value, str) else value for value in input_data]
        
                # Perform scaling on the input data
                scaler = StandardScaler()
                input_data_scaled = scaler.fit_transform([input_data])
        
                # Make predictions
                air_prediction = model_data.predict(input_data_scaled)
        
                if air_prediction[0] == 0:
                    air_diagnosis = 'Air Layak Minum'
                else:
                    air_diagnosis = 'Air Tidak Layak Minum'
        
                #st.success(air_diagnosis)
                # ...
                #pass
        
            elif selected_method == "Extreme Learning Machine":
                import pickle
                
                with open('ELM-FRISKA_FIX.sav', 'rb') as file:
                    model_data = pickle.load(file)
        
                kelayakan_model = model_data

                # Preprocess the input data
                input_data = [EColli, Coliform, Arsen, Kromium, Kadmium, Nitrat, Nitrit, Sianida, Selenium, Alumunium, Besi, Kesadahan, Klorida, Mangan, pH, Seng, Sulfat, Tembaga, Amonia, Chlor, Bau, Warna, Kekeruhan, Rasa, TDS]
                
                # Convert non-numeric fields to numeric values
                input_data = [float(value) if isinstance(value, str) else value for value in input_data]
                
                # Perform scaling on the input data
                scaler = StandardScaler()
                input_data_scaled = scaler.fit_transform([input_data])
                
                # Make predictions
                air_prediction = kelayakan_model.predict(input_data_scaled)
                
                if air_prediction[0] == 1:
                    air_diagnosis = 'Air Layak Minum'
                else:
                    air_diagnosis = 'Air Tidak Layak Minum'
                
                #st.success(air_diagnosis)
                # ...
                #pass
            
            # Check if input values are within the specified limits
            is_within_limits = (
                0 <= float(EColli) <= 0.0 and
                0 <= float(Coliform) <= 0.0 and
                0 <= float(Arsen) <= 0.01 and
                0 <= float(Kromium) <= 0.05 and
                0 <= float(Kadmium) <= 0.003 and
                0 <= float(Nitrat) <= 50 and
                0 <= float(Nitrit) <= 3 and
                0 <= float(Sianida) <= 0.07 and
                0 <= float(Selenium) <= 0.01 and
                0 <= float(Alumunium) <= 0.2 and
                0 <= float(Besi) <= 0.3 and
                0 <= float(Kesadahan) <= 500 and
                0 <= float(Klorida) <= 250 and
                0 <= float(Mangan) <= 0.4 and
                6.5 <= float(pH) <= 8.5 and
                0 <= float(Seng) <= 3 and
                0 <= float(Sulfat) <= 250 and
                0 <= float(Tembaga) <= 2 and
                0 <= float(Amonia) <= 1.5 and
                0.2 <= float(Chlor) <= 1.0 and
                float(Bau) == 1 and
                0 <= float(Warna) <= 15 and
                0 <= float(Kekeruhan) <= 5 and
                float(Rasa) == 1 and
                float(TDS) <= 500
            )
            # Update 'Potabilitas' value based on is_within_limits
            if is_within_limits:
                potabilitas_value = 1 if air_diagnosis == 'Air Layak Minum' else 0
                st.success(air_diagnosis)
            else:
                potabilitas_value = 0
                st.warning("Data Air Minum yang Anda Masukkan Berada Diluar Baku Mutu Kualitas Air, Air Tidak Layak Minum.")
        
            # Create a dictionary with the input data and prediction result
            result = {
                'EColli': EColli,
                'Coliform': Coliform,
                'Arsen': Arsen,
                'Kromium': Kromium,
                'Kadmium': Kadmium,
                'Nitrat': Nitrat,
                'Nitrit': Nitrit,
                'Sianida': Sianida,
                'Selenium': Selenium,
                'Alumunium': Alumunium,
                'Besi': Besi,
                'Kesadahan': Kesadahan,
                'Klorida': Klorida,
                'Mangan': Mangan,
                'pH': pH,
                'Seng': Seng,
                'Sulfat': Sulfat,
                'Tembaga': Tembaga,
                'Amonia': Amonia,
                'Chlor': Chlor,
                'Bau': Bau,
                'Warna': Warna,
                'Kekeruhan': Kekeruhan,
                'Rasa':Rasa,
                'TDS': TDS,
                'Potabilitas': potabilitas_value  # Updated 'Potabilitas' value
            }
            
            # Create a dataframe from the result dictionary
            df_result = pd.DataFrame([result])
            
            # Load existing data from CSV file
            try:
                existing_data = pd.read_csv('hasil_prediksi.csv')
            except FileNotFoundError:
                existing_data = pd.DataFrame()
            
            # Append new data to existing data
            existing_data = existing_data.append(df_result, ignore_index=True)
            
            # Save the combined data to CSV file
            existing_data.to_csv('hasil_prediksi.csv', index=False)
            
            # Display the updated result dataframe
            st.write(existing_data)
            
            # Create download button
            button_id = str(time.time())  # Unique ID for the button
            button_label = 'Download CSV'
            
            # Generate CSV file and get its content as base64
            csv = existing_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            
            # Generate button CSS style
            button_style = """
                <style>
                .download-button {
                    background-color: #538cc6;
                    border: none;
                    color: white;
                    padding: 8px 18px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 15px;
                    margin: 3px 2px;
                    cursor: pointer;
                    border-radius: 4px;
                }
                </style>
            """
            
            # Display the download button
            st.markdown(button_style, unsafe_allow_html=True)
            st.markdown(f'<a href="data:file/csv;base64,{b64}" download="hasil_prediksi.csv"><button class="download-button">{button_label}</button></a>', unsafe_allow_html=True)







    elif input_option == "Upload File":
        # Membaca file CSV yang diunggah
        uploaded_file = st.file_uploader("Unggah file CSV", type="csv")
    
        if uploaded_file is not None:
            # Baca dataset dari file CSV
            df = pd.read_csv(uploaded_file)
    
            # Preprocessing data
            df['Potabilitas'] = df['Potabilitas'].astype('category')  # Ubah tipe data kolom Potabilitas menjadi category
    
            # Label Encoder untuk kolom Rasa dan Bau
            label_encoder = LabelEncoder()
            df['Rasa '] = label_encoder.fit_transform(df['Rasa '])
            df['Bau'] = label_encoder.fit_transform(df['Bau'])
    
            # Drop kolom BOD5, COD, dan Suhu
            df = df.drop(['BOD5', 'COD', 'Suhu'], axis=1)
    
            # Bagi data menjadi fitur (X) dan label (y)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
    
            # Inisialisasi model klasifikasi berdasarkan pilihan metode
            if selected_method == "K-Nearest Neighbor":
                clf = KNeighborsClassifier()
                model_file = 'K-Nearest Neighbors.sav'
            elif selected_method == "Decision Tree":
                clf = tree.DecisionTreeClassifier()
                model_file = 'Decisiontree.sav'
            elif selected_method == "Extreme Learning Machine":
                clf = MLPClassifier()
                model_file = 'ELM-FRISKA_FIX.sav'
    
            # Load model yang sudah disimpan sebelumnya
            model = joblib.load(model_file)
    
            # Prediksi label untuk data yang baru diunggah
            y_pred = model.predict(X)
    
            # Tampilkan dataset setelah preprocessing
            st.write(df)
    
            # Tambahkan kolom "Potabilitas" ke data frame
            df['Potabilitas'] = np.where(y_pred == 1, 'Air Layak Minum', 'Air Tidak Layak Minum')
    
            # Tampilkan hasil klasifikasi dengan kolom "Potabilitas"
            # Tampilkan hasil klasifikasi dengan kolom "Potabilitas"
            st.markdown("<h2 style='text-align: center;'>Hasil Klasifikasi</h2>", unsafe_allow_html=True)
            st.dataframe(df)


            # Hitung proporsi potabilitas
            potabilitas_counts = df['Potabilitas'].value_counts()
            
            # Tampilkan diagram batang proporsi potabilitas
            colors = ['#336B87', '#f63366']  # Daftar warna dengan panjang 2
            plt.figure(figsize=(8, 6))
            sns.barplot(x=potabilitas_counts.index, y=potabilitas_counts.values, palette=colors)  # Menggunakan daftar warna
            plt.xlabel('Potabilitas')
            plt.ylabel('Count')
            st.pyplot(plt)
    
            # Tombol download
            def download_csv():
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # Encode ke base64
                href = f'<a href="data:file/csv;base64,{b64}" download="hasil_klasifikasi.csv"><button style="padding: 0.5rem 1rem; background-color: #f63366; color: white; border: none; border-radius: 4px; cursor: pointer;">Download Hasil Klasifikasi</button></a>'
                st.markdown(href, unsafe_allow_html=True)
    
            download_csv()


        
    
    
    
    
    
    
                                    
    













            












    st.write("\n")
    st.write("\n")   
    option = st.radio("Keterangan dan Penjelasan", ("Baku Mutu Kualitas Air", "Parameter Pengujian Kualitas Air"))

    if option == "Baku Mutu Kualitas Air":

        with st.container():
            st.markdown('''
            <div>
                <h1 style="font-size: 30px;">Baku Mutu Kualitas Air</h1>
                <hr style="width: 99%; border-style: solid; border-width: 1px; margin-top: 2px; margin-bottom: 7px;">
            </div>
            <p>Baku Mutu Kualitas Air di bawah ini menjelaskan persyaratan maksimum yang harus diperhatikan untuk setiap parameter guna memastikan kepatuhan terhadap standar kualitas air yang ditetapkan Permenkes.</p>
        ''', unsafe_allow_html=True)
            
            with st.container():
                columns = st.columns(4)
                with columns[0]:
                    st.markdown('''
                        <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">E.Colli</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">0 jml/100 mL</span>
                        </h4>
                    ''', unsafe_allow_html=True)

                with columns[1]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Coliform</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">0 jml/100 mL</span>
                        </h4>
                ''', unsafe_allow_html=True)

                with columns[2]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Arsen</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">0.01 mg/L</span>
                        </h4>
                ''', unsafe_allow_html=True)
                
                with columns[3]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Kromium</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">0.05 mg/L</span>
                        </h4>
                ''', unsafe_allow_html=True)
                    
            with st.container():
                columns = st.columns(4)
                with columns[0]:
                    st.markdown('''
                        <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Kadmium</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">0.003 mg/L</span>
                        </h4>
                    ''', unsafe_allow_html=True)

                with columns[1]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Nitrit</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">3 mg/L</span>
                        </h4>
                ''', unsafe_allow_html=True)

                with columns[2]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Nitrat</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">50 mg/L</span>
                        </h4>
                ''', unsafe_allow_html=True)
                
                with columns[3]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Sianida</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">0.07 mg/L</span>
                        </h4>
                ''', unsafe_allow_html=True)
            
            with st.container():
                columns = st.columns(4)
                with columns[0]:
                    st.markdown('''
                        <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Selenium</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">0.01 mg/L</span>
                        </h4>
                    ''', unsafe_allow_html=True)

                with columns[1]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Alumunium</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">0.2 mg/L</span>
                        </h4>
                ''', unsafe_allow_html=True)

                with columns[2]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Besi</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">0.3 mg/L</span>
                        </h4>
                ''', unsafe_allow_html=True)
                
                with columns[3]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Kesadahan</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">500 mg/L</span>
                        </h4>
                ''', unsafe_allow_html=True)
            
            with st.container():
                columns = st.columns(4)
                with columns[0]:
                    st.markdown('''
                        <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Klorida</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">250 mg/L</span>
                        </h4>
                    ''', unsafe_allow_html=True)

                with columns[1]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Mangan</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">0.4 mg/L</span>
                        </h4>
                ''', unsafe_allow_html=True)

                with columns[2]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">pH</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">6.5 - 8.5</span>
                        </h4>
                ''', unsafe_allow_html=True)
                
                with columns[3]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Seng</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">3 mg/L</span>
                        </h4>
                ''', unsafe_allow_html=True)
            
            with st.container():
                columns = st.columns(4)
                with columns[0]:
                    st.markdown('''
                        <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Sulfat</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">250</span>
                        </h4>
                    ''', unsafe_allow_html=True)

                with columns[1]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Tembaga</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">2 mg/L</span>
                        </h4>
                ''', unsafe_allow_html=True)

                with columns[2]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Amonia</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">1.5 mg/L</span>
                        </h4>
                ''', unsafe_allow_html=True)
                
                with columns[3]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Chlor</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">0.2 - 1.0 mg/L</span>
                        </h4>
                ''', unsafe_allow_html=True)
            
            with st.container():
                columns = st.columns(4)
                with columns[0]:
                    st.markdown('''
                        <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">BOD5</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">- mg/L</span>
                        </h4>
                    ''', unsafe_allow_html=True)

                with columns[1]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">COD</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">- mg/L</span>
                        </h4>
                ''', unsafe_allow_html=True)

                with columns[2]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Bau</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">Tidak Berbau : 1</span>
                        </h4>
                ''', unsafe_allow_html=True)
                
                with columns[3]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Warna</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">15 TCU</span>
                        </h4>
                ''', unsafe_allow_html=True)
            
            with st.container():
                columns = st.columns(4)
                with columns[0]:
                    st.markdown('''
                        <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">TDS</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">500 mg/L</span>
                        </h4>
                    ''', unsafe_allow_html=True)

                with columns[1]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Kekeruhan</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">5 NTU</span>
                        </h4>
                ''', unsafe_allow_html=True)

                with columns[2]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Rasa</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">Tidak Berasa : 1</span>
                        </h4>
                ''', unsafe_allow_html=True)
                
                with columns[3]:
                    st.markdown('''
                    <h4 style="font-size: 15px;">
                            <span style="color: black; font-size: 15px; margin-right: 13px;">&#9679</span>
                            <span style="vertical-align: middle;">Suhu</span>
                            <span style="display: block; margin-left: 25px; font-size: 13px; font-weight: normal;">Suhu Udara +- 3 C</span>
                        </h4>
                ''', unsafe_allow_html=True)
                    
    elif option == "Parameter Pengujian Kualitas Air":
        with st.container():
            st.markdown('''
            <div>
                <h1 style="font-size: 30px;">Parameter Pengujian Kualitas Air</h1>
                <p>Mengukur kualitas air secara rutin melalui parameter pengujian yang ditetapkan sangat penting untuk memastikan air yang dikonsumsi aman. Berikut ini penjelasan masing-masing parameter pengujian air.</p>
            </div>
        ''', unsafe_allow_html=True)
            
        with st.container():
            b1, b2, b3, b4 = st.columns(4)

        with b1:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/E COLLI.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">E.Colli</h4>
                <p style="font-size: 14px; text-align: center;">Bakteri yang menunjukkan adanya kontaminasi kotoran manusia atau hewan dalam air</p>
            </div>
            ''', unsafe_allow_html=True)

        with b2:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/COLIFORM.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Coliform</h4>
                <p style="font-size: 14px; text-align: center;">Bakteri yang digunakan sebagai indikator adanya kontaminasi mikroba dalam air</p>
            </div>
            ''', unsafe_allow_html=True)
    
        with b3:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/ARSEN.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Arsen</h4>
                <p style="font-size: 14px; text-align: center;">Senyawa kimia beracun yang terdapat dalam air akibat dari aktivitas manusia dan alam</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with b4:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/KROMIUM.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Kromium</h4>
                <p style="font-size: 14px; text-align: center;">Kromium yaitu sebuah logam berat yang dapat terdapat dalam air sebagai polutan</p>
            </div>
            ''', unsafe_allow_html=True)

        with st.container():
            b1, b2, b3, b4 = st.columns(4)

        with b1:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/KADMIUM.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Kadmium</h4>
                <p style="font-size: 14px; text-align: center;">Logam berat yang ada di lingkungan, namun dapat tercemar melalui aktivitas industri dan pertanian</p>
            </div>
            ''', unsafe_allow_html=True)

        with b2:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/NITRIT.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Nitrit</h4>
                <p style="font-size: 14px; text-align: center;">Hasil dari aktivitas bakteri yang terlibat dalam siklus nitrogen</p>
            </div>
            ''', unsafe_allow_html=True)
    
        with b3:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/NITRAT.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Nitrat</h4>
                <p style="font-size: 14px; text-align: center;">Hasil oksidasi nitrogen yang berasal dari pertanian, limbah organik, dan pemrosesan limbah</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with b4:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/SIANIDA.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Sianida</h4>
                <p style="font-size: 14px; text-align: center;">Senyawa beracun yang dapat berasal dari industri, pertambangan, dan limbah domestik</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with st.container():
            b1, b2, b3, b4 = st.columns(4)

        with b1:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/SELENIUM.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Selenium</h4>
                <p style="font-size: 14px; text-align: center;">Mineral penting yang terdapat secara alami di lingkungan, termasuk dalam air dan tanah</p>
            </div>
            ''', unsafe_allow_html=True)

        with b2:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/ALUMUNIUM.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Aluminium</h4>
                <p style="font-size: 14px; text-align: center;">Keberadaan unsur aluminium (Al) dalam konsentrasi tertentu di dalam air</p>
            </div>
            ''', unsafe_allow_html=True)
    
        with b3:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/BESI.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Besi</h4>
                <p style="font-size: 14px; text-align: center;">Kandungan besi dalam air bisa berpengaruh pada rasa, warna, dan kejernihan air</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with b4:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/KESADAHAN.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Kesadahan</h4>
                <p style="font-size: 14px; text-align: center;">Merujuk pada kandungan mineral, terutama kalsium dan magnesium, yang ada dalam air</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with st.container():
            b1, b2, b3, b4 = st.columns(4)

        with b1:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/KLORIDA.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Klorida</h4>
                <p style="font-size: 14px; text-align: center;">Konsentrasi klorida dalam air dapat mempengaruhi rasa dan keamanan air minum</p>
            </div>
            ''', unsafe_allow_html=True)

        with b2:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/MANGAN.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Mangan</h4>
                <p style="font-size: 14px; text-align: center;">Konsentrasi mangan dalam air dapat mempengaruhi kualitas air minum</p>
            </div>
            ''', unsafe_allow_html=True)
    
        with b3:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/ph.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">pH</h4>
                <p style="font-size: 14px; text-align: center;">pH dalam konteks kandungan air merujuk pada tingkat keasaman atau kebasaan air</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with b4:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/SENG.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Seng</h4>
                <p style="font-size: 14px; text-align: center;">Sng dalam konteks kandungan air adalah salah satu mineral yang bisa ditemukan dalam air</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with st.container():
            b1, b2, b3, b4 = st.columns(4)

        with b1:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/SULFAT.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Sulfat</h4>
                <p style="font-size: 14px; text-align: center;">Sulfat dalam konteks kandungan air adalah senyawa kimia yang terbentuk oleh atom sulfur dan oksigen</p>
            </div>
            ''', unsafe_allow_html=True)

        with b2:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/TEMBAGA.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Tembaga</h4>
                <p style="font-size: 14px; text-align: center;">Tembaga dalam konteks kandungan air adalah logam yang terdapat dalam air</p>
            </div>
            ''', unsafe_allow_html=True)
    
        with b3:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/AMONIA.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Amonia</h4>
                <p style="font-size: 14px; text-align: center;">Amonia dalam kandungan air minum yaitu senyawa kimia yang terbentuk oleh nitrogen dan hidrogen</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with b4:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/CHLOR.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Chlor</h4>
                <p style="font-size: 14px; text-align: center;">Unsur kimia yang digunakan untuk membunuh mikroorganisme dan menghilangkan kuman</p>
            </div>
            ''', unsafe_allow_html=True)
    
        with st.container():
            b1, b2, b3, b4 = st.columns(4)

        with b1:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/bod5.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">BOD5</h4>
                <p style="font-size: 14px; text-align: center;">BOD5 mengacu pada jumlah oksigen yang dibutuhkan oleh mikroorganisme dalam air</p>
            </div>
            ''', unsafe_allow_html=True)

        with b2:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/cod.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">COD</h4>
                <p style="font-size: 14px; text-align: center;">COD mengacu pada jumlah oksigen yang dibutuhkan untuk mengoksidasi bahan kimia organik dan anorganik</p>
            </div>
            ''', unsafe_allow_html=True)
    
        with b3:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/bau (1).png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Bau</h4>
                <p style="font-size: 14px; text-align: center;">Bau dalam air dapat berasal dari bahan organik, zat kimia, atau kontaminan lainnya</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with b4:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/warna.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Warna</h4>
                <p style="font-size: 14px; text-align: center;">Perubahan warna yang signifikan dalam air diindikasi adanya perubahan kualitas air</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with st.container():
            b1, b2, b3, b4 = st.columns(4)

        with b1:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/tds (1).png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">TDS</h4>
                <p style="font-size: 14px; text-align: center;">TDS dalam konteks kandungan air merujuk pada jumlah total zat terlarut yang ada dalam air</p>
            </div>
            ''', unsafe_allow_html=True)

        with b2:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/kekeruhan.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Kekeruhan</h4>
                <p style="font-size: 14px; text-align: center;">Kekeruhan dalam konteks kandungan air mengacu pada sejauh mana air terlihat keruh atau kabur</p>
            </div>
            ''', unsafe_allow_html=True)
    
        with b3:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/rasa.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Rasa</h4>
                <p style="font-size: 14px; text-align: center;">rasa dalam konteks kandungan air mengacu pada sensasi yang dirasakan saat meminum air</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with b4:
            image_div = st.empty()
            st.write("")  # tambahkan baris kosong untuk jarak
            st.image('simbol/suhu.png', width=200, use_column_width=True)

            # membuat div untuk caption
            caption_div = st.empty()
            caption_div.markdown('''
            <div class="caption">
                <h4 style="font-size: 16px; text-align: center;">Suhu</h4>
                <p style="font-size: 14px; text-align: center;">Suhu dalam konteks kandungan air merujuk pada tingkat panas atau dinginnya air</p>
            </div>
            ''', unsafe_allow_html=True)



















































































elif selected == "Informasi":
    st.markdown('''
    <div style="text-align: center;">
        <h1 style="font-size: 25px;">Informasi Terkait Kesehatan</h1>
    </div>
    ''', unsafe_allow_html=True)


    # Membuat CSS
    css = '''
    <style>
    .caption {
    font-size: 16px;
    margin-top: 5px;
    }

    .caption h4 {
    font-size: 20px;
    margin-top: 0;
    }
    </style>
    '''

    # Menampilkan CSS
    st.markdown(css, unsafe_allow_html=True)

    import streamlit as st
    import base64

    with st.container():
        b1, b2 = st.columns(2)

    with b1:
        image_div = st.empty()
        st.write("")  # tambahkan baris kosong untuk jarak
        st.image('simbol/permenkes (2).png', width=200, use_column_width=True)

        # membuat div untuk caption
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <h4>Permenkes</h4>
            <p>Permenkes No. 492 Tahun 2010 ini mengatur tentang standar kualitas air minum yang akan konsumsi manusia.</p>
        </div>
        ''', unsafe_allow_html=True)
        
        with open('Permenkes Persyaratan Kualitas Air Minum.pdf', 'rb') as f:
            pdf_contents = f.read()

        # mengonversi konten PDF ke dalam format Base64
        b64_pdf = base64.b64encode(pdf_contents).decode('utf-8')

        # membuat tombol download PDF dan mengunduh file saat tombol diklik
        st.markdown(
            f'<a href="data:application/pdf;base64,{b64_pdf}" download="Permenkes Persyaratan Kualitas Air.pdf">'
            f'<button style="padding: 10px 140px; font-size: 14px; border: 0.2px solid black; background-color: #454D77; color: white;border-radius: 7px;">Download</button>'
            f'</a>',
            unsafe_allow_html=True
        )

    with b2:
        image_div = st.empty()
        st.write("")  # tambahkan baris kosong untuk jarak
        st.image('simbol/resiko (1).png', width=200, use_column_width=True)

        # membuat div untuk caption
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <h4>Resiko Penyakit</h4>
            <p> Berikut adalah beberapa risiko kesehatan yang dapat terjadi akibat kelebihan atau kekurangan beberapa komponen air.</p>
        </div>
        ''', unsafe_allow_html=True)
        
        with open('Resiko Penyakit.pdf', 'rb') as f:
            pdf_contents = f.read()

        # mengonversi konten PDF ke dalam format Base64
        b64_pdf = base64.b64encode(pdf_contents).decode('utf-8')

        # membuat tombol download PDF dan mengunduh file saat tombol diklik
        st.markdown(
            f'<a href="data:application/pdf;base64,{b64_pdf}" download="Resiko Penyakit.pdf">'
            f'<button style="padding: 10px 140px; font-size: 14px; border: 0.2px solid black; background-color: #454D77; color: white;border-radius: 7px;">Download</button>'
            f'</a>',
            unsafe_allow_html=True
        )

    st.markdown('<hr style="border: 1px solid black; margin-top: 30px;">', unsafe_allow_html=True)

    # INI SOAL ALGORITMA BOOSTING

    st.markdown('''
        <div style="text-align: center;">
            <h1 style="font-size: 25px;">Algoritma Boosting</h1>
        </div>
        ''', unsafe_allow_html=True)


    # Membuat CSS
    css = '''
        <style>
        .caption {
        font-size: 16px;
        margin-top: 5px;
        }

        .caption h4 {
        font-size: 18px;
        margin-top: 0;
        }
        </style>
    '''

    # Menampilkan CSS
    st.markdown(css, unsafe_allow_html=True)

    with st.container():
        b1, b2 = st.columns(2)

    with b1:
        image_div = st.empty()
        st.write("")  # tambahkan baris kosong untuk jarak
        st.image('simbol/boosting (1) (1).png', width=200, use_column_width=True)

        # membuat div untuk caption
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <h4>AdaBoost</h4>
            <p>Tujuan Adaboost adalah untuk meningkatkan akurasi klasifikasi pada model mesin learning.</p>
            <p style="text-align: right;"><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html"><button style="padding: 10px 135px;border: 0.1px solid black;background-color: #454D77; color: white;border-radius: 5px;">Read More</button></a></p>
        </div>
        ''', unsafe_allow_html=True)

    with b2:
        image_div = st.empty()
        st.write("")  # tambahkan baris kosong untuk jarak
        st.image('simbol/smote (2).png', width=200, use_column_width=True)

        # membuat div untuk caption
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <h4>SMOTE</h4>
            <p>Tujuan SMOTE adalah untuk menyeimbangkan kelas pada dataset yang tidak seimbang. </p>
            <p style="text-align: right;"><a href="https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html"><button style="padding: 10px 135px;border: 0.1px solid black;background-color: #454D77; color: white;border-radius: 5px;">Read More</button></a></p>
        </div>
        ''', unsafe_allow_html=True)
    st.markdown('<hr style="border: 1px solid black; margin-top: 30px;">', unsafe_allow_html=True)


    st.markdown('''
            <div style="text-align: center;">
                <h1 style="font-size: 25px;">Algoritma Machine Learning</h1>
            </div>
            ''', unsafe_allow_html=True)
        # Membuat CSS
    css = '''
            <style>
            .caption {
            font-size: 16px;
            margin-top: 5px;
            }

            .caption h4 {
            font-size: 18px;
            margin-top: 0;
            }
            </style>
            '''

        # Menampilkan CSS
    st.markdown(css, unsafe_allow_html=True)
    with st.container():
        c1, c2, c3 = st.columns(3)

    with c1:
        image_div = st.empty()
        st.write("")  # tambahkan baris kosong untuk jarak
        st.image('simbol/DT (1).png', width=150, use_column_width=True)

        # membuat div untuk caption
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <h4>Decision Tree</h4>
            <p>Decision Tree adalah sebuah metode dalam machine learning yang menggunakan struktur pohon untuk membuat keputusan berdasarkan fitur-fitur yang ada.</p>
            <p style="text-align: right;"><a href="https://scikit-learn.org/stable/modules/tree.html"><button style="padding: 3px 8px;border: 0.1px solid black;background-color: #454D77; color: white;border-radius: 5px;">Read More</button></a></p>
        </div>
        ''', unsafe_allow_html=True)

            

    with c2:
        image_div = st.empty()
        st.write("")  # tambahkan baris kosong untuk jarak
        st.image('simbol/knn (2).png', width=150, use_column_width=True)

        # membuat div untuk caption
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <h4>K-Nearest Neighbor</h4>
            <p>KNN adalah metode dalam machine learning yang melakukan prediksi berdasarkan data terdekat dengan menggunakan jarak Euclidean atau metrik lainnya.</p>
            <p style="text-align: right;"><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html"><button style="padding: 3px 8px;border: 0.1px solid black;background-color: #454D77; color: white;border-radius: 5px;">Read More</button></a></p>
        </div>
        ''', unsafe_allow_html=True)
            
    with c3:
        image_div = st.empty()
        st.write("")  # tambahkan baris kosong untuk jarak
        st.image('simbol/elm (2).png', width=150, use_column_width=True)

        # membuat div untuk caption
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <h4>Extreme Learning Machine</h4>
            <p>ELM adalah sebuah metode machine learning yang menggunakan sebuah lapisan tersembunyi dengan bobot acak untuk melakukan prediksi dengan cepat dan efisien.</p>
            <p style="text-align: right;"><a href="https://en.wikipedia.org/wiki/Extreme_learning_machine"><button style="padding: 3px 8px;border: 0.1px solid black;background-color: #454D77; color: white;border-radius: 5px;">Read More</button></a></p>
            <br>
        </div>
        ''', unsafe_allow_html=True) 
    st.markdown('<hr style="border: 1px solid black; margin-top: 30px;">', unsafe_allow_html=True)




    st.markdown('''
    <div style="text-align: center;">
        <h1 style="font-size: 36px;">Dataset</h1>
    </div>
    <div style="text-align: justify;">
        <p>Dataset yang digunakan untuk machine learning mengenai kualitas air minum berasal dari berbagai PDAM dan perusahaan air minum di Indonesia. Dataset ini berisi informasi penting seperti pH, kekeruhan, kandungan zat kimia, dan mikroba dalam air minum. Pengumpulan data dari berbagai sumber ini memungkinkan analisis yang komprehensif dan prediksi yang akurat terkait kualitas air minum. Dengan memanfaatkan teknologi machine learning, diharapkan dapat ditemukan pola dan wawasan yang berharga untuk meningkatkan pengelolaan air minum serta menjaga kualitas air yang aman dan sehat bagi masyarakat Indonesia.</p>
    </div>
    ''', unsafe_allow_html=True)
    st.image("simbol/petaku.png")



    css = '''
    <style>
    .caption {
        font-size: 16px;
        margin-top: 5px;
    }

    .caption h4 {
        font-size: 18px;
        margin-top: 0;
    }

    .slide-container {
        overflow: hidden;
    }

    .slide {
        white-space: nowrap;
        animation: slide 5s linear infinite;
    }

    @keyframes slide {
        0% { transform: translateX(0%); }
        100% { transform: translateX(-100%); }
    }
    </style>
    '''
    # Menampilkan CSS
    st.markdown(css, unsafe_allow_html=True)

    with st.container():
        c1, c2, c3 = st.columns(3)

    with c1:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">1</span>
            <h4>Banda Aceh</h4>
            <p>Dinas Kesehatan</p>
        </div>
        ''', unsafe_allow_html=True)
            

    with c2:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">2</span>
            <h4>Sulawesi Utara</h4>
            <p>Dinas Kesehatan Provinsi</p>
        </div>
        ''', unsafe_allow_html=True)
            
    with c3:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">3</span>
            <h4>Bogor</h4>
            <p>Dinas Kesehatan Kota</p>
        </div>
        ''', unsafe_allow_html=True) 
    
    with st.container():
        c1, c2, c3 = st.columns(3)

    with c1:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">4</span>
            <h4>Bandar Lampung</h4>
            <p>Dinas Kesehatan Provinsi</p>
        </div>
        ''', unsafe_allow_html=True)
            

    with c2:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">5</span>
            <h4>Bekasi</h4>
            <p>Dinas Kesehatan Kota</p>
        </div>
        ''', unsafe_allow_html=True)
            
    with c3:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">6</span>
            <h4>Bekasi</h4>
            <p>Dinas Kesehatan Kabupaten</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with st.container():
        c1, c2, c3 = st.columns(3)

    with c1:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">7</span>
            <h4>Sumatra Barat</h4>
            <p>Balai Laboratorium Kesehatan Provinsi</p>
        </div>
        ''', unsafe_allow_html=True)   

    with c2:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">8</span>
            <h4>Kalimantan Barat</h4>
            <p>Dinas Kesehatan Pemerintah Provinsi</p>
        </div>
        ''', unsafe_allow_html=True)
            
    with c3:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">9</span>
            <h4>Balikpapan</h4>
            <p>Dinas Kesehatan Pemerintah Kota</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with st.container():
        c1, c2, c3 = st.columns(3)

    with c1:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">10</span>
            <h4>Kalimantan Tengah</h4>
            <p>Dinas Kesehatan Kabupaten Kapuas</p>
        </div>
        ''', unsafe_allow_html=True)     

    with c2:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">11</span>
            <h4>Papua</h4>
            <p>Balai Laboratorium Kesehatan Jayapura</p>
        </div>
        ''', unsafe_allow_html=True)
            
    with c3:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">12</span>
            <h4>Kabupaten Malang</h4>
            <p>Institute Teknologi Bandung (ITB)</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with st.container():
        c1, c2, c3 = st.columns(3)

    with c1:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">13</span>
            <h4>Jawa Timur</h4>
            <p>Dinas Kesehatan Pemerintah Kabupaten Blora</p>
        </div>
        ''', unsafe_allow_html=True)
            
    with c2:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">14</span>
            <h4>Jawa Timur</h4>
            <p>Dinas Kesehatan Pemerintah Kabupaten Malang </p>
        </div>
        ''', unsafe_allow_html=True)
            
    with c3:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">15</span>
            <h4>Riau</h4>
            <p>UPT Laboratorium Kesehatan dan Lingkungan</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with st.container():
        c1, c2, c3 = st.columns(3)

    with c1:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">16</span>
            <h4>Bandung</h4>
            <p>Laboratoriu Mikrobiologi Departemen Teknik Lingkungan</p>
        </div>
        ''', unsafe_allow_html=True)
            
    with c2:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">17</span>
            <h4>Kabupaten Sabu</h4>
            <p>Poltek Kemenkes Kupang - Jurusan Kesehatan Lingkungan</p>
        </div>
        ''', unsafe_allow_html=True)
            
    with c3:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">18</span>
            <h4>Kota Bandung</h4>
            <p>UPT Laboratorium Kesehatan Kota Bandung</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with st.container():
        c1, c2, c3 = st.columns(3)

    with c1:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">19</span>
            <h4>Kabupaten Malang</h4>
            <p>Balai Besar Teknik Kesehatan Lingkungan dan Pengendalian Penyakit (BBTKLLP) Surabaya</p>
        </div>
        ''', unsafe_allow_html=True)
            
    with c2:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">20</span>
            <h4>Kabupaten Sidoarjo</h4>
            <p>Balai Besar Teknik Kesehatan Lingkungan dan Pengendalian Penyakit (BBTKLLP) Surabaya</p>
        </div>
        ''', unsafe_allow_html=True)
            
    with c3:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">21</span>
            <h4>Institute Pertanian Bogor</h4>
            <p>Unit Laboratorium Jasa Pengujian, Kalibrasi dan Sertifikasi</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with st.container():
        c1, c2, c3 = st.columns(3)

    with c1:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">22</span>
            <h4>PT. Coway Internasional Indonesia</h4>
            <p>Institute Teknologi Bandung</p>
        </div>
        ''', unsafe_allow_html=True)
            
    with c2:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">23</span>
            <h4>PDAM Surya Sembada</h4>
            <p>Jl. Mayjen Prof. Dr. Moestopo No.2</p>
        </div>
        ''', unsafe_allow_html=True)
            
    with c3:
        caption_div = st.empty()
        caption_div.markdown('''
        <div class="caption">
            <span style="color: black; font-size: 15px; margin-right: 13px;">24</span>
            <h4>PDAM Tirtawening</h4>
            <p>Jl. Badak Singa No.10 Bandung</p>
        </div>
        ''', unsafe_allow_html=True)


    

    




























if selected == "Credits":
    # Judul
    st.markdown('''
    <div style="text-align: center;">
        <h1 style="font-size: 36px;">About Us.</h1>
    </div>
    <div style="text-align: justify;">
        <p>Website kualitas air minum bertujuan untuk memberikan informasi mengenai kualitas air minum di suatu wilayah atau tempat, sehingga masyarakat dapat mengetahui apakah air minum yang mereka konsumsi sudah memenuhi standar kesehatan atau tidak. Hal ini penting dilakukan untuk menjaga kesehatan dan mencegah terjadinya penyakit yang disebabkan oleh air minum yang tidak layak konsumsi.</p>
        <p>Namun tidak hanya itu, di dalam website ini juga terdapat informasi terkait Peraturan Menteri Kesehatan (Permenkes), resiko penyakit yang dapat ditimbulkan akibat konsumsi air yang tidak layak, serta informasi tentang algoritma machine learning yang digunakan untuk mengolah data pengujian air.</p>
    </div>
    <div>
    <h1 style="font-size: 36px;">Team</h1>
    <hr style="width: 100%; border-style: solid; border-width: 2px; margin-top: 5px; margin-bottom: 10px;">
    </div>
    ''', unsafe_allow_html=True)

    # Container untuk menampilkan gambar anggota tim secara berderet 3 ke samping
    with st.container():
        col1, col2, col3 = st.columns(3)

    # Anggota Tim 1
    with col1:
        image_div = st.empty()
        st.write("")  # tambahkan baris kosong untuk jarak
        
        # membuat div untuk caption
        caption_div = st.empty()
        caption_div.markdown('**Ivana Meiska**<br><p style="margin-top:-10px;">S1 Teknik Komputer</p><hr style="width: 100%; margin: 3px 0;">', unsafe_allow_html=True)
        st.image('simbol/ivana.png', width=200, use_column_width=True)


    with col2:
        image_div = st.empty()
        st.write("")  # tambahkan baris kosong untuk jarak
        
        # membuat div untuk caption
        caption_div = st.empty()
        caption_div.markdown('**Brilliant Friezka Aina**<br><p style="margin-top:-10px;">S1 Teknik Komputer</p><hr style="width: 100%; margin: 3px 0;">', unsafe_allow_html=True)
        st.image('simbol/firezka.png', width=200, use_column_width=True)

    with col3:
        image_div = st.empty()
        st.write("")  # tambahkan baris kosong untuk jarak
        
        # membuat div untuk caption
        caption_div = st.empty()
        caption_div.markdown('**Syifa Melinda Nafan**<br><p style="margin-top:-10px;">S1 Teknik Komputer</p><hr style="width: 100%; margin: 3px 0;">', unsafe_allow_html=True)
        st.image('simbol/syifa.png', width=200, use_column_width=True)
    
    
    st.markdown('''
    <div>
    <h1 style="font-size: 36px;">Pembimbing</h1>
    <hr style="width: 100%; border-style: solid; border-width: 2px; margin-top: 5px; margin-bottom: 10px;">
    </div>
    ''', unsafe_allow_html=True)

    with st.container():
        a1, a2 = st.columns(2)

    with a1:
        image_div = st.empty()
        st.write("")  # tambahkan baris kosong untuk jarak
        
        # membuat div untuk caption
        caption_div = st.empty()
        caption_div.markdown('**Dr. Meta Kalista, S.Si., M.Si.**<br><p style="margin-top:-10px;">Pembimbing 1</p>', unsafe_allow_html=True)
         

    with a2:
        image_div = st.empty()
        st.write("")  # tambahkan baris kosong untuk jarak
        
        # membuat div untuk caption
        caption_div = st.empty()
        caption_div.markdown('**IG. Prasetya Dwi Wibawa, S.T., M.T.**<br><p style="margin-top:-10px;">Pembimbing 2</p>', unsafe_allow_html=True)


        
        

        

    
