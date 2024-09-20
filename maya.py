# Aplikasi Streamlit untuk Analisis Data Perbaikan Kondisi Mental Pasien Terapi Komplementer RS Nur Hidayah Bantul
# Import library dari Python

import streamlit as st
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
from itertools import combinations
from lifelines.statistics import proportional_hazard_test
from lifelines.statistics import logrank_test, multivariate_logrank_test


class Aplikasi:

    def __init__(self):
        self.data = pd.DataFrame()
        self.data_attributes = list()
        self.data_form = dict()
        self.submitted_key_column = False
        self.submitted_form_data = False
        self.file_is_uploaded = False
        self.kmf = KaplanMeierFitter()
        self.cph = CoxPHFitter()
        self.data_key_columns = {
            "duration": None,
            "event_observed": None,
            "category": None
        }

    def init_app(self):
        # st.title("ANALISIS PERBAIKAN KONDISI MENTAL PASIEN TERAPI KOMPLEMENTER DI RUMAH SAKIT NUR HIDAYAH BANTUL YOGYAKARTA MENGGUNAKAN REGRESI SURVIVAL")
        st.markdown("#\n## Analisis Perbaikan Kondisi Mental Pasien Terapi Komplementer di RS Nur Hidayah Bantul Dengan Regresi Survival")
        # st.markdown("<h1 style='text-align: center; color: blue;'>Analisis Perbaikan Kondisi Mental Pasien Terapi Komplementer Di RS Nur Hidayah Bantul Menggunakan Regresi Survival</h1>", unsafe_allow_html=True)
        st.write("Studi ini mengeksplorasi efektivitas terapi komplementer dalam meningkatkan kesehatan mental di RS Nur Hidayah.")
        st.sidebar.title("Input Your Data")

    # Menambahkan background pada website
    #def add_background(self):
        #st.markdown(
           # f"""
           # <style>
           # .stApp {{
             #   background: url("https://www.example.com/path/to/your/background.jpg");
             #   background-size: cover;
           # }}
            #</style>
           # """,
           # unsafe_allow_html=True
       # )

    def get_data_excel(self):
        uploaded_file = st.sidebar.file_uploader("Choose a file", type=["xlsx", "csv"])
        if uploaded_file is not None and uploaded_file.name.endswith(".xlsx"):
            self.data = pd.read_excel(uploaded_file)
        elif uploaded_file is not None and uploaded_file.name.endswith(".csv"):
            self.data = pd.read_csv(uploaded_file)

        if not self.data.empty:
            self.data = self.data.dropna()
            single_unique_cols = self.data.columns[self.data.nunique() == 1]
            self.data = self.data.drop(single_unique_cols, axis=1)

            self.data_attributes = self.data.columns
            self.file_is_uploaded = True

            self.show_data_dataframe()
            self.get_data_key_column()
            self.show_form()

    def show_data_dataframe(self):
        if self.file_is_uploaded:
            col1, col2 = st.columns(2)
            col1.text("Data :")
            col1.dataframe(self.data)

            col2.text("Data Attributes :")
            col2.dataframe(self.data_attributes)

    def get_data_key_column(self):
        if self.file_is_uploaded:
            with st.sidebar.form("data_key_column"):
                self.data_key_columns["duration"] = st.selectbox(
                    "Duration", self.data_attributes)
                self.data_key_columns["event_observed"] = st.selectbox(
                    "Event Observed", self.data_attributes)
                self.data_key_columns["category"] = st.selectbox(
                    "Category (Optional)", ["None"] + list(self.data_attributes))
                self.submitted_key_column = st.form_submit_button("Submit")

    def show_form(self):
        if self.file_is_uploaded:
            with st.sidebar.form("data_form"):
                for attr in self.data_attributes:
                    if self.data[attr].dtype == 'int64':
                        self.data_form[attr] = st.number_input(
                            attr, value=0, step=1)
                    elif self.data[attr].dtype == 'float64':
                        self.data_form[attr] = st.number_input(
                            attr, value=0, step=0.01)
                    else:
                        self.data_form[attr] = st.selectbox(
                            attr, self.data[attr].unique())
# perlu nambah fitur, di antara banyak kolom, mana yg kategorik? trus yg kepilih harus di faktorisasi ()
                self.submitted_form_data = st.form_submit_button("Submit")

                if self.submitted_form_data:
                    self.add_data_input()
                    self.show_data_input()
                    self.fit_cox_ph()
                    self.predict_survival()

    def add_data_input(self):
        self.data = pd.concat([self.data, pd.DataFrame(self.data_form, index=[0])], ignore_index=True)

    def show_data_input(self):
        st.title("Data Input Value :")
        for key, value in self.data_form.items():
            st.write(key, ":", value)

    def clean_data(self):
        # Konversi kolom durasi ke tipe numerik
        self.data[self.data_key_columns["duration"]] = pd.to_numeric(
            self.data[self.data_key_columns["duration"]], errors='coerce')
        # Konversi kolom event yang diamati ke tipe integer
        self.data[self.data_key_columns["event_observed"]] = pd.to_numeric(
            self.data[self.data_key_columns["event_observed"]], errors='coerce').astype(int)
        # Menghapus baris yang memiliki nilai NaN setelah konversi
        self.data = self.data.dropna(subset=[self.data_key_columns["duration"], self.data_key_columns["event_observed"]])

    def plot_kaplan_meier(self):
        if self.file_is_uploaded and self.submitted_key_column:
            self.clean_data()  # Memanggil fungsi untuk membersihkan data
            st.text("Kurva Kaplan-Meier")
            duration_col = self.data_key_columns["duration"]
            event_col = self.data_key_columns["event_observed"]
            category_col = self.data_key_columns.get("category", None)

            if category_col and category_col != "None":
                unique_categories = self.data[category_col].unique()
                plt.figure(figsize=(10, 6))
                for category in unique_categories:
                    category_data = self.data[self.data[category_col] == category]
                    self.kmf.fit(durations=category_data[duration_col],
                                 event_observed=category_data[event_col],
                                 label=str(category))
                    self.kmf.plot_survival_function()

                plt.title("Kurva Kaplan-Meier berdasarkan " + category_col)
                plt.xlabel("Waktu (Bulan)")
                plt.ylabel("Peluang Perbaikan Kondisi Mental")
                st.pyplot(plt)

                st.markdown("**Interpretasi Kurva Kaplan-Meier:**")
                st.markdown(
                    """
                    Kurva Kaplan-Meier menunjukkan peluang pasien terapi komplementer mengalami perbaikan kondisi mental. 
                    Garis lebih tinggi menunjukkan peluang mengalami perbaikan kondisi mental yang lebih rendah. 
                    Perbandingan antara kategori menunjukkan perbedaan dalam peluang perbaikan kondisi mental.
                    """
                )

                # Uji Log-Rank
                if len(unique_categories) == 2:
                    group1 = self.data[self.data[category_col] == unique_categories[0]]
                    group2 = self.data[self.data[category_col] == unique_categories[1]]
                    results = logrank_test(group1[duration_col], group2[duration_col],
                                           event_observed_A=group1[event_col],
                                           event_observed_B=group2[event_col])
                    st.text(f"Log-Rank Test p-value: {results.p_value:.4f}")
                    st.markdown("**Interpretasi Uji Log-Rank:**")
                    st.markdown(
                        """
                        Uji Log-Rank digunakan untuk menguji perbedaan pada kurva survival antara 2 kelompok atau lebih secara statistik. 
                        Jika nilai p-value kurang dari 0.05, maka ada perbedaan peluang membaik secara signifikan antara dua kelompok atau lebih. 
                        """
                    )

                else:
                    # Uji Log-Rank Multivariat untuk kategori lebih dari 2
                    results = multivariate_logrank_test(self.data[duration_col], self.data[category_col], self.data[event_col])
                    st.text(f"Multivariate Log-Rank Test p-value: {results.p_value:.4f}")
                    st.markdown("**Interpretasi Uji Log-Rank Multivariat:**")
                    st.markdown(
                        """
                        Uji Log-Rank Multivariat membandingkan lebih dari dua kurva survival Kaplan-Meier.
                        Jika nilai p-value kurang dari 0.05, maka ada perbedaan peluang membaik secara signifikan antara tiga kelompok atau lebih.
                        """
                    )
            else:
                self.kmf.fit(durations=self.data[duration_col],
                             event_observed=self.data[event_col],
                             label="All Attributes")
                plt.figure(figsize=(10, 6))
                self.kmf.plot_survival_function()
                plt.title("Kurva Kaplan-Meier")
                plt.xlabel("Waktu (Bulan)")
                plt.ylabel("Peluang Perbaikan Kondisi Mental")
                st.pyplot(plt)
                
                st.markdown("**Interpretasi Kurva Kaplan-Meier:**")
                st.markdown(
                    """
                    Kurva Kaplan-Meier menunjukkan peluang perbaikan kondisi mental dari waktu ke waktu.
                    Garis lebih tinggi menunjukkan peluang mengalami perbaikan kondisi mental yang lebih rendah. 
                    """
                )

    def fit_cox_ph(self):
        if self.file_is_uploaded and self.submitted_key_column:
            self.clean_data()  # Memanggil fungsi untuk membersihkan data
            duration_col = self.data_key_columns["duration"]
            event_col = self.data_key_columns["event_observed"]
            category_col = self.data_key_columns.get("category", None)

            if category_col and category_col != "None":
                cph_data = self.data[[duration_col, event_col, category_col]]
            else:
                cph_data = self.data[[duration_col, event_col] + [col for col in self.data.columns if col not in [duration_col, event_col]]]

            self.cph.fit(cph_data, duration_col=duration_col, event_col=event_col)
            st.subheader('Cox Proportional Hazards Model')
            st.write(self.cph.summary)
            st.markdown("**Interpretasi Model Cox Proportional Hazards:**")
            st.markdown(
                """
                Model Cox PH memberikan estimasi pengaruh dari setiap variabel terhadap peluang perbaikan kondisi mental.
                Koefisien positif menunjukkan peningkatan peluang perbaikan, sementara koefisien negatif menunjukkan penurunan peluang perbaikan.
                Nilai p-value yang rendah menunjukkan pengaruh yang signifikan terhadap waktu perbaikan kondisi mental para pasien secara statistik.
                """
            )

    def predict_survival(self):
        if self.file_is_uploaded and self.submitted_key_column:
            self.clean_data()  # Memanggil fungsi untuk membersihkan data
            input_data = pd.DataFrame([self.data_form])

            # Memastikan input_data memiliki kolom yang sama dengan data yang digunakan untuk fit_cox_ph
            for col in self.data.columns:
                if col not in input_data.columns:
                    input_data[col] = [0]  # Menambahkan kolom yang hilang dengan nilai default

            # Menghitung prediksi peluang perbaikan kondisi mental
            survival_probability = self.cph.predict_survival_function(input_data).values[0, -1]
            st.subheader('Prediksi Peluang Perbaikan Kondisi Mental')
            st.write(f'Peluang Perbaikan Kondisi Mental: {survival_probability:.4f}')
            st.markdown("**Interpretasi Peluang Perbaikan Kondisi Mental:**")
            st.markdown(
                """
                Peluang perbaikan kondisi mental menunjukkan kemungkinan individu untuk mengalami perbaikan kondisi mental hingga waktu tertentu.
                Nilai yang lebih tinggi menunjukkan kemungkinan perbaikan kondisi mental yang lebih tinggi.
                Probabilitas ini membantu dalam memahami efek dari variabel-variabel pada peluang perbaikan kondisi mental individu.
                """
            )

# Membuat instance dari kelas Aplikasi
aplikasi = Aplikasi()

# Memanggil metode init_app untuk menginisialisasi antarmuka
aplikasi.init_app()

# Memanggil metode get_data_excel untuk mengunggah dan memuat data
aplikasi.get_data_excel()

# Memproses dan menampilkan data jika file telah diunggah dan kolom kunci telah dipilih
if aplikasi.file_is_uploaded and aplikasi.submitted_key_column:
    # Menampilkan kurva Kaplan-Meier
    aplikasi.plot_kaplan_meier()

    # Menampilkan hasil Cox Proportional Hazards Model
    aplikasi.fit_cox_ph()

    # Menampilkan prediksi probabilitas perbaikan kondisi mental
    aplikasi.predict_survival()
else:
    st.write("Silakan unggah file Excel atau CSV untuk melanjutkan.")

