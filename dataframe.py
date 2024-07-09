import pandas as pd
import streamlit as st

# Carregar os dados
file_path = 'data/censo-ma.csv'
try:
    df = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, delimiter=';', encoding='latin1')

# Filtrar as escolas de tempo integral
df_integral = df[df['INTEGRAL'] == 1]

# Listar todas as colunas disponíveis
st.write("### Colunas disponíveis no DataFrame")
st.write(df_integral.columns.tolist())
