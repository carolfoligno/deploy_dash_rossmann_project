import math
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st
from PIL import Image
import inflection
import plotly.express as px
import plotly.graph_objects as go
import re


@st.cache_data()
def get_data():

    # low_memory = False (ler todo o arquivo na mesma hora coloca na memoria)
    df_raw = pd.read_csv ('data/data_clean.csv')
    
    return df_raw

def millify(n):
    n = float(n)
    
    millnames = ['',' K',' M',' B',' T']
    
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return str(round(n / 10**(3 * millidx),2)) + str(millnames[millidx])

def filtro_year(data):
    anos = list(data['year'].unique())

    with st.sidebar:
        filtro = st.multiselect('SELECIONA O ANO', anos, default= anos)

    return filtro

def metric(data, filtro):

    df = data[['sales', 'customers', 'store', 'year']]
    # números de vendas
    if (filtro != []):
        df = df.loc[df['year'].isin(filtro), :] 

    x1 = millify(df['sales'].sum())
    x2 = millify(df['customers'].sum())
    x3 = millify(df['store'].sum())

    col1, col2, col3 = st.columns(3)
    col1.metric('Nº VENDAS', x1, f'{filtro}')
    col2.metric('Nº CLIENTES', x2, f'{filtro}')
    col3.metric('Nº LOJAS', x3, f'{filtro}')

    return None

def sales_time(data, filtro):
    
    
    # total de vendas ao longo do tempo
    sns.set_theme(style="whitegrid")

    aux_plot = data[['week_of_year','year' ,'sales']].groupby(['week_of_year', 'year']).mean().reset_index().sort_values(by= ['year','week_of_year'])

    aux_plot['week_of_year'] = aux_plot['week_of_year'].astype(str)

    # números de vendas
    if (filtro != []):
        aux_plot = aux_plot.loc[aux_plot['year'].isin(filtro), :] 

    fig = plt.figure(figsize=(18,5))
    ax = sns.lineplot(x="week_of_year", y="sales",
                 hue="year",
                 data=aux_plot);
    ax.set( xlabel = 'Semanas' ,ylabel='Média de vendas')

    st.pyplot(fig)
    
    return None

def sales_month(data, filtro):

    df = data[['year', 'month', 'sales']]

    legenda = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun', 7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez' }

    # filtro por ano
    if (filtro != []):
        df = df.loc[df['year'].isin(filtro), :]  

    # df_plot = df[['month','sales']].groupby('month').sum().reset_index().sort_values(by='sales', ascending=False) 
    df_plot = df[['month','sales']].groupby('month').sum().reset_index().sort_values(by='month', ascending=True) 

    df_plot['month'] = df_plot['month'].map(legenda)


    ax = sns.set_style("whitegrid")
    f, ax = plt.subplots()
    f = plt.figure(figsize=(19,3))
    ax = sns.lineplot(data=df_plot, x= 'month', y='sales', marker='o')

    for v in df_plot.iterrows():
        ax.text(v[1][0] , v[1][1] + 19990000, f'{round(v[1][1]/1000000, 2)} M', horizontalalignment='left', size='medium', color='black', weight='semibold')
    ax.ticklabel_format(style='plain', axis="y")

    # ax.bar_label(ax.containers[0], labels= df_plot['sales'].apply(lambda x: '{:,.2f}'.format(x/1000000) + 'M'))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x/1000000) + 'M'))

    ax.set( xlabel = f'Meses - {filtro}',ylabel='Total de vendas por mês')

    # ax.bar_label(ax.containers[0], fmt='%d')

    st.pyplot(f)
    
    return None

def holiday(data):
    
    # média de vendas nos feriados
    aux = data[(data['state_holiday'] != 'regular_day') & (data['year'] != 2015)]
    aux = aux.rename(columns= {'state_holiday': 'Feriados'})
    feriados = {'christmas': ' Natal', 'easter_holiday': 'Páscoa', 'public_holiday': 'Feriado Público'}
    aux['Feriados'] = aux['Feriados'].map(feriados)

    aux1 = aux[['Feriados','sales']].groupby('Feriados').mean().reset_index().sort_values(by='sales', ascending=True)
    aux2 = aux[['year', 'Feriados', 'sales']].groupby(['year', 'Feriados']).mean().reset_index().sort_values(by='sales', ascending=True)

    col1, col2= st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        # fig = plt.figure(figsize=(6,7))
        ax = sns.barplot(x='Feriados', y='sales', data=aux1, palette= 'flare')
        ax.ticklabel_format(style='plain', axis="y")
        ax.bar_label(ax.containers[0], fmt='%d', padding=4, color='black', weight='semibold')
        ax.set( xlabel = 'Feriados',ylabel='Média de vendas')

        st.pyplot(fig)

    with col2:
        fig1, ax1 = plt.subplots()
        # fig1 = plt.figure(figsize=(6,7))
        ax1 = sns.barplot(x='year', y='sales', hue='Feriados', data=aux2, palette= 'flare')
        ax1.ticklabel_format(style='plain', axis="y")
        ax1.set( xlabel = 'Ano',ylabel='Média de vendas')

        st.pyplot(fig1)

    return None

def promo(data, filtro):

    # análise de promoções
    #Total de vendas em lojas com promoções - filtrar por ano
    df = data[['sales', 'is_promo', 'year']]

    # filtro por ano
    if (filtro != []):
        df = df.loc[df['year'].isin(filtro), :]

    df1 = df.loc[df['is_promo'] != 0, ['sales', 'year']]
    df2 = df.loc[df['is_promo'] == 0, ['sales', 'year']]

    x2 = millify(df1['sales'].mean())
    x1 = millify(df1['sales'].sum())
    x3 = millify(df2['sales'].mean())

    col1, col2, col3 = st.columns(3)
    col1.metric('TOTAL DE VENDAS', x1, f'PROMO - {filtro}')
    col2.metric('MÉDIA DE VENDAS', x2, f'PROMO - {filtro}')
    col3.metric('MÉDIA DE VENDAS', x3, f'SEM PROMO - {filtro}')
        
    return None

def store_promo(data, filtro):

    sns.set_theme(style="whitegrid")

    df = data[['sales', 'year', 'is_promo', 'week_of_year']]

    # filtro por ano
    if (filtro != []):
        df = df.loc[df['year'].isin(filtro), :] 
    
    #Gráfico de média de vendas de lojas com promoção e sem promoção
    df_plot = df[['is_promo','sales', 'week_of_year']].groupby(['week_of_year' ,'is_promo']).mean().reset_index().sort_values(by= 'week_of_year')
    legenda = {0: 'Sem Promoção', 1: 'Com Promoção'}
    df_plot['is_promo'] = df_plot['is_promo'].map(legenda)

    df_plot['week_of_year'] = df_plot['week_of_year'].astype(str)

    fig = plt.figure(figsize=(18,5))
    ax = sns.lineplot(x="week_of_year", y="sales",
                 hue="is_promo",
                 data=df_plot);
    ax.set( xlabel = 'Semanas' ,ylabel='Média de vendas')

    st.pyplot(fig)

    return None

def assortment(data, filtro):

    df = data[['assortment', 'store', 'sales', 'year']]

    legenda = {'basic': 'Básico', 'extended': 'Extendido', 'extra': 'Extra'}
    df['assortment'] = df['assortment'].map(legenda)

     # filtro por ano
    if (filtro != []):
        df = df.loc[df['year'].isin(filtro), :] 

    col1, col2= st.columns(2)
    with col1:
        df_plot = df[['assortment', 'store']].groupby('assortment').count().reset_index() 

        # grafico de donout do assortment

        labels = df_plot['assortment']
        size = df_plot['store']
        colors = ['indigo', 'steelblue', 'lightseagreen']

        fig, ax = plt.subplots()
        ax.pie(size, labels=labels, autopct= '%1.1f%%', colors=colors, textprops={'fontsize': 12}, labeldistance=1.1, pctdistance=0.5)

        circle = plt.Circle((0,0), 0.7, color='white')
        p=plt.gcf()
        p.gca().add_artist(circle)


        # ax.title('Tipos de sortimentos')
        ax.axis('equal')
        
        st.pyplot(fig)

    with col2:

        df1 = df[['sales', 'assortment', 'year']]
        # números de vendas
        if (filtro != []):
            df1 = df1.loc[df1['year'].isin(filtro), :] 

        x1 = millify(df1.loc[df1['assortment'] == 'Extra','sales'].mean())
        x2 = millify(df1.loc[df1['assortment'] == 'Extendido','sales'].mean())
        x3 = millify(df1.loc[df1['assortment'] == 'Básico','sales'].mean())

        st.metric('Média de Vendas', x1, 'Extra')
        st.metric('Média de Vendas', x2, 'Extendido')
        st.metric('Média de Vendas', x3, 'Básico')
    
    return None




if __name__ == '__main__':
    st.set_page_config(
        page_title='Dashboard',
        layout='wide'
    )
    data = get_data()
    filtro = filtro_year(data)

    st.sidebar.title("Projeto")
    st.sidebar.info(
        "Este protejo é uma Análise Descritiva dos dados das Lojas Rossmann. "
        "Dados disponivel no [Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales/overview). "

    )
    st.sidebar.title("Sobre")
    st.sidebar.info(
        """
        Com o propósito de uma Análise Descritiva dos dados das Lojas Rossmann, 
        foi produzido um dashboard através da linguagem Python
        no Framework Streamlit.
""")

    st.image(Image.open('rossmann.png'))


    metric(data, filtro)
    
    st.markdown("<h2 style='text-align: center; color: grey;'>Análise das Vendas ao longo do tempo</h2>", unsafe_allow_html = True)
    sales_time(data, filtro)
    # sales_year(data)
    st.markdown("<h2 style='text-align: center; color: grey;'>Média de vendas por mês</h2>", unsafe_allow_html = True)
    sales_month(data, filtro)
    st.markdown("<h2 style='text-align: center; color: grey;'>Média de vendas por feriados</h2>", unsafe_allow_html = True)
    holiday(data)
    st.markdown("<h2 style='text-align: center; color: grey;'>Análise de Vendas durante Promoção</h2>", unsafe_allow_html = True)
    promo(data, filtro)
    store_promo(data, filtro)
    st.markdown("<h2 style='text-align: center; color: grey;'>Análise de Vendas por sortimentos</h2>", unsafe_allow_html = True)
    assortment(data, filtro)
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('Dataset usado para essa Análise Exploratória de Dados: ')
    # check button
    x = st.checkbox('Exemplo do Dataset')
    if x:
        st.dataframe(data.head(), width=2000)