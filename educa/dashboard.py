import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Carregar os dados
file_path = 'censo-ma.csv'
try:
    df = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, delimiter=';', encoding='latin1')

# Filtrar as escolas de tempo integral
df_integral = df[df['INTEGRAL'] == 1]

# Verificar o número total de escolas de tempo integral
total_escolas_integral = df_integral.shape[0]

# Adicionar menu de navegação
menu = st.sidebar.selectbox("Selecione a Página", ["Dashboard", "Algoritmo de IA"])

if menu == "Dashboard":
    # Título para a aplicação
    st.title('Dashboard de Necessidades de Infraestrutura das Escolas de Tempo Integral do Maranhão')

    # Mostrar o número total de escolas de tempo integral
    st.subheader('Total de Escolas de Tempo Integral: 212')

    # Lista de colunas de infraestrutura
    infraestrutura_colunas = {
        'IN_AGUA_POTAVEL': 'Água Potável',
        'IN_ENERGIA_REDE_PUBLICA': 'Energia Elétrica',
        'IN_ESGOTO_REDE_PUBLICA': 'Esgoto',
        'IN_COMPUTADOR': 'Computador',
        'IN_QUADRA_ESPORTES': 'Quadra de Esportes',
        'IN_LABORATORIO_CIENCIAS': 'Laboratório de Ciências',
        'IN_LABORATORIO_INFORMATICA': 'Laboratório de Informática',
        'IN_BIBLIOTECA': 'Biblioteca',
        'IN_REFEITORIO': 'Refeitório',
        'IN_COZINHA': 'Cozinha',
        'IN_AUDITORIO': 'Auditório',
        'IN_PATIO_COBERTO': 'Pátio Coberto'
    }

    # Gerar gráficos de barras para cada tipo de infraestrutura
    for coluna, descricao in infraestrutura_colunas.items():
        df_counts = df_integral[coluna].value_counts().reset_index()
        df_counts.columns = [coluna, 'count']
        fig = px.bar(df_counts, x=coluna, y='count', title=f'Disponibilidade de {descricao}', labels={coluna: descricao, 'count': 'Número de Escolas'}, text='count')
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig)
        st.write("**Legenda:** 0 = Não, 1 = Sim")

    # Gráfico de pizza geral para a distribuição de escolas de tempo integral com base em um aspecto de infraestrutura
    # Selecionando uma infraestrutura para o gráfico de pizza (exemplo: Água Potável)
    coluna_pizza = 'IN_AGUA_POTAVEL'
    descricao_pizza = 'Água Potável'

    df_pizza = df_integral[coluna_pizza].value_counts().reset_index()
    df_pizza.columns = [coluna_pizza, 'count']
    fig_pizza = px.pie(df_pizza, values='count', names=coluna_pizza, title=f'Distribuição de Escolas com {descricao_pizza}', labels={coluna_pizza: descricao_pizza, 'count': 'Número de Escolas'})
    st.plotly_chart(fig_pizza)
    st.write("**Legenda:** 0 = Não, 1 = Sim")

    # Listar escolas de tempo integral com maior necessidade de infraestrutura
    st.header('Escolas de Tempo Integral com Maior Necessidade de Infraestrutura')

    # Selecionar escolas que não possuem as infraestruturas básicas
    escolas_necessidade = df_integral[
        (df_integral['IN_AGUA_POTAVEL'] == 0) |
        (df_integral['IN_ENERGIA_REDE_PUBLICA'] == 0) |
        (df_integral['IN_ESGOTO_REDE_PUBLICA'] == 0) |
        (df_integral['IN_COMPUTADOR'] == 0) |
        (df_integral['IN_QUADRA_ESPORTES'] == 0) |
        (df_integral['IN_LABORATORIO_CIENCIAS'] == 0) |
        (df_integral['IN_LABORATORIO_INFORMATICA'] == 0) |
        (df_integral['IN_BIBLIOTECA'] == 0) |
        (df_integral['IN_REFEITORIO'] == 0) |
        (df_integral['IN_COZINHA'] == 0) |
        (df_integral['IN_AUDITORIO'] == 0) |
        (df_integral['IN_PATIO_COBERTO'] == 0)
    ]

    # Mostrar tabela com as escolas e suas necessidades
    st.write(escolas_necessidade[['NO_ENTIDADE', 'NO_MUNICIPIO', 'IN_AGUA_POTAVEL', 'IN_ENERGIA_REDE_PUBLICA', 'IN_ESGOTO_REDE_PUBLICA', 'IN_COMPUTADOR', 'IN_QUADRA_ESPORTES', 'IN_LABORATORIO_CIENCIAS', 'IN_LABORATORIO_INFORMATICA', 'IN_BIBLIOTECA', 'IN_REFEITORIO', 'IN_COZINHA', 'IN_AUDITORIO', 'IN_PATIO_COBERTO']])
    st.write("**Legenda:** 0 = Não, 1 = Sim")

elif menu == "Algoritmo de IA":
    st.title("Resultado com Análise de Algoritmo de Inteligência Artificial")

    # Lista de colunas de infraestrutura
    infraestrutura_colunas = [
        'IN_AGUA_POTAVEL', 'IN_ENERGIA_REDE_PUBLICA', 'IN_ESGOTO_REDE_PUBLICA', 
        'IN_COMPUTADOR', 'IN_QUADRA_ESPORTES', 'IN_LABORATORIO_CIENCIAS', 
        'IN_LABORATORIO_INFORMATICA', 'IN_BIBLIOTECA', 'IN_REFEITORIO', 
        'IN_COZINHA', 'IN_AUDITORIO', 'IN_PATIO_COBERTO'
    ]

    # Criar coluna target: 1 se a escola necessita de infraestrutura, 0 caso contrário
    df_integral['NECESSIDADE'] = df_integral[infraestrutura_colunas].apply(lambda row: 1 if row.sum() < len(infraestrutura_colunas) else 0, axis=1)

    # Separar features e target
    X = df_integral[infraestrutura_colunas]
    y = df_integral['NECESSIDADE']

    # Dividir em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinar o modelo
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Prever no conjunto de teste
    y_pred = clf.predict(X_test)

    # Mostrar relatório de classificação
    report = classification_report(y_test, y_pred, target_names=['Sem Necessidade', 'Com Necessidade'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.rename(columns={
        'precision': 'Precisão',
        'recall': 'Revocação',
        'f1-score': 'F1-Score',
        'support': 'Suporte'
    })
    report_df = report_df.rename(index={
        'accuracy': 'Acurácia',
        'macro avg': 'Média Macro',
        'weighted avg': 'Média Ponderada'
    })
    st.text("Relatório de Classificação:")
    st.write(report_df)

    # Importância das variáveis
    feature_importances = pd.DataFrame(clf.feature_importances_, index=X.columns, columns=['Importância']).sort_values('Importância', ascending=False)
    fig_importance = px.bar(feature_importances, x=feature_importances.index, y='Importância', title='Importância das Variáveis', text='Importância')
    fig_importance.update_traces(textposition='outside')
    st.plotly_chart(fig_importance)

    # Descrição do gráfico
    st.write("### Descrição do Gráfico de Importância das Variáveis")
    st.write("O gráfico acima mostra a importância relativa de cada característica na previsão da necessidade de infraestrutura. "
             "Quanto maior a importância, maior o impacto dessa característica na determinação da necessidade de infraestrutura "
             "das escolas de tempo integral.")

    # Classificar e ranquear as escolas
    df_integral['PROB_NECESSIDADE'] = clf.predict_proba(X)[:, 1]
    escolas_ranqueadas = df_integral.sort_values('PROB_NECESSIDADE', ascending=False)
    st.header('Escolas de Tempo Integral com Maior Necessidade de Infraestrutura (Ranqueadas)')
    st.write(escolas_ranqueadas[['NO_ENTIDADE', 'NO_MUNICIPIO', 'PROB_NECESSIDADE']])
