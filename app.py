# ==============================================================================
# SEÇÃO 1: IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

print("Bibliotecas importadas com sucesso.\n")

# ==============================================================================
# SEÇÃO 2: CARREGAMENTO E AGREGAÇÃO DOS DADOS
# ==============================================================================
try:
    df_transacional = pd.read_csv('tudo.csv')
    df_transacional['Data_Pedido'] = pd.to_datetime(df_transacional['Data_Pedido'])
    print("Dataset 'tudo.csv' carregado com sucesso.")
except FileNotFoundError:
    print("Erro: Arquivo 'tudo.csv' não encontrado. Por favor, execute a Parte A do código da resposta anterior primeiro.")
    exit()

print("\n--- Agregando dados para criar a Série Temporal Mensal por Item ---")

# Agrupando por Mês e por ID_Produto, somando o volume de lotes
df_series = df_transacional.groupby(
    ['ID_Produto', pd.Grouper(key='Data_Pedido', freq='M')]
)['Volume_Vendido_Lotes'].sum().reset_index()

# Renomeando colunas para clareza
df_series.rename(columns={'Data_Pedido': 'Mes', 'Volume_Vendido_Lotes': 'Vendas_Totais_Mes'}, inplace=True)

print("Dados agregados com sucesso. Amostra da série temporal:")
print(df_series.head())

# ==============================================================================
# SEÇÃO 3: ENGENHARIA DE ATRIBUTOS PARA SÉRIES TEMPORAIS
# ==============================================================================
print("\n--- Criando features de série temporal (Lags e Médias Móveis) ---")

# Ordenar os dados é crucial para criar lags e médias móveis corretamente
df_series = df_series.sort_values(by=['ID_Produto', 'Mes'])

# Criando "Lags": vendas dos meses anteriores
# Usamos groupby() para garantir que o lag seja calculado dentro de cada grupo de produto
df_series['Vendas_Lag_1M'] = df_series.groupby('ID_Produto')['Vendas_Totais_Mes'].shift(1) # Vendas do mês anterior
df_series['Vendas_Lag_2M'] = df_series.groupby('ID_Produto')['Vendas_Totais_Mes'].shift(2) # Vendas de 2 meses atrás
df_series['Vendas_Lag_3M'] = df_series.groupby('ID_Produto')['Vendas_Totais_Mes'].shift(3) # Vendas de 3 meses atrás

# Criando "Médias Móveis": tendência dos últimos meses
df_series['Vendas_Media_Movel_3M'] = df_series.groupby('ID_Produto')['Vendas_Totais_Mes'].shift(1).rolling(window=3, min_periods=1).mean()
df_series['Vendas_Media_Movel_6M'] = df_series.groupby('ID_Produto')['Vendas_Totais_Mes'].shift(1).rolling(window=6, min_periods=1).mean()

# Criando Features baseadas na data
df_series['Mes_Numero'] = df_series['Mes'].dt.month
df_series['Ano'] = df_series['Mes'].dt.year

# Remover linhas com NaN (geradas pelos lags iniciais)
df_modelo = df_series.dropna()

print("Features de série temporal criadas. Amostra do dataframe do modelo:")
print(df_modelo.head())

# ==============================================================================
# SEÇÃO 4: PRÉ-PROCESSAMENTO E DIVISÃO DOS DADOS
# ==============================================================================
print("\n--- Pré-processamento e Divisão dos Dados ---")

# Codificação da variável categórica 'ID_Produto'
df_modelo = pd.get_dummies(df_modelo, columns=['ID_Produto'])

# Remover a coluna 'Mes' original, pois já extraímos as informações dela
df_modelo = df_modelo.drop('Mes', axis=1)

# --- Divisão Temporal Dinâmica (Forma Corrigida e Robusta) ---
# Vamos usar os últimos 6 meses de dados como conjunto de teste, não importa quando os dados terminem.

# Criar uma coluna de data temporária para facilitar a divisão
# Usamos .astype(str) aqui porque estamos aplicando a uma SÉRIE (coluna) inteira do Pandas, o que é correto.
df_modelo['Data_Completa_Temp'] = pd.to_datetime(df_modelo['Ano'].astype(str) + '-' + df_modelo['Mes_Numero'].astype(str))

# Calcular a data de corte (6 meses antes do último mês nos dados)
data_corte = df_modelo['Data_Completa_Temp'].max() - pd.DateOffset(months=6)
print(f"Data de corte para o conjunto de teste: após {data_corte.strftime('%Y-%m')}")

# Dividir os dados com base na data de corte
train = df_modelo[df_modelo['Data_Completa_Temp'] <= data_corte]
test = df_modelo[df_modelo['Data_Completa_Temp'] > data_corte]

# Agora podemos remover a coluna temporária, pois ela não é uma feature para o modelo
train = train.drop('Data_Completa_Temp', axis=1)
test = test.drop('Data_Completa_Temp', axis=1)

# Separar features (X) e alvo (y)
X_train = train.drop('Vendas_Totais_Mes', axis=1)
y_train = train['Vendas_Totais_Mes']
X_test = test.drop('Vendas_Totais_Mes', axis=1)
y_test = test['Vendas_Totais_Mes']

print(f"Dados divididos temporalmente: {len(X_train)} para treino, {len(X_test)} para teste.")
# ==============================================================================
# SEÇÃO 5: TREINAMENTO E AVALIAÇÃO DO MODELO
# ==============================================================================
print("\n--- Treinando o Modelo Random Forest ---")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=2)
model.fit(X_train, y_train)

print("Modelo treinado. Avaliando no conjunto de teste...")
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"MAE (Erro Médio Absoluto) no conjunto de teste: {mae:.2f} lotes")

media_vendas_teste = y_test.mean()
print(f"Média de Vendas Mensais no conjunto de teste: {media_vendas_teste:.2f} lotes")
# ------------------------------------

# Visualizando as previsões para um item de exemplo
plt.figure(figsize=(15, 6))

# Visualizando as previsões para um item de exemplo
plt.figure(figsize=(15, 6))
produto_exemplo = 'PROD_J01'
idx_produto_exemplo = test['ID_Produto_' + produto_exemplo] == 1
plt.plot(test.loc[idx_produto_exemplo].index, test.loc[idx_produto_exemplo, 'Vendas_Totais_Mes'], label='Vendas Reais', marker='o')
plt.plot(test.loc[idx_produto_exemplo].index, predictions[idx_produto_exemplo], label='Vendas Previstas', linestyle='--', marker='x')
plt.title(f'Previsão vs. Real para o Produto: {produto_exemplo}')
plt.legend()
plt.show()

# ==============================================================================
# SEÇÃO 6: GERANDO A PREVISÃO PARA O PRÓXIMO MÊS
# ==============================================================================
print("\n--- Gerando a Previsão de Vendas para o Próximo Mês (Agosto de 2025) ---")

# Preparando o dataframe para a previsão futura
previsao_data = []
data_base_previsao = df_series[df_series['Mes'] == df_series['Mes'].max()]

for index, linha in data_base_previsao.iterrows():
    id_produto = linha['ID_Produto']
    
    # Coletando os últimos dados conhecidos para calcular as features futuras
    ultimas_vendas = df_series[df_series['ID_Produto'] == id_produto].sort_values('Mes', ascending=False)

    # Criando as features para o mês que queremos prever
    features_futuras = {
        'Vendas_Lag_1M': ultimas_vendas['Vendas_Totais_Mes'].iloc[0], # Vendas de Julho
        'Vendas_Lag_2M': ultimas_vendas['Vendas_Totais_Mes'].iloc[1], # Vendas de Junho
        'Vendas_Lag_3M': ultimas_vendas['Vendas_Totais_Mes'].iloc[2], # Vendas de Maio
        'Vendas_Media_Movel_3M': ultimas_vendas['Vendas_Totais_Mes'].iloc[0:3].mean(),
        'Vendas_Media_Movel_6M': ultimas_vendas['Vendas_Totais_Mes'].iloc[0:6].mean(),
        'Mes_Numero': 8, # Previsão para Agosto
        'Ano': 2025,
        'ID_Produto': id_produto
    }
    previsao_data.append(features_futuras)

# Criando o dataframe com os dados futuros
df_previsao = pd.DataFrame(previsao_data)

# Aplicando o mesmo pré-processamento
df_previsao_encoded = pd.get_dummies(df_previsao, columns=['ID_Produto'])

# Alinhando as colunas com o modelo treinado
# Isso garante que todas as colunas de produtos existam, preenchendo com 0 as que não se aplicam
X_previsao_final = df_previsao_encoded.reindex(columns=X_train.columns, fill_value=0)

# Fazendo a previsão final
previsao_final = model.predict(X_previsao_final)

# Organizando e exibindo o resultado
df_resultado_final = pd.DataFrame({
    'ID_Produto': df_previsao['ID_Produto'],
    'Previsao_Vendas_Agosto_2025': previsao_final.round().astype(int)
})

print("\nResultado Final da Previsão:")
print(df_resultado_final)