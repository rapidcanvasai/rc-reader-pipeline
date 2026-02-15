# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Required imports

from utils.notebookhelpers.helpers import Helpers
from utils.dtos.templateOutputCollection import TemplateOutputCollection
from utils.dtos.templateOutput import TemplateOutput
from utils.dtos.templateOutput import OutputType
from utils.dtos.templateOutput import ChartType
from utils.dtos.variable import Metadata
from utils.rcclient.commons.variable_datatype import VariableDatatype
from utils.dtos.templateOutput import FileType
from utils.dtos.rc_ml_model import RCMLModel
from utils.notebookhelpers.helpers import Helpers

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ================================================================================
  # RECIPE: historico_pedidos
  # ================================================================================
  # DEPENDÊNCIAS: Este recipe requer os seguintes datasets:
  #     1. historico_pedidos (EXTERNAL-Azure: dados brutos de pedidos)
  # ================================================================================
  #
  # PROPÓSITO: Calcular lead times de pedidos históricos para análise de desempenho
  #            de fornecedores e produtos, gerando entrada para cálculos de estoque
  #            de segurança e pontos de pedido.
  #
  # INPUTS:
  #   - historico_pedidos (pedidos históricos com datas e quantidades)
  #
  # OUTPUT:
  #   - df_historico_pedidos (pedidos com lead times calculados)
  #
  # FILTROS APLICADOS:
  #   1. pd.to_datetime(..., errors='coerce') (linhas 27-28): Converte strings para datetime
  #      Exemplo: df['DATA_SI'] = pd.to_datetime(df['DATA_SI'], errors='coerce')
  #      * Consequência: Datas inválidas viram NaT, não causam erro de processamento
  #
  #   2. dropna(subset=['LEAD_TIME_DIAS']) (linha 38): Remove registros sem lead time
  #      Exemplo: df.dropna(subset=['LEAD_TIME_DIAS']).groupby(...)
  #      * Consequência: Análises agregadas consideram apenas pedidos com entregas válidas
  #
  # LÓGICA:
  #   FASE 1 - CARREGAMENTO (linha 23):
  #     1. Carrega historico_pedidos da fonte externa Azure
  #
  #   FASE 2 - CONVERSÃO DE DATAS (linhas 26-28):
  #     1. Converte DATA_SI (data pedido) para datetime com tratamento de erros
  #     2. Converte ENTREGA_EFET (entrega efetiva) para datetime
  #
  #   FASE 3 - CÁLCULO DE LEAD TIME (linhas 30-34):
  #     1. LEAD_TIME_DIAS = (ENTREGA_EFET - DATA_SI).dt.days
  #     2. Print de amostra com 5 primeiros registros
  #
  #   FASE 4 - ANÁLISE POR FORNECEDOR/PRODUTO (linhas 37-51):
  #     1. Remove nulos de LEAD_TIME_DIAS
  #     2. Agrupa por COD_FORNE e PRODUTO
  #     3. Calcula estatísticas: mean, median, min, max, count
  #     4. Arredonda média para 1 casa decimal
  #     5. Ordena por mean descendente (piores fornecedores primeiro)
  #
  #   FASE 5 - ANÁLISE TEMPORAL (linhas 55-72):
  #     1. Extrai ANO e MES de DATA_SI
  #     2. Agrupa por ANO e MES
  #     3. Calcula mean, median, count de lead time
  #
  # EXEMPLO COMPLETO:
  #   INPUT:
  #     historico_pedidos:
  #       COD_FORNE="5001", PRODUTO="ABC123", DATA_SI="2024-01-15", 
  #       ENTREGA_EFET="2024-02-20"
  #   
  #   PROCESSAMENTO:
  #   → Conversão: DATA_SI=2024-01-15, ENTREGA_EFET=2024-02-20 (datetime)
  #   → Cálculo: LEAD_TIME_DIAS = (2024-02-20 - 2024-01-15) = 36 dias
  #   → Análise fornecedor: COD_FORNE="5001" + PRODUTO="ABC123" → mean=36.0
  #   → Análise temporal: ANO=2024, MES=1 → mean lead time do mês
  #   
  #   OUTPUT:
  #   {
  #     "COD_FORNE": "5001",
  #     "PRODUTO": "ABC123",
  #     "DATA_SI": "2024-01-15",
  #     "ENTREGA_EFET": "2024-02-20",
  #     "LEAD_TIME_DIAS": 36,
  #     "ANO": 2024,
  #     "MES": 1
  #   }
  # ================================================================================

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import pandas as pd

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Your code goes here
df_historico_pedidos = Helpers.getEntityData(context, 'df_historico_pedidos')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_historico_pedidos['PRECO_UNIT'] = df_historico_pedidos['PRECO_TOTAL']/df_historico_pedidos['QUANTIDADE']
df_historico_pedidos['DATA_SI'] = pd.to_datetime(df_historico_pedidos['DATA_SI'], errors='coerce')
df_historico_pedidos['ENTREGA_EFET'] = pd.to_datetime(df_historico_pedidos['ENTREGA_EFET'], errors='coerce')

df_historico_pedidos = (
    df_historico_pedidos
    .sort_values(
        ['PEDIDO', 'COD_FORNE', 'PRODUTO', 'DATA_SI', 'PRECO_UNIT'],
        ascending=[True, True, True, True, False]  # maior PRECO_UNIT primeiro
    )
    .drop_duplicates(subset=['PEDIDO', 'COD_FORNE', 'PRODUTO', 'DATA_SI'], keep='first')
    .reset_index(drop=True)
).drop(['PRECO_UNIT'], axis=1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 2. Calcular o Lead Time em dias
df_historico_pedidos['LEAD_TIME_DIAS'] = (df_historico_pedidos['ENTREGA_EFET'] - df_historico_pedidos['DATA_SI']).dt.days

print("DataFrame com Lead Time calculado:")
print(df_historico_pedidos[['DATA_SI', 'ENTREGA_EFET', 'LEAD_TIME_DIAS']].head())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 3. Agrupar por Fornecedor e Produto
analise_fornecedor_produto = df_historico_pedidos.dropna(subset=['LEAD_TIME_DIAS']).groupby(
    ['COD_FORNE', 'PRODUTO']
)['LEAD_TIME_DIAS'].agg(
    ['mean', 'median', 'min', 'max', 'count']
)

# Arredondar a média para melhor visualização
analise_fornecedor_produto['mean'] = analise_fornecedor_produto['mean'].round(1)

# Ordenar pelos maiores lead times médios
analise_fornecedor_produto = analise_fornecedor_produto.sort_values('mean', ascending=False)

print("Análise de Lead Time por Fornecedor e Produto (Top 10 piores):")
print(analise_fornecedor_produto.head(10))
print("\n")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 4. Criar colunas de Ano e Mês
df_analise_temporal = df_historico_pedidos.dropna(subset=['LEAD_TIME_DIAS', 'DATA_SI']).copy()

df_analise_temporal['ANO'] = df_analise_temporal['DATA_SI'].dt.year
df_analise_temporal['MES'] = df_analise_temporal['DATA_SI'].dt.month

# 5. Agrupar por Ano e Mês
analise_temporal = df_analise_temporal.groupby(
    ['ANO', 'MES']
)['LEAD_TIME_DIAS'].agg(
    ['mean', 'median', 'count']
)

# Arredondar a média
analise_temporal['mean'] = analise_temporal['mean'].round(1)

print("Análise de Lead Time por Ano e Mês:")
print(analise_temporal)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
Helpers.save_output_dataset(context=context, output_name='df_historico_pedidos_lead_time', data_frame=df_historico_pedidos)