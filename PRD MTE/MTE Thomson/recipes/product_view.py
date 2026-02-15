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
from utils.libutils.vectorStores.utils import VectorStoreUtils

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import datetime
import pandas as pd
import numpy as np

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# CARREGAR DATASETS DE ENTRADA
print("Carregando datasets...")

# Dataset 1: Previsões Multi-Horizonte (Histórico Recente + Futuro)
# Entity: new_multihorizon_products_abcxyz
df_forecast = Helpers.getEntityData(context, 'new_multihorizon_products_abcxyz')

# Dataset 2: Histórico de Vendas Mensal
# Entity: new_monthly_portalvendas
df_historical = Helpers.getEntityData(context, 'new_monthly_portalvendas')

# Dataset 3: Estrutura de Produto (BOM)
# Entity: estrutura_produto
df_bom = Helpers.getEntityData(context, 'df_estrutura_produto')

print(f"Datasets carregados:")
print(f"  - Forecast (Bruto): {df_forecast.shape}")
print(f"  - Historical: {df_historical.shape}")
print(f"  - BOM: {df_bom.shape}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# FUNÇÕES AUXILIARES
def sanitize_value(value):
    """
    Sanitiza valores numéricos: null/NaN/'' → 0
    """
    if pd.isna(value) or value == '' or value == '<NA>':
        return 0
    try:
        num = float(value)
        if np.isnan(num) or np.isinf(num):
            return 0
        return num
    except (ValueError, TypeError):
        return 0

def sanitize_date(date):
    """
    Sanitiza datas: inválidas → None
    """
    if pd.isna(date):
        return None
    try:
        return pd.to_datetime(date)
    except:
        return None

def extract_base_code(code: str) -> str:
    """
    Remove sufixos conhecidos para obter o código base do produto.
    Exemplos: R4261AR → R4261, VT443.82BA → VT443.82
    """
    KNOWN_SUFFIXES = ['AR', 'BM', 'HM', 'IF', 'IN', 'RD', 'BA', 'AK',
                      'EF', 'SL', 'SM', 'LM', 'EP', 'PE', 'HJ', 'HK',
                      'F', 'S', 'H', 'N', 'D']

    code = str(code).strip()

    # Ordenar por tamanho decrescente (AR antes de A, etc.)
    for suffix in sorted(KNOWN_SUFFIXES, key=len, reverse=True):
        if code.endswith(suffix) and len(code) > len(suffix):
            base = code[:-len(suffix)]
            # Verificar se base termina com número ou caractere válido (./-)
            if base and (base[-1].isdigit() or base[-1] in './-'):
                return base

    return code

print("Funcoes auxiliares definidas")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==============================================================================
# ETAPA 1: PROCESSAR FORECAST (Previsões)
# Objetivo: Limpar dados, manter APENAS o horizonte futuro e remover inválidos
# ==============================================================================
print("Processando previsoes de vendas...")

df_forecast_clean = df_forecast.copy()

# 1. Garantir tipos de dados corretos (Crítico para filtragem de datas)
df_forecast_clean['DATA_PEDIDO'] = pd.to_datetime(df_forecast_clean['DATA_PEDIDO'])
df_forecast_clean['base_date'] = pd.to_datetime(df_forecast_clean['base_date'])
df_forecast_clean['COD_MTE_COMP'] = df_forecast_clean['COD_MTE_COMP'].astype(str)
df_forecast_clean['QTDE_PEDIDA'] = df_forecast_clean['QTDE_PEDIDA'].apply(sanitize_value)

# 2. FILTRO TEMPORAL: MANTER APENAS O FUTURO
# O dataset original traz histórico + previsão. Aqui filtramos apenas o que é previsão.
print(f"  Registros antes do corte temporal: {len(df_forecast_clean)}")

df_forecast_clean = df_forecast_clean[
    df_forecast_clean['DATA_PEDIDO'] > df_forecast_clean['base_date']
]

print(f"  Registros apos manter apenas previsao futura: {len(df_forecast_clean)}")

# 3. FILTRO DE VALIDADE: Remover registros com quantidade zero ou negativa
mask_forecast_zero = df_forecast_clean['QTDE_PEDIDA'] <= 0
df_forecast_clean = df_forecast_clean[~mask_forecast_zero]

# 4. Ordenação final
df_forecast_clean = df_forecast_clean.sort_values(['COD_MTE_COMP', 'base_date', 'DATA_PEDIDO'])

# 5. CONSOLIDAR VARIANTES NO CÓDIGO BASE
print("Consolidando variantes no codigo base (forecast)...")
df_forecast_clean['COD_BASE'] = df_forecast_clean['COD_MTE_COMP'].apply(extract_base_code)

# Agregar: somar quantidades, manter primeira classificação ABC/XYZ
df_forecast_clean = df_forecast_clean.groupby(
    ['COD_BASE', 'DATA_PEDIDO', 'base_date'], as_index=False
).agg({
    'QTDE_PEDIDA': 'sum',
    'classe_abc': 'first',
    'classe_xyz': 'first',
    'classe_abc_xyz': 'first'
})

# Renomear para manter nome original da coluna
df_forecast_clean = df_forecast_clean.rename(columns={'COD_BASE': 'COD_MTE_COMP'})

print(f"Forecast processado (Final): {df_forecast_clean.shape}")
print(f"  Produtos unicos apos consolidacao: {df_forecast_clean['COD_MTE_COMP'].nunique()}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 2: PROCESSAR HISTÓRICO DE VENDAS
print("Processando historico de vendas...")

df_historical_clean = df_historical.copy()

# Garantir tipos de dados corretos
df_historical_clean['MONTH'] = pd.to_datetime(df_historical_clean['MONTH'])
df_historical_clean['COD_MTE_COMP'] = df_historical_clean['COD_MTE_COMP'].astype(str)

# Sanitizar colunas numéricas
numeric_cols = ['QTDE_PEDIDA', 'QTDE_SALDO', 'QTDE_ENTREGUE', 'VALOR_FATURADO', 'VALOR_SALDO', 'VALOR_PEDIDO']
for col in numeric_cols:
    if col in df_historical_clean.columns:
        df_historical_clean[col] = df_historical_clean[col].apply(sanitize_value)

# Ordenar por produto e mês
df_historical_clean = df_historical_clean.sort_values(['COD_MTE_COMP', 'MONTH'])

# Filtrar apenas os últimos 2 anos de histórico
data_corte = pd.Timestamp.now() - pd.DateOffset(years=2)
df_historical_clean = df_historical_clean[df_historical_clean['MONTH'] >= data_corte]

# CONSOLIDAR VARIANTES NO CÓDIGO BASE
print("Consolidando variantes no codigo base (historico)...")
df_historical_clean['COD_BASE'] = df_historical_clean['COD_MTE_COMP'].apply(extract_base_code)

# Agregar quantidades por código base e mês
numeric_cols_hist = ['QTDE_PEDIDA', 'QTDE_SALDO', 'QTDE_ENTREGUE',
                     'VALOR_FATURADO', 'VALOR_SALDO', 'VALOR_PEDIDO']

df_historical_clean = df_historical_clean.groupby(
    ['COD_BASE', 'MONTH'], as_index=False
)[numeric_cols_hist].sum()

# Renomear para manter nome original da coluna
df_historical_clean = df_historical_clean.rename(columns={'COD_BASE': 'COD_MTE_COMP'})

print(f"Historico processado: {df_historical_clean.shape}")
print(f"  Produtos unicos apos consolidacao: {df_historical_clean['COD_MTE_COMP'].nunique()}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 3: PROCESSAR ESTRUTURA DE PRODUTO (BOM)
print("Processando estrutura de produtos...")

df_bom_clean = df_bom.copy()

# Garantir tipos de dados corretos
df_bom_clean['COD_PRODUTO'] = df_bom_clean['COD_PRODUTO'].astype(str)
df_bom_clean['COD_COMPONENTE'] = df_bom_clean['COD_COMPONENTE'].astype(str)
df_bom_clean['G1_QUANT'] = df_bom_clean['G1_QUANT'].apply(sanitize_value)

# FILTRO 2: Remover componentes com quantidade zero ou negativa
mask_bom_zero = df_bom_clean['G1_QUANT'] <= 0
df_bom_clean = df_bom_clean[~mask_bom_zero]

# Ordenar por produto e componente
df_bom_clean = df_bom_clean.sort_values(['COD_PRODUTO', 'COD_COMPONENTE'])

# CONSOLIDAR VARIANTES E FILTRAR POR HISTÓRICO
# (product_structure será usado como seletor de produtos)
produtos_validos = set(df_historical_clean['COD_MTE_COMP'].unique())

print(f"  Produtos no BOM antes do filtro: {df_bom_clean['COD_PRODUTO'].nunique()}")

# Consolidar variantes no BOM
df_bom_clean['COD_PRODUTO_BASE'] = df_bom_clean['COD_PRODUTO'].apply(extract_base_code)

# Agregar quantidades por produto base e componente
df_bom_clean = df_bom_clean.groupby(
    ['COD_PRODUTO_BASE', 'COD_COMPONENTE'], as_index=False
)['G1_QUANT'].sum()

df_bom_clean = df_bom_clean.rename(columns={'COD_PRODUTO_BASE': 'COD_PRODUTO'})

# Filtrar apenas produtos que existem no histórico
df_bom_clean = df_bom_clean[df_bom_clean['COD_PRODUTO'].isin(produtos_validos)]

print(f"BOM processado: {df_bom_clean.shape}")
print(f"  Produtos no BOM apos filtro: {df_bom_clean['COD_PRODUTO'].nunique()}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 4: SALVAR OUTPUTS
print("Salvando datasets de saida...")

# 1. Output Forecast: Resetar index para garantir limpeza
df_forecast_clean = df_forecast_clean.reset_index(drop=True)

Helpers.save_output_dataset(
    context=context,
    output_name='product_forecast',
    data_frame=df_forecast_clean
)
print(f"  Salvo: product_forecast ({df_forecast_clean.shape})")

# 2. Output Histórico: Resetar index
df_historical_clean = df_historical_clean.reset_index(drop=True)

Helpers.save_output_dataset(
    context=context,
    output_name='product_history',
    data_frame=df_historical_clean
)
print(f"  Salvo: product_history ({df_historical_clean.shape})")

# 3. Output BOM: Resetar index
df_bom_clean = df_bom_clean.reset_index(drop=True)

Helpers.save_output_dataset(
    context=context,
    output_name='product_structure',
    data_frame=df_bom_clean
)
print(f"  Salvo: product_structure ({df_bom_clean.shape})")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# RESUMO FINAL
print("=" * 80)
print("PROCESSAMENTO CONCLUIDO COM SUCESSO!")
print("=" * 80)
print("")
print("RESUMO DOS DATASETS GERADOS:")
print("")
print(f"  1. product_forecast")
print(f"     - Registros: {df_forecast_clean.shape[0]:,}")
print(f"     - Colunas: {df_forecast_clean.shape[1]}")
print(f"     - Uso: Previsoes de vendas futuras (Horizonte > Base Date)")
print("")
print(f"  2. product_history")
print(f"     - Registros: {df_historical_clean.shape[0]:,}")
print(f"     - Colunas: {df_historical_clean.shape[1]}")
print(f"     - Uso: Historico real de vendas mensais")
print("")
print(f"  3. product_structure")
print(f"     - Registros: {df_bom_clean.shape[0]:,}")
print(f"     - Colunas: {df_bom_clean.shape[1]}")
print(f"     - Uso: Estrutura BOM (componentes por produto)")
print("")
print("=" * 80)
print("")
print("ESTATISTICAS:")
print(f"  - Total de produtos: {len(df_forecast_clean['COD_MTE_COMP'].unique())}")
print(f"  - Total de datas base: {len(df_forecast_clean['base_date'].unique())}")
print(f"  - Total de componentes: {len(df_bom_clean['COD_COMPONENTE'].unique())}")
print(f"  - Periodo de previsao (Futuro): {df_forecast_clean['DATA_PEDIDO'].min()} -> {df_forecast_clean['DATA_PEDIDO'].max()}")
print(f"  - Periodo historico (Passado): {df_historical_clean['MONTH'].min()} -> {df_historical_clean['MONTH'].max()}")
print("")
print("=" * 80)