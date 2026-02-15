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
from utils.rc.client.auth import AuthClient
from utils.rc.client.requests import Requests
from utils.rc.dtos.user import User

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import logging
import pandas as pd
import numpy as np
import datetime as dt
from typing import List, Dict, Tuple, Union, Optional, Callable
from functools import wraps
import holidays
import calendar
import regex as re
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    make_scorer, mean_absolute_error, mean_squared_error, 
    r2_score, mean_absolute_percentage_error, median_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- 
# FUNÇÕES AUXILIARES (MÉTRICAS E UTILIDADES)
# -------------------------------------------------------------------------------- 
def wmape(y_true, y_pred):
    """Weighted Mean Absolute Percentage Error"""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    ape = np.abs(y_true - y_pred) / y_true
    
    if np.isscalar(ape):
        if np.isfinite(ape):
            return ape
        else:
            return 1
    else:
        ape[~np.isfinite(ape)] = 1
    return np.mean(ape)

def get_metrics(df_actual, df_prediction, index=None):
    """Calcula métricas de performance"""
    if index is not None:
        df_actual = df_actual.loc[index]
        df_prediction = df_prediction.loc[index]
    
    metrics_dict = {
        "MAE": mean_absolute_error(df_actual, df_prediction),
        "MSE": mean_squared_error(df_actual, df_prediction),
        "RMSE": np.sqrt(mean_squared_error(df_actual, df_prediction)),
        "R2": r2_score(df_actual, df_prediction),
        "Total Error": np.sum(df_actual - df_prediction),
        "Percentage error": np.sum(df_actual-df_prediction)/np.sum(df_actual),
        "MAPE": mean_absolute_percentage_error(df_actual, df_prediction),
        "Median Absolute Error": median_absolute_error(df_actual, df_prediction),
    }
    return metrics_dict

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- 
# 1. CARREGAMENTO E TRATAMENTO INICIAL
# -------------------------------------------------------------------------------- 
df_historico_pedidos = Helpers.getEntityData(context, 'df_historico_pedidos_lead_time').rename(columns={
    'PRODUTO': 'COMPONENT'
})

# Obtendo valor unitario
df_historico_pedidos['PRECO_UNIT'] = df_historico_pedidos['PRECO_TOTAL']/df_historico_pedidos['QUANTIDADE']

# Tratamento de Lead Time
df_historico_pedidos['LEAD_TIME'] = df_historico_pedidos['LEAD_TIME_DIAS'] / 30

# Preenchimento de Lead Times vazios (Hierarquia: Comp+Forn -> Forn -> Global -> 3)
media_comp_fornec = np.ceil(df_historico_pedidos.groupby(['COMPONENT', 'COD_FORNE'])['LEAD_TIME'].transform('mean'))
df_historico_pedidos['LEAD_TIME'] = df_historico_pedidos['LEAD_TIME'].fillna(media_comp_fornec)

media_fornec = np.ceil(df_historico_pedidos.groupby('COD_FORNE')['LEAD_TIME'].transform('mean'))
df_historico_pedidos['LEAD_TIME'] = df_historico_pedidos['LEAD_TIME'].fillna(media_fornec)

media_global = np.ceil(df_historico_pedidos['LEAD_TIME'].mean())
df_historico_pedidos['LEAD_TIME'] = df_historico_pedidos['LEAD_TIME'].fillna(media_global)

# Limpeza e Conversão
df_historico_pedidos['LEAD_TIME'] = df_historico_pedidos['LEAD_TIME'].replace([np.inf, -np.inf], 1)
df_historico_pedidos['LEAD_TIME'] = np.ceil(df_historico_pedidos['LEAD_TIME']).astype(int)

# Caso o lead time seja menor que 1 mes (algo raro, optamos por usar o lead time médio daquele fornecedor, mas se esse valor for NaN, então
# usamos um lead time de 3 meses
df_historico_pedidos.loc[
    df_historico_pedidos['LEAD_TIME'] == 1,
    'LEAD_TIME'
] = media_fornec.fillna(3)

# Tornando as colunas datas
df_historico_pedidos['DATA_SI'] = pd.to_datetime(df_historico_pedidos['DATA_SI'])
df_historico_pedidos['MES'] = df_historico_pedidos['DATA_SI'].dt.to_period('M').dt.to_timestamp()

# Mantendo apenas uma observacao por "PEDIDO", "COD_FORNE", "COMPONENT", "DATA_SI"
df_historico_pedidos = (
    df_historico_pedidos
    .sort_values("PRECO_TOTAL", ascending=False)
    .drop_duplicates(subset=["MES", "COD_FORNE", "COMPONENT", "DATA_SI"], keep="first")
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- 
# 1.2. PADRONIZAÇÃO MENSAL DO HISTÓRICO
# -------------------------------------------------------------------------------- 
print("📅 Agregando dados mensalmente e preenchendo lacunas...")

# Agregar mensalmente
df_monthly = df_historico_pedidos.groupby(['COMPONENT', 'COD_FORNE', 'MES']).agg({
    'QUANTIDADE': 'sum',
    'LEAD_TIME': 'max' 
}).reset_index()

# Garantir continuidade temporal (Cross Join)
unique_pairs = df_monthly[['COMPONENT', 'COD_FORNE']].drop_duplicates()
all_months = pd.date_range(df_monthly['MES'].min(), df_monthly['MES'].max(), freq='MS')
df_dates = pd.DataFrame({'MES': all_months})

unique_pairs['key'] = 1
df_dates['key'] = 1
df_full_idx = pd.merge(unique_pairs, df_dates, on='key').drop('key', axis=1)

# Merge final
df_final = pd.merge(df_full_idx, df_monthly, on=['COMPONENT', 'COD_FORNE', 'MES'], how='left')

# Preencher Nulos
df_final['QUANTIDADE'] = df_final['QUANTIDADE'].fillna(0)
df_final['LEAD_TIME'] = df_final.groupby(['COMPONENT', 'COD_FORNE'])['LEAD_TIME'].ffill().bfill().fillna(1).astype(int)

df_final = df_final.sort_values(['COMPONENT', 'COD_FORNE', 'MES'])

print(f"✅ Base mensal padronizada: {df_final.shape}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_vendas = Helpers.getEntityData(context, 'df_vendas')
df_vendas['QUANTIDADE'] = df_vendas['QTD_VENDA_NACIONAL'] + df_vendas['QTD_VENDA_EXPORTACAO']
df_vendas['MES'] = df_vendas['DATA'].dt.to_period('M').dt.to_timestamp()
df_vendas = df_vendas[['B1_COD', 'MES', 'QUANTIDADE']].rename(columns = {'B1_COD': 'COMPONENT'})
df_vendas = df_vendas.groupby(['COMPONENT', 'MES']).agg({
    'QUANTIDADE': 'sum' 
}).reset_index().sort_values(['COMPONENT',  'MES'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# 3. CRIAÇÃO DO TARGET HÍBRIDO (INTERNO + EXTERNO/VENDAS)
# ==================================================================================
print("\n🎯 Criando TARGET (Prioridade: Real Futuro Interno > Real Futuro Vendas)...")

df_final['TARGET'] = np.nan 
lead_times_unicos = df_final['LEAD_TIME'].unique()

# A) FASE 1: Tentar preencher com dados REAIS da própria base (Shift Negativo no df_final)
print("\n   Fase 1: Preenchendo com dados reais futuros (interno)...")
for lt in lead_times_unicos:
    if pd.isna(lt) or lt <= 0: 
        continue
    lt_meses = int(lt)
    
    # Busca venda real no futuro do dataset atual
    shift_futuro = df_final.groupby(['COMPONENT', 'COD_FORNE'])['QUANTIDADE'].shift(-lt_meses)
    
    mask = (df_final['LEAD_TIME'] == lt)
    df_final.loc[mask, 'TARGET'] = shift_futuro[mask]

n_nans_antes = df_final['TARGET'].isna().sum()
print(f"   TARGETs vazios após dados internos: {n_nans_antes}")

# B) FASE 2: Preencher NaNs restantes usando o df_vendas
if n_nans_antes > 0:
    print("\n   Fase 2: Preenchendo com dados do df_vendas...")
    
    df_vendas_clean = df_vendas.copy()
    df_vendas_clean['QUANTIDADE'] = df_vendas_clean['QUANTIDADE'].fillna(0)
    
    # Calcular DATA_ALVO_NECESSARIA
    df_final['DATA_ALVO_NECESSARIA'] = [
        m + pd.DateOffset(months=int(lt)) 
        for m, lt in zip(df_final['MES'], df_final['LEAD_TIME'])
    ]
    
    print(f"      Período necessário: {df_final['DATA_ALVO_NECESSARIA'].min()} até {df_final['DATA_ALVO_NECESSARIA'].max()}")
    print(f"      Período disponível (df_vendas): {df_vendas_clean['MES'].min()} até {df_vendas_clean['MES'].max()}")
    
    # Merge com df_vendas
    df_merged = pd.merge(
        df_final,
        df_vendas_clean[['COMPONENT', 'MES', 'QUANTIDADE']], 
        left_on=['COMPONENT', 'DATA_ALVO_NECESSARIA'],      
        right_on=['COMPONENT', 'MES'],                      
        how='left',
        suffixes=('', '_vendas')
    )
    
    # Preencher TARGET onde ainda está vazio
    mask_nan = df_final['TARGET'].isna()
    
    if 'QUANTIDADE_vendas' in df_merged.columns:
        coluna_alvo = 'QUANTIDADE_vendas'
    else:
        coluna_alvo = 'QUANTIDADE' 

    df_final.loc[mask_nan, 'TARGET'] = df_merged.loc[mask_nan, coluna_alvo]
    
    # Limpar colunas auxiliares
    df_final = df_final.drop(columns=['DATA_ALVO_NECESSARIA'])
    
    n_nans_depois = df_final['TARGET'].isna().sum()
    print(f"   TARGETs vazios após df_vendas: {n_nans_depois}")
    print(f"   ✅ Recuperamos {n_nans_antes - n_nans_depois} linhas!")

print(f"\n✅ TARGET criado com sucesso!")
print(f"   Total de linhas: {len(df_final):,}")
print(f"   TARGETs válidos: {df_final['TARGET'].notna().sum():,}")
print(f"   TARGETs NaN (Mantidos): {df_final['TARGET'].isna().sum()}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- 
# CLASSIFICAÇÃO ABC (PARETO DE VOLUME DE COMPRAS)
# -------------------------------------------------------------------------------- 
print("\n" + "="*80)
print("📊 CLASSIFICAÇÃO ABC (ANÁLISE DE PARETO)")
print("="*80)

# Agregar volume total de compras por COMPONENT (últimos 12 meses para classificação)
last_12_months = df_final['MES'].max() - pd.DateOffset(months=12)
df_abc = df_final[df_final['MES'] >= last_12_months].groupby('COMPONENT')['QUANTIDADE'].sum().reset_index()
df_abc = df_abc.sort_values('QUANTIDADE', ascending=False).reset_index(drop=True)

# Calcular Pareto
df_abc['cumsum'] = df_abc['QUANTIDADE'].cumsum()
total_volume = df_abc['QUANTIDADE'].sum()
df_abc['cumsum_pct'] = (df_abc['cumsum'] / total_volume) * 100

def get_abc_class(cumsum_pct):
    if cumsum_pct <= 80: return 'A'
    elif cumsum_pct <= 95: return 'B'
    else: return 'C'

df_abc['CLASSE_ABC'] = df_abc['cumsum_pct'].apply(get_abc_class)

# Criar mapa COMPONENT -> Classe ABC
product_class_map = df_abc.set_index('COMPONENT')['CLASSE_ABC'].to_dict()

# Aplicar ao dataset
df_final['CLASSE_ABC'] = df_final['COMPONENT'].map(product_class_map).fillna('C')

print(f"\n   📊 Distribuição de COMPONENTs por Classe ABC:")
for classe in ['A', 'B', 'C']:
    n_components = (df_abc['CLASSE_ABC'] == classe).sum()
    volume = df_abc[df_abc['CLASSE_ABC'] == classe]['QUANTIDADE'].sum()
    pct_volume = (volume / total_volume) * 100
    print(f"      Classe {classe}: {n_components:,} COMPONENTs ({pct_volume:.1f}% do volume)")

# ==================================================================================
# ✅ SOLUÇÃO 1: FILTRAR COMPONENTES ESPORÁDICOS
# ==================================================================================
print("\n" + "="*80)
print("🔍 SOLUÇÃO 1: FILTRAR COMPONENTES ESPORÁDICOS")
print("="*80)

# Calcular % de meses com compra por COMPONENT+FORNECEDOR
df_freq = df_final.groupby(['COMPONENT', 'COD_FORNE']).agg({
    'QUANTIDADE': lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0
}).reset_index()
df_freq.columns = ['COMPONENT', 'COD_FORNE', 'FREQ_COMPRA']

print(f"\n📊 Distribuição de Frequência de Compra:")
print(df_freq['FREQ_COMPRA'].describe())
print(f"\nQuantis:")
print(df_freq['FREQ_COMPRA'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))

# ✅ THRESHOLD: Manter apenas pares que compram em >= 20% dos meses
THRESHOLD_FREQ = 0.20

componentes_validos = df_freq[df_freq['FREQ_COMPRA'] >= THRESHOLD_FREQ]

print(f"\n✂️  Filtragem com threshold = {THRESHOLD_FREQ*100}%:")
print(f"   Total pares COMPONENT+FORNECEDOR: {len(df_freq):,}")
print(f"   Com frequência >= {THRESHOLD_FREQ*100}%: {len(componentes_validos):,}")
print(f"   Removidos: {len(df_freq) - len(componentes_validos):,} ({(1-len(componentes_validos)/len(df_freq))*100:.1f}%)")

# Aplicar filtro
df_final_antes = df_final.copy()
df_final = pd.merge(
    df_final,
    componentes_validos[['COMPONENT', 'COD_FORNE']],
    on=['COMPONENT', 'COD_FORNE'],
    how='inner'
)

print(f"\n📉 Impacto no Dataset:")
print(f"   Antes: {len(df_final_antes):,} linhas")
print(f"   Depois: {len(df_final):,} linhas")
print(f"   Redução: {(1-len(df_final)/len(df_final_antes))*100:.1f}%")

# 🎯 VERIFICAR DISTRIBUIÇÃO DE ZEROS
n_zeros_antes = (df_final_antes['TARGET'] == 0).sum()
n_zeros_depois = (df_final['TARGET'] == 0).sum()

print(f"\n🎯 Distribuição de TARGET:")
print(f"   ANTES:")
print(f"      Zeros: {n_zeros_antes:,} ({n_zeros_antes/len(df_final_antes)*100:.1f}%)")
print(f"      Não-zeros: {len(df_final_antes) - n_zeros_antes:,} ({(1-n_zeros_antes/len(df_final_antes))*100:.1f}%)")
print(f"   DEPOIS:")
print(f"      Zeros: {n_zeros_depois:,} ({n_zeros_depois/len(df_final)*100:.1f}%)")
print(f"      Não-zeros: {len(df_final) - n_zeros_depois:,} ({(1-n_zeros_depois/len(df_final))*100:.1f}%)")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# 4. CRIAÇÃO DAS FEATURES DE HISTÓRICO (LAGS)
# ==================================================================================
print("\n⏪ Criando colunas de histórico anterior (Lags 1 a 12)...")

cols_lags = []
for i in range(1, 13):
    col_name = f'HISTORICO_VENDAS_LAG{i}'
    cols_lags.append(col_name)
    df_final[col_name] = df_final.groupby(['COMPONENT', 'COD_FORNE'])['QUANTIDADE'].shift(i)

print(f"   ✅ {len(cols_lags)} LAGs criados")

# ==================================================================================
# ✅ SOLUÇÃO 2: FEATURES DE ESPORADICIDADE
# ==================================================================================
print("\n" + "="*80)
print("📊 SOLUÇÃO 2: CRIANDO FEATURES DE ESPORADICIDADE")
print("="*80)

df_final = df_final.sort_values(['COMPONENT', 'COD_FORNE', 'MES'])

# 1. Meses desde última compra
print("   Criando: MESES_DESDE_ULTIMA_COMPRA...")
df_final['MESES_DESDE_ULTIMA_COMPRA'] = 0

for (comp, forn), group in df_final.groupby(['COMPONENT', 'COD_FORNE']):
    meses_sem_compra = 0
    valores = []
    
    for idx, row in group.iterrows():
        valores.append(meses_sem_compra)
        
        if row['QUANTIDADE'] > 0:
            meses_sem_compra = 0
        else:
            meses_sem_compra += 1
    
    df_final.loc[group.index, 'MESES_DESDE_ULTIMA_COMPRA'] = valores

# 2. Frequência de compra (rolling 12 meses)
print("   Criando: FREQ_COMPRA_12M...")
def calc_freq_compra(series):
    return (series > 0).sum() / len(series) if len(series) > 0 else 0

df_final['FREQ_COMPRA_12M'] = (
    df_final.groupby(['COMPONENT', 'COD_FORNE'])['QUANTIDADE']
    .transform(lambda x: x.rolling(12, min_periods=1).apply(calc_freq_compra, raw=False))
)

# 3. Valor médio quando compra
print("   Criando: VALOR_MEDIO_QUANDO_COMPRA...")
def valor_medio_quando_compra(series):
    valores_positivos = series[series > 0]
    return valores_positivos.mean() if len(valores_positivos) > 0 else 0

df_final['VALOR_MEDIO_QUANDO_COMPRA'] = (
    df_final.groupby(['COMPONENT', 'COD_FORNE'])['QUANTIDADE']
    .transform(lambda x: x.rolling(12, min_periods=1).apply(valor_medio_quando_compra, raw=False))
)

# 4. Flag: Comprou no mês anterior?
print("   Criando: COMPROU_MES_ANTERIOR...")
df_final['COMPROU_MES_ANTERIOR'] = (
    df_final.groupby(['COMPONENT', 'COD_FORNE'])['QUANTIDADE']
    .shift(1) > 0
).astype(int)

print(f"\n   ✅ 4 Features de esporadicidade criadas!")

# ==================================================================================
# 5. CRIAÇÃO DAS FEATURES DE SAZONALIDADE E OUTRAS
# ==================================================================================
df_final['MES_NUM'] = df_final['MES'].dt.month
df_final['TRIMESTRE'] = df_final['MES'].dt.quarter
df_final['ANO'] = df_final['MES'].dt.year

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- 
# CRIAÇÃO DE FEATURES ADICIONAIS
# -------------------------------------------------------------------------------- 
print("\n" + "="*80)
print("🛠️ CRIAÇÃO DE FEATURES ADICIONAIS")
print("="*80)

df_final = df_final.sort_values(['COMPONENT', 'COD_FORNE', 'MES'])

# 1. Diferenças
print("   Criando diferenças...")
df_final['diff_1'] = df_final['QUANTIDADE'] - df_final['HISTORICO_VENDAS_LAG1']
df_final['diff_2'] = df_final['HISTORICO_VENDAS_LAG1'] - df_final['HISTORICO_VENDAS_LAG2']
df_final['diff_3'] = df_final['HISTORICO_VENDAS_LAG2'] - df_final['HISTORICO_VENDAS_LAG3']

# 2. Rolling Statistics
print("   Criando rolling statistics...")

# Rolling Mean
df_final['rolling_mean_3'] = df_final[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3']].mean(axis=1)
df_final['rolling_mean_6'] = df_final[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3', 
                                        'HISTORICO_VENDAS_LAG4', 'HISTORICO_VENDAS_LAG5', 'HISTORICO_VENDAS_LAG6']].mean(axis=1)
df_final['rolling_mean_12'] = df_final[[f'HISTORICO_VENDAS_LAG{i}' for i in range(1, 13)]].mean(axis=1)

# Rolling Std
df_final['rolling_std_3'] = df_final[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3']].std(axis=1)
df_final['rolling_std_6'] = df_final[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3', 
                                       'HISTORICO_VENDAS_LAG4', 'HISTORICO_VENDAS_LAG5', 'HISTORICO_VENDAS_LAG6']].std(axis=1)
df_final['rolling_std_12'] = df_final[[f'HISTORICO_VENDAS_LAG{i}' for i in range(1, 13)]].std(axis=1)

# Rolling Max
df_final['rolling_max_3'] = df_final[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3']].max(axis=1)
df_final['rolling_max_6'] = df_final[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3', 
                                       'HISTORICO_VENDAS_LAG4', 'HISTORICO_VENDAS_LAG5', 'HISTORICO_VENDAS_LAG6']].max(axis=1)
df_final['rolling_max_12'] = df_final[[f'HISTORICO_VENDAS_LAG{i}' for i in range(1, 13)]].max(axis=1)

# Rolling Min
df_final['rolling_min_3'] = df_final[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3']].min(axis=1)
df_final['rolling_min_6'] = df_final[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3', 
                                       'HISTORICO_VENDAS_LAG4', 'HISTORICO_VENDAS_LAG5', 'HISTORICO_VENDAS_LAG6']].min(axis=1)
df_final['rolling_min_12'] = df_final[[f'HISTORICO_VENDAS_LAG{i}' for i in range(1, 13)]].min(axis=1)

# 3. Lag do lead time
print("   Criando lag de lead time...")
df_final['lead_time_lag_1'] = df_final.groupby(['COMPONENT', 'COD_FORNE'])['LEAD_TIME'].shift(1)

# 4. Features temporais
print("   Criando features temporais...")
df_final['month'] = df_final['MES'].dt.month
df_final['quarter'] = df_final['MES'].dt.quarter
df_final['quarter_start'] = df_final['MES'].dt.is_quarter_start.astype(int)
df_final['quarter_end'] = df_final['MES'].dt.is_quarter_end.astype(int)

# One-Hot para mês
df_final = pd.get_dummies(df_final, columns=['month'], prefix='month')

# 5. Limpar valores infinitos e NaNs
print("   Limpando valores infinitos e NaNs...")
numeric_cols = df_final.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df_final[col] = df_final[col].replace([np.inf, -np.inf], np.nan)
    
# Preencher NaNs nas rolling statistics
rolling_cols = [col for col in df_final.columns if 'rolling_' in col]
for col in rolling_cols:
    df_final[col] = df_final[col].fillna(0)

max_date = df_final['MES'].max()
df_final = df_final[df_final['MES'] < max_date]
print(f"   ✅ Features criadas: {len(df_final.columns)} colunas totais")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- 
# PREPARAÇÃO PARA MODELAGEM
# -------------------------------------------------------------------------------- 
print("\n" + "="*80)
print("📐 PREPARAÇÃO PARA MODELAGEM")
print("="*80)

df_model_data = df_final.copy()

print(f"   Dataset para modelagem: {df_model_data.shape}")

# Definir features
ignore_cols = [
    'COMPONENT', 'COD_FORNE', 'MES', 'TARGET', 'DATA_SI', 'CLASSE_ABC', 'QUANTIDADE'
]
ignore_cols = [c for c in ignore_cols if c in df_model_data.columns]

all_possible_features = [c for c in df_model_data.columns if c not in ignore_cols]

# Filtrar apenas numéricas
features = []
for col in all_possible_features:
    dtype = df_model_data[col].dtype
    if dtype in ['int64', 'float64', 'int32', 'float32', 'bool', 'uint8', 'int8']:
        features.append(col)

print(f"\n   ✅ Features numéricas selecionadas: {len(features)}")

# Split temporal (80/20)
months = sorted(df_model_data['MES'].unique())
split_idx = int(len(months) * 0.8)
train_months = months[:split_idx]
test_months = months[split_idx:]

df_train = df_model_data[df_model_data['MES'].isin(train_months)].copy()
df_test = df_model_data[df_model_data['MES'].isin(test_months)].copy()

print(f"\n   📅 Split Temporal:")
print(f"      Treino: {len(train_months)} meses ({train_months[0]} até {train_months[-1]})")
print(f"      Teste:  {len(test_months)} meses ({test_months[0]} até {test_months[-1]})")
print(f"      Registros Treino: {len(df_train):,}")
print(f"      Registros Teste:  {len(df_test):,}")

# Preencher NaNs
print(f"\n   🧹 Preenchendo NaNs nas features...")
df_train[features] = df_train[features].fillna(0)
df_test[features] = df_test[features].fillna(0)

print(f"\n   ✅ Dados preparados para treinamento!")

# ==================================================================================
# ✅ SOLUÇÃO 3: FUNÇÃO TWO-STAGE MODEL
# ==================================================================================

def train_abc_two_stage_models(df_train, df_test, features, product_class_map, n_iter_search=30, verbose=True):
    """
    Modelo Two-Stage:
    - Stage 1: Classificação (vai comprar ou não?)
    - Stage 2: Regressão (se sim, quanto?)
    """
    
    if verbose:
        print("\n" + "="*80)
        print("🚀 SOLUÇÃO 3: TREINAMENTO TWO-STAGE (CLASSIFICAÇÃO + REGRESSÃO)")
        print("="*80)
    
    # Classificar treino e teste
    classes_train = df_train['COMPONENT'].map(product_class_map).fillna('C').values
    classes_test = df_test['COMPONENT'].map(product_class_map).fillna('C').values
    
    if verbose:
        n_A = (classes_train == 'A').sum()
        n_B = (classes_train == 'B').sum()
        n_C = (classes_train == 'C').sum()
        
        print(f"\n📊 Distribuição de LINHAS de treino:")
        print(f"    Classe A: {n_A:,} linhas ({n_A/len(classes_train)*100:.1f}%)")
        print(f"    Classe B: {n_B:,} linhas ({n_B/len(classes_train)*100:.1f}%)")
        print(f"    Classe C: {n_C:,} linhas ({n_C/len(classes_train)*100:.1f}%)")
    
    # Preparar dados
    X_train = df_train[features].fillna(0).astype('float64')
    y_train = df_train["TARGET"].fillna(0).astype('float64')
    X_test = df_test[features].fillna(0).astype('float64')
    y_test = df_test["TARGET"].fillna(0).astype('float64')
    
    # STAGE 1: Classificação (0 vs >0)
    y_train_class = (y_train > 0).astype(int)
    y_test_class = (y_test > 0).astype(int)
    
    abc_models = {
        'classifiers': {},
        'regressors': {},
        'feature_names': features,
        'product_class_map': product_class_map
    }
    
    y_pred_train = np.zeros(len(y_train))
    y_pred_test = np.zeros(len(y_test))
    
    for classe in ['A', 'B', 'C']:
        if verbose:
            print(f"\n{'='*60}")
            print(f"🧠 Classe {classe}")
            print(f"{'='*60}")
        
        mask_train = (classes_train == classe)
        X_train_classe = X_train[mask_train]
        y_train_classe_class = y_train_class[mask_train]
        y_train_classe_reg = y_train[mask_train]
        
        if len(X_train_classe) == 0:
            if verbose:
                print(f"    ⚠️ Sem dados para Classe {classe}")
            continue
        
        # STAGE 1: CLASSIFICADOR
        if verbose:
            print(f"\n   🎯 STAGE 1: Classificador (Compra Sim/Não)")
        
        n_positivos = (y_train_classe_class == 1).sum()
        n_negativos = (y_train_classe_class == 0).sum()
        
        if verbose:
            print(f"      Positivos (compra): {n_positivos} ({n_positivos/len(y_train_classe_class)*100:.1f}%)")
            print(f"      Negativos (não compra): {n_negativos} ({n_negativos/len(y_train_classe_class)*100:.1f}%)")
        
        # ✅ AJUSTAR scale_pos_weight: Usar raiz quadrada
        scale_weight_raw = n_negativos / n_positivos if n_positivos > 0 else 1.0
        scale_weight = np.sqrt(scale_weight_raw)
        
        if verbose:
            print(f"      scale_pos_weight: {scale_weight:.2f} (raw: {scale_weight_raw:.2f})")
        
        classifier_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'scale_pos_weight': scale_weight,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'objective': 'binary:logistic',
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        
        classifier = xgb.XGBClassifier(**classifier_params)
        classifier.fit(X_train_classe, y_train_classe_class)
        
        y_pred_class_train = classifier.predict(X_train_classe)
        y_pred_proba_train = classifier.predict_proba(X_train_classe)[:, 1]
        
        acc = accuracy_score(y_train_classe_class, y_pred_class_train)
        prec = precision_score(y_train_classe_class, y_pred_class_train, zero_division=0)
        rec = recall_score(y_train_classe_class, y_pred_class_train, zero_division=0)
        f1 = f1_score(y_train_classe_class, y_pred_class_train, zero_division=0)
        
        if verbose:
            try:
                auc = roc_auc_score(y_train_classe_class, y_pred_proba_train)
                print(f"      ✅ Acc: {acc:.3f} | Prec: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
            except:
                print(f"      ✅ Acc: {acc:.3f} | Prec: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
        
        abc_models['classifiers'][classe] = classifier
        
        # STAGE 2: REGRESSOR
        if verbose:
            print(f"\n   📊 STAGE 2: Regressor (Quanto vai comprar?)")
        
        mask_positive = y_train_classe_reg > 0
        X_train_classe_reg = X_train_classe[mask_positive]
        y_train_classe_reg_positive = y_train_classe_reg[mask_positive]
        
        if len(X_train_classe_reg) == 0:
            if verbose:
                print(f"      ⚠️ Sem valores positivos para treinar regressor")
            abc_models['regressors'][classe] = None
            continue
        
        if verbose:
            print(f"      Amostras: {len(X_train_classe_reg):,}")
            print(f"      Média: {y_train_classe_reg_positive.mean():.1f} | Mediana: {y_train_classe_reg_positive.median():.1f}")
        
        # ✅ HIPERPARÂMETROS AJUSTADOS
        regressor_params_base = {
            'n_estimators': 200,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 20,
            'reg_alpha': 10.0,
            'reg_lambda': 10.0,
            'gamma': 5.0,
            'random_state': 42,
            'objective': 'reg:squarederror',
            'n_jobs': -1
        }
        
        # ✅ EARLY STOPPING
        if len(X_train_classe_reg) > 50:
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train_classe_reg, 
                y_train_classe_reg_positive,
                test_size=0.2,
                random_state=42
            )
            
            regressor = xgb.XGBRegressor(**regressor_params_base, early_stopping_rounds=20)
            regressor.fit(
                X_train_split, 
                y_train_split,
                eval_set=[(X_val_split, y_val_split)],
                verbose=False
            )
            
            best_n = regressor.best_iteration if hasattr(regressor, 'best_iteration') else regressor_params_base['n_estimators']
            
            regressor_params_final = regressor_params_base.copy()
            regressor_params_final['n_estimators'] = min(best_n + 10, regressor_params_base['n_estimators'])
            
            regressor = xgb.XGBRegressor(**regressor_params_final)
            regressor.fit(X_train_classe_reg, y_train_classe_reg_positive)
        else:
            regressor = xgb.XGBRegressor(**regressor_params_base)
            regressor.fit(X_train_classe_reg, y_train_classe_reg_positive)
        
        y_pred_reg_train = regressor.predict(X_train_classe_reg)
        y_pred_reg_train = np.maximum(y_pred_reg_train, 0)
        
        mae_reg = mean_absolute_error(y_train_classe_reg_positive, y_pred_reg_train)
        mape_reg = mape(y_train_classe_reg_positive, y_pred_reg_train)
        wmape_reg = wmape(y_train_classe_reg_positive, y_pred_reg_train)
        
        if verbose:
            print(f"      ✅ MAE: {mae_reg:.2f} | MAPE: {mape_reg:.4f} | WMAPE: {wmape_reg:.4f}")
        
        abc_models['regressors'][classe] = regressor
        
        # PREVISÃO COMBINADA COM THRESHOLD
        prob_compra_train = classifier.predict_proba(X_train_classe)[:, 1]
        pred_reg_train = regressor.predict(X_train_classe)
        pred_reg_train = np.maximum(pred_reg_train, 0)
        
        threshold = 0.65
        prob_compra_train_calibrada = np.where(prob_compra_train >= threshold, prob_compra_train, 0)
        
        pred_train_final = prob_compra_train_calibrada * pred_reg_train
        
        y_pred_train[mask_train] = pred_train_final
        
        # Teste
        mask_test = (classes_test == classe)
        if mask_test.sum() > 0:
            X_test_classe = X_test[mask_test]
            prob_compra_test = classifier.predict_proba(X_test_classe)[:, 1]
            pred_reg_test = regressor.predict(X_test_classe)
            pred_reg_test = np.maximum(pred_reg_test, 0)
            
            prob_compra_test_calibrada = np.where(prob_compra_test >= threshold, prob_compra_test, 0)
            
            pred_test_final = prob_compra_test_calibrada * pred_reg_test
            
            y_pred_test[mask_test] = pred_test_final
    
    y_pred_train = np.maximum(y_pred_train, 0)
    y_pred_test = np.maximum(y_pred_test, 0)
    
    # MÉTRICAS FINAIS
    if verbose:
        print("\n" + "="*80)
        print("📊 MÉTRICAS FINAIS TWO-STAGE")
        print("="*80)
        
        for classe in ['A', 'B', 'C']:
            mask_train = (classes_train == classe)
            mask_test = (classes_test == classe)
            
            if mask_train.sum() > 0:
                mae_train = mean_absolute_error(y_train[mask_train], y_pred_train[mask_train])
                mape_train = mape(y_train[mask_train], y_pred_train[mask_train])
                wmape_train = wmape(y_train[mask_train], y_pred_train[mask_train])
                print(f"    TREINO Classe {classe} - MAE: {mae_train:.2f} | MAPE: {mape_train:.4f} | WMAPE: {wmape_train:.4f}")
                
            if mask_test.sum() > 0:
                mae_test = mean_absolute_error(y_test[mask_test], y_pred_test[mask_test])
                mape_test = mape(y_test[mask_test], y_pred_test[mask_test])
                wmape_test = wmape(y_test[mask_test], y_pred_test[mask_test])
                print(f"    TESTE  Classe {classe} - MAE: {mae_test:.2f} | MAPE: {mape_test:.4f} | WMAPE: {wmape_test:.4f}")
        
        print("-" * 40)
        mae_geral_train = mean_absolute_error(y_train, y_pred_train)
        mae_geral_test = mean_absolute_error(y_test, y_pred_test)
        wmape_geral_train = wmape(y_train, y_pred_train)
        wmape_geral_test = wmape(y_test, y_pred_test)
        mape_geral_train = mape(y_train, y_pred_train)
        mape_geral_test = mape(y_test, y_pred_test)
        
        print(f"    GERAL (Treino) - MAE: {mae_geral_train:.2f} | MAPE: {mape_geral_train:.4f} | WMAPE: {wmape_geral_train:.4f}")
        print(f"    GERAL (Teste)  - MAE: {mae_geral_test:.2f} | MAPE: {mape_geral_test:.4f} | WMAPE: {wmape_geral_test:.4f}")
        
        print("\n" + "-" * 40)
        print("🔬 ANÁLISE DE OVERFITTING (Razão Teste/Treino):")
        for classe in ['A', 'B', 'C']:
            mask_train = (classes_train == classe)
            mask_test = (classes_test == classe)
            
            if mask_train.sum() > 0 and mask_test.sum() > 0:
                wmape_train_c = wmape(y_train[mask_train], y_pred_train[mask_train])
                wmape_test_c = wmape(y_test[mask_test], y_pred_test[mask_test])
                ratio = wmape_test_c / wmape_train_c if wmape_train_c > 0 else 0
                
                status = "✅" if ratio < 1.5 else "⚠️" if ratio < 3.0 else "❌"
                print(f"    {status} Classe {classe}: {ratio:.2f}x (Treino: {wmape_train_c:.2f} → Teste: {wmape_test_c:.2f})")
        
        ratio_geral = wmape_geral_test / wmape_geral_train if wmape_geral_train > 0 else 0
        status_geral = "✅" if ratio_geral < 1.5 else "⚠️" if ratio_geral < 3.0 else "❌"
        print(f"    {status_geral} GERAL: {ratio_geral:.2f}x (Treino: {wmape_geral_train:.2f} → Teste: {wmape_geral_test:.2f})")
        print("="*80)
    
    return (abc_models, y_train.values, y_pred_train, y_test.values, y_pred_test)


def predict_ensemble(ensemble_model, df_features):
    """Faz previsões usando o modelo Two-Stage."""
    
    features = ensemble_model['feature_names']
    X = df_features[features].fillna(0).astype('float64')
    
    predictions = np.zeros(len(df_features))
    
    for classe in ['A', 'B', 'C']:
        mask = (df_features['CLASSE_ABC'] == classe)
        if mask.sum() == 0:
            continue
        
        X_classe = X.loc[mask]
        
        classifier = ensemble_model['classifiers'].get(classe)
        regressor = ensemble_model['regressors'].get(classe)
        
        if classifier is None or regressor is None:
            predictions[mask.values] = 0.0
            continue
        
        prob_compra = classifier.predict_proba(X_classe)[:, 1]
        qtd_pred = regressor.predict(X_classe)
        qtd_pred = np.maximum(qtd_pred, 0)
        
        threshold = 0.65
        prob_compra_calibrada = np.where(prob_compra >= threshold, prob_compra, 0)
        
        pred_final = prob_compra_calibrada * qtd_pred
        
        predictions[mask.values] = pred_final
    
    return np.maximum(predictions, 0)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# VALIDAÇÃO TEMPORAL DINÂMICA
# ==================================================================================

def time_series_cross_validation(df_model_data, features, product_class_map, n_splits=5, min_train_months=24):
    """Validação cruzada temporal dinâmica."""
    
    print("\n" + "="*80)
    print("🔄 VALIDAÇÃO TEMPORAL DINÂMICA (TIME SERIES CV)")
    print("="*80)
    
    print(f"\n📋 Configuração:")
    print(f"   N° de folds: {n_splits}")
    print(f"   Mínimo de meses no treino inicial: {min_train_months}")
    
    meses_disponiveis = sorted(df_model_data['MES'].unique())
    n_meses_total = len(meses_disponiveis)
    
    print(f"\n📅 Dados Temporais:")
    print(f"   Período total: {meses_disponiveis[0].strftime('%Y-%m')} a {meses_disponiveis[-1].strftime('%Y-%m')}")
    print(f"   Total de meses: {n_meses_total}")
    
    meses_disponiveis_para_split = n_meses_total - min_train_months
    
    if meses_disponiveis_para_split < n_splits:
        print(f"\n⚠️  AVISO: Poucos meses disponíveis para {n_splits} folds")
        n_splits = max(2, meses_disponiveis_para_split)
        print(f"   Ajustando para {n_splits} folds")
    
    test_size = meses_disponiveis_para_split // n_splits
    if test_size < 1:
        test_size = 1
    
    print(f"\n🔢 Estratégia de Split:")
    print(f"   Tamanho do fold de teste: ~{test_size} meses")
    print(f"   Treino inicial: {min_train_months} meses")
    print(f"   Treino cresce incrementalmente a cada fold")
    
    splits = []
    for i in range(n_splits):
        train_end_idx = min_train_months + (i * test_size) - 1
        test_end_idx = min(train_end_idx + test_size, n_meses_total - 1)
        
        if train_end_idx >= n_meses_total - 1:
            break
        
        train_months = meses_disponiveis[:train_end_idx + 1]
        test_months = meses_disponiveis[train_end_idx + 1:test_end_idx + 1]
        
        if len(test_months) == 0:
            break
        
        splits.append({
            'fold': i + 1,
            'train_months': train_months,
            'test_months': test_months,
            'train_period': f"{train_months[0].strftime('%Y-%m')} a {train_months[-1].strftime('%Y-%m')}",
            'test_period': f"{test_months[0].strftime('%Y-%m')} a {test_months[-1].strftime('%Y-%m')}",
            'n_train_months': len(train_months),
            'n_test_months': len(test_months)
        })
    
    n_splits_real = len(splits)
    print(f"\n✅ {n_splits_real} folds criados com sucesso!")
    
    results = []
    all_models = []
    
    for split_info in splits:
        fold = split_info['fold']
        print(f"\n{'='*80}")
        print(f"🔄 FOLD {fold}/{n_splits_real}")
        print(f"{'='*80}")
        print(f"   📅 Treino:  {split_info['train_period']} ({split_info['n_train_months']} meses)")
        print(f"   📅 Teste:   {split_info['test_period']} ({split_info['n_test_months']} meses)")
        
        df_train_fold = df_model_data[df_model_data['MES'].isin(split_info['train_months'])].copy()
        df_test_fold = df_model_data[df_model_data['MES'].isin(split_info['test_months'])].copy()
        
        n_train = len(df_train_fold)
        n_test = len(df_test_fold)
        
        zeros_train = (df_train_fold['TARGET'] == 0).sum()
        zeros_test = (df_test_fold['TARGET'] == 0).sum()
        pct_zeros_train = zeros_train / n_train * 100 if n_train > 0 else 0
        pct_zeros_test = zeros_test / n_test * 100 if n_test > 0 else 0
        
        print(f"\n   📊 Distribuição:")
        print(f"      Treino: {n_train:,} linhas ({pct_zeros_train:.1f}% zeros)")
        print(f"      Teste:  {n_test:,} linhas ({pct_zeros_test:.1f}% zeros)")
        
        ratio_zeros = pct_zeros_test / pct_zeros_train if pct_zeros_train > 0 else 1.0
        if ratio_zeros > 1.5:
            print(f"      ⚠️  Data shift detectado: Teste tem {ratio_zeros:.2f}x mais zeros")
        elif ratio_zeros < 0.67:
            print(f"      ⚠️  Data shift detectado: Treino tem {1/ratio_zeros:.2f}x mais zeros")
        else:
            print(f"      ✅ Distribuição similar (razão: {ratio_zeros:.2f}x)")
        
        try:
            print(f"\n   🚀 Treinando modelo no fold {fold}...")
            
            ensemble_model, y_train_fold, y_pred_train_fold, y_test_fold, y_pred_test_fold = train_abc_two_stage_models(
                df_train_fold,
                df_test_fold,
                features,
                product_class_map,
                n_iter_search=20,
                verbose=False
            )
            
            all_models.append({
                'fold': fold,
                'model': ensemble_model,
                'train_period': split_info['train_period'],
                'test_period': split_info['test_period']
            })
            
            mae_train = mean_absolute_error(y_train_fold, y_pred_train_fold)
            mae_test = mean_absolute_error(y_test_fold, y_pred_test_fold)
            wmape_train = wmape(y_train_fold, y_pred_train_fold)
            wmape_test = wmape(y_test_fold, y_pred_test_fold)
            mape_train = mape(y_train_fold, y_pred_train_fold)
            mape_test = mape(y_test_fold, y_pred_test_fold)
            
            mask_zeros_train = (y_train_fold == 0)
            mask_zeros_test = (y_test_fold == 0)
            fp_train = (y_pred_train_fold[mask_zeros_train] > 10).sum() / mask_zeros_train.sum() * 100 if mask_zeros_train.sum() > 0 else 0
            fp_test = (y_pred_test_fold[mask_zeros_test] > 10).sum() / mask_zeros_test.sum() * 100 if mask_zeros_test.sum() > 0 else 0
            
            overfitting_ratio = wmape_test / wmape_train if wmape_train > 0 else 0
            
            results.append({
                'Fold': fold,
                'Train_Period': split_info['train_period'],
                'Test_Period': split_info['test_period'],
                'N_Train_Months': split_info['n_train_months'],
                'N_Test_Months': split_info['n_test_months'],
                'N_Train_Samples': n_train,
                'N_Test_Samples': n_test,
                'Pct_Zeros_Train': pct_zeros_train,
                'Pct_Zeros_Test': pct_zeros_test,
                'Ratio_Zeros': ratio_zeros,
                'MAE_Train': mae_train,
                'MAE_Test': mae_test,
                'WMAPE_Train': wmape_train,
                'WMAPE_Test': wmape_test,
                'MAPE_Train': mape_train,
                'MAPE_Test': mape_test,
                'FP_Train_Pct': fp_train,
                'FP_Test_Pct': fp_test,
                'Overfitting_Ratio': overfitting_ratio
            })
            
            print(f"\n   ✅ Fold {fold} concluído!")
            print(f"      WMAPE: Treino={wmape_train:.2f} | Teste={wmape_test:.2f} | Ratio={overfitting_ratio:.2f}x")
            
        except Exception as e:
            print(f"\n   ❌ Erro no fold {fold}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("\n❌ Nenhum fold foi executado com sucesso!")
        return None, None, None
    
    print("\n" + "="*80)
    print("📊 RESULTADOS DA VALIDAÇÃO TEMPORAL")
    print("="*80)
    
    print(f"\n📈 Métricas Médias (across {len(results_df)} folds):")
    print(f"   WMAPE Treino: {results_df['WMAPE_Train'].mean():.4f} ± {results_df['WMAPE_Train'].std():.4f}")
    print(f"   WMAPE Teste:  {results_df['WMAPE_Test'].mean():.4f} ± {results_df['WMAPE_Test'].std():.4f}")
    print(f"   MAE Treino:   {results_df['MAE_Train'].mean():.2f} ± {results_df['MAE_Train'].std():.2f}")
    print(f"   MAE Teste:    {results_df['MAE_Test'].mean():.2f} ± {results_df['MAE_Test'].std():.2f}")
    print(f"   FP Treino:    {results_df['FP_Train_Pct'].mean():.1f}% ± {results_df['FP_Train_Pct'].std():.1f}%")
    print(f"   FP Teste:     {results_df['FP_Test_Pct'].mean():.1f}% ± {results_df['FP_Test_Pct'].std():.1f}%")
    print(f"   Overfitting:  {results_df['Overfitting_Ratio'].mean():.2f}x ± {results_df['Overfitting_Ratio'].std():.2f}x")
    
    best_fold_idx = results_df['WMAPE_Test'].idxmin()
    best_fold = results_df.loc[best_fold_idx]
    
    print(f"\n🏆 MELHOR FOLD: Fold {int(best_fold['Fold'])}")
    print(f"   Teste: {best_fold['Test_Period']}")
    print(f"   WMAPE Teste: {best_fold['WMAPE_Test']:.4f}")
    print(f"   Overfitting: {best_fold['Overfitting_Ratio']:.2f}x")
    
    worst_fold_idx = results_df['WMAPE_Test'].idxmax()
    worst_fold = results_df.loc[worst_fold_idx]
    
    print(f"\n⚠️  PIOR FOLD: Fold {int(worst_fold['Fold'])}")
    print(f"   Teste: {worst_fold['Test_Period']}")
    print(f"   WMAPE Teste: {worst_fold['WMAPE_Test']:.4f}")
    print(f"   Overfitting: {worst_fold['Overfitting_Ratio']:.2f}x")
    print(f"   Possível causa: Ratio zeros = {worst_fold['Ratio_Zeros']:.2f}x")
    
    print(f"\n📉 ANÁLISE DE TENDÊNCIA TEMPORAL:")
    first_fold_wmape = results_df.iloc[0]['WMAPE_Test']
    last_fold_wmape = results_df.iloc[-1]['WMAPE_Test']
    
    if last_fold_wmape > first_fold_wmape * 1.2:
        print(f"   ⚠️  Performance PIORANDO ao longo do tempo")
        print(f"      Primeiro fold: {first_fold_wmape:.4f}")
        print(f"      Último fold: {last_fold_wmape:.4f}")
        print(f"      Degradação: {((last_fold_wmape/first_fold_wmape - 1)*100):.1f}%")
        print(f"   💡 Sugestão: Modelo pode estar desatualizado. Retreinar periodicamente.")
    elif last_fold_wmape < first_fold_wmape * 0.8:
        print(f"   ✅ Performance MELHORANDO ao longo do tempo")
        print(f"      Primeiro fold: {first_fold_wmape:.4f}")
        print(f"      Último fold: {last_fold_wmape:.4f}")
        print(f"      Melhoria: {((1 - last_fold_wmape/first_fold_wmape)*100):.1f}%")
    else:
        print(f"   ✅ Performance ESTÁVEL ao longo do tempo")
        print(f"      Variação: {((last_fold_wmape/first_fold_wmape - 1)*100):.1f}%")
    
    print(f"\n📋 TABELA RESUMO (ordenada por WMAPE Teste):")
    results_display = results_df[['Fold', 'Test_Period', 'WMAPE_Test', 'Overfitting_Ratio', 'Ratio_Zeros']].copy()
    results_display = results_display.sort_values('WMAPE_Test')
    results_display['WMAPE_Test'] = results_display['WMAPE_Test'].round(4)
    results_display['Overfitting_Ratio'] = results_display['Overfitting_Ratio'].round(2)
    results_display['Ratio_Zeros'] = results_display['Ratio_Zeros'].round(2)
    print(results_display.to_string(index=False))
    
    print("\n" + "="*80)
    
    return results_df, int(best_fold['Fold']), all_models

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# EXECUTAR VALIDAÇÃO TEMPORAL (OPCIONAL - COMENTAR SE NÃO QUISER)
# ==================================================================================

# Descomentar para executar validação temporal
"""
results_cv, best_fold_num, trained_models = time_series_cross_validation(
    df_model_data=df_model_data,
    features=features,
    product_class_map=product_class_map,
    n_splits=5,
    min_train_months=24
)

if results_cv is not None:
    print(f"\n💡 Insights da Validação Temporal:")
    print(f"   • WMAPE médio esperado: {results_cv['WMAPE_Test'].mean():.4f} ± {results_cv['WMAPE_Test'].std():.4f}")
    print(f"   • Overfitting médio: {results_cv['Overfitting_Ratio'].mean():.2f}x")
"""

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- 
# TREINAMENTO COM TWO-STAGE (MODELO FINAL)
# -------------------------------------------------------------------------------- 
ensemble_model, y_train, y_pred_train, y_test, y_pred_test = train_abc_two_stage_models(
    df_train,
    df_test,
    features,
    product_class_map,
    n_iter_search=30,
    verbose=True
)

print("\n" + "="*80)
print(f"✅ TREINAMENTO TWO-STAGE CONCLUÍDO")
print(f"Modelos disponíveis: {list(ensemble_model['classifiers'].keys())}")
print(f"Total de features utilizadas: {len(features)}")
print("="*80)

# ✅ DIAGNÓSTICO ADICIONAL: Análise de Predições
print("\n" + "="*80)
print("🔬 DIAGNÓSTICO: ANÁLISE DE PREDIÇÕES vs REAIS")
print("="*80)

for split_name, y_true_split, y_pred_split in [("TREINO", y_train, y_pred_train), ("TESTE", y_test, y_pred_test)]:
    print(f"\n📊 {split_name}:")
    
    mask_zeros = (y_true_split == 0)
    mask_nonzeros = (y_true_split > 0)
    
    if mask_zeros.sum() > 0:
        pred_em_zeros = y_pred_split[mask_zeros]
        fp_rate = (pred_em_zeros > 10).sum() / len(pred_em_zeros) * 100
        media_pred_zeros = pred_em_zeros.mean()
        print(f"   Zeros reais ({mask_zeros.sum():,} casos):")
        print(f"      Média predita: {media_pred_zeros:.1f}")
        print(f"      Falsos positivos (pred>10): {fp_rate:.1f}%")
    
    if mask_nonzeros.sum() > 0:
        y_true_nonzeros = y_true_split[mask_nonzeros]
        y_pred_nonzeros = y_pred_split[mask_nonzeros]
        
        mae_nonzeros = mean_absolute_error(y_true_nonzeros, y_pred_nonzeros)
        wmape_nonzeros = wmape(y_true_nonzeros, y_pred_nonzeros)
        
        erros = y_pred_nonzeros - y_true_nonzeros
        subestimacao = (erros < 0).sum() / len(erros) * 100
        superestimacao = (erros > 0).sum() / len(erros) * 100
        
        print(f"   Não-zeros reais ({mask_nonzeros.sum():,} casos):")
        print(f"      MAE: {mae_nonzeros:.1f} | WMAPE: {wmape_nonzeros:.2%}")
        print(f"      Subestimação: {subestimacao:.1f}% | Superestimação: {superestimacao:.1f}%")
        print(f"      Média real: {y_true_nonzeros.mean():.1f} | Média pred: {y_pred_nonzeros.mean():.1f}")

# ⚠️ AVISO SOBRE DATA SHIFT
print("\n" + "="*80)
print("⚠️  AVISO: POSSÍVEL DATA SHIFT TEMPORAL DETECTADO")
print("="*80)

pct_zeros_train = (y_train == 0).sum() / len(y_train) * 100
pct_zeros_test = (y_test == 0).sum() / len(y_test) * 100
ratio_shift = pct_zeros_test / pct_zeros_train

print(f"\nDistribuição de zeros (TARGET == 0):")
print(f"   TREINO: {pct_zeros_train:.1f}%")
print(f"   TESTE:  {pct_zeros_test:.1f}%")
print(f"   Razão:  {ratio_shift:.2f}x")

if ratio_shift > 1.5:
    print(f"\n❌ ALERTA: Teste tem {ratio_shift:.1f}x mais zeros que treino!")
    print("   Possíveis causas:")
    print("      1. Sazonalidade: Meses de teste são período de baixa demanda")
    print("      2. Split temporal enviesado: Últimos meses != primeiros meses")
    print("      3. Mudança de padrão: Comportamento mudou ao longo do tempo")
    print("\n   Recomendações:")
    print("      • Validar com meses aleatórios (não apenas últimos)")
    print("      • Adicionar features sazonais (trimestre, mês, etc.)")
    print("      • Considerar usar todo o histórico com CV temporal")
elif ratio_shift < 0.67:
    print(f"\n⚠️  AVISO: Treino tem mais zeros que teste (razão inversa)")
else:
    print(f"\n✅ Distribuição similar entre treino e teste")

print("="*80)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- 
# CRIAR DATAFRAMES DE PREDIÇÕES
# -------------------------------------------------------------------------------- 
print("\n📊 Criando DataFrames de predições...")

df_pred_train = df_train[['COMPONENT', 'COD_FORNE', 'MES', 'TARGET']].copy()
df_pred_train['PREDITO'] = y_pred_train
df_pred_train['TIPO'] = 'treino'
df_pred_train['CLASSE_ABC'] = df_pred_train['COMPONENT'].map(product_class_map).fillna('C')

df_pred_test = df_test[['COMPONENT', 'COD_FORNE', 'MES', 'TARGET']].copy()
df_pred_test['PREDITO'] = y_pred_test
df_pred_test['TIPO'] = 'teste'
df_pred_test['CLASSE_ABC'] = df_pred_test['COMPONENT'].map(product_class_map).fillna('C')

df_predictions_compras = pd.concat([df_pred_train, df_pred_test], ignore_index=True)
df_predictions_compras['PREDITO'] = np.ceil(df_predictions_compras['PREDITO'])

print(f"   ✅ Predições: {df_predictions_compras.shape}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# FUNÇÃO PARA FORECAST RECURSIVO
# ==================================================================================

def recursive_prediction(df_base, model, n_months, base_month, product_class_map):
    """Previsão recursiva multi-horizonte."""
    
    print(f"🔮 Previsão recursiva iniciada")
    print(f"    Base: {base_month}")
    print(f"    Horizonte: {n_months} meses")
    
    df_work = df_base.copy()
    df_work['MES'] = pd.to_datetime(df_work['MES'])
    
    all_predictions = []
    current_month = base_month
    
    for HORIZON in range(1, n_months + 1):
        target_month = current_month + pd.DateOffset(months=1)
        
        print(f"\n    📅 Horizonte {HORIZON}/{n_months}: Prevendo {target_month.strftime('%Y-%m')}")
        
        df_current = df_work[df_work['MES'] == current_month].copy()
        
        if len(df_current) == 0:
            print(f"         ⚠️  Sem dados para {current_month.strftime('%Y-%m')}")
            current_month = target_month
            continue
        
        df_current = df_current.dropna(subset=['HISTORICO_VENDAS_LAG12'])
        
        if len(df_current) == 0:
            print(f"         ⚠️  Sem histórico suficiente")
            current_month = target_month
            continue
        
        if 'CLASSE_ABC' not in df_current.columns:
            df_current['CLASSE_ABC'] = df_current['COMPONENT'].map(product_class_map).fillna('C')
        
        for feat in model['feature_names']:
            if feat not in df_current.columns:
                df_current[feat] = 0
        
        df_current[model['feature_names']] = df_current[model['feature_names']].fillna(0)
        
        try:
            predictions = predict_ensemble(model, df_current)
            predictions = np.maximum(predictions, 0)
            print(f"         ✅ {len(predictions):,} previsões | Volume: {predictions.sum():,.0f}")
        except Exception as e:
            print(f"         ❌ Erro: {e}")
            raise
        
        df_next = df_current.copy()
        df_next['MES'] = target_month
        
        # Shiftar LAGs
        for i in range(12, 1, -1):
            df_next[f'HISTORICO_VENDAS_LAG{i}'] = df_next[f'HISTORICO_VENDAS_LAG{i-1}']
        df_next['HISTORICO_VENDAS_LAG1'] = df_current['QUANTIDADE'].values
        
        df_next['QUANTIDADE'] = predictions
        
        # Recalcular features derivadas
        df_next['diff_1'] = df_next['QUANTIDADE'] - df_next['HISTORICO_VENDAS_LAG1']
        df_next['diff_2'] = df_next['HISTORICO_VENDAS_LAG1'] - df_next['HISTORICO_VENDAS_LAG2']
        df_next['diff_3'] = df_next['HISTORICO_VENDAS_LAG2'] - df_next['HISTORICO_VENDAS_LAG3']
        
        df_next['rolling_mean_3'] = df_next[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3']].mean(axis=1)
        df_next['rolling_mean_6'] = df_next[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3', 
                                              'HISTORICO_VENDAS_LAG4', 'HISTORICO_VENDAS_LAG5', 'HISTORICO_VENDAS_LAG6']].mean(axis=1)
        df_next['rolling_mean_12'] = df_next[[f'HISTORICO_VENDAS_LAG{i}' for i in range(1, 13)]].mean(axis=1)
        
        df_next['rolling_std_3'] = df_next[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3']].std(axis=1)
        df_next['rolling_std_6'] = df_next[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3', 
                                             'HISTORICO_VENDAS_LAG4', 'HISTORICO_VENDAS_LAG5', 'HISTORICO_VENDAS_LAG6']].std(axis=1)
        df_next['rolling_std_12'] = df_next[[f'HISTORICO_VENDAS_LAG{i}' for i in range(1, 13)]].std(axis=1)
        
        df_next['rolling_max_3'] = df_next[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3']].max(axis=1)
        df_next['rolling_max_6'] = df_next[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3', 
                                             'HISTORICO_VENDAS_LAG4', 'HISTORICO_VENDAS_LAG5', 'HISTORICO_VENDAS_LAG6']].max(axis=1)
        df_next['rolling_max_12'] = df_next[[f'HISTORICO_VENDAS_LAG{i}' for i in range(1, 13)]].max(axis=1)
        
        df_next['rolling_min_3'] = df_next[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3']].min(axis=1)
        df_next['rolling_min_6'] = df_next[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3', 
                                             'HISTORICO_VENDAS_LAG4', 'HISTORICO_VENDAS_LAG5', 'HISTORICO_VENDAS_LAG6']].min(axis=1)
        df_next['rolling_min_12'] = df_next[[f'HISTORICO_VENDAS_LAG{i}' for i in range(1, 13)]].min(axis=1)
        
        df_next['lead_time_lag_1'] = df_current['LEAD_TIME'].values
        df_next['quarter'] = df_next['MES'].dt.quarter
        df_next['quarter_start'] = df_next['MES'].dt.is_quarter_start.astype(int)
        df_next['quarter_end'] = df_next['MES'].dt.is_quarter_end.astype(int)
        
        target_month_num = target_month.month
        for m in range(1, 13):
            df_next[f'month_{m}'] = int(m == target_month_num)
        
        numeric_cols = df_next.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_next[col] = df_next[col].replace([np.inf, -np.inf], np.nan)
        
        rolling_cols = [col for col in df_next.columns if 'rolling_' in col]
        for col in rolling_cols:
            df_next[col] = df_next[col].fillna(0)
        
        df_work = pd.concat([df_work, df_next], ignore_index=True)
        
        df_pred = df_next[['COMPONENT', 'COD_FORNE', 'MES', 'QUANTIDADE', 'LEAD_TIME', 'CLASSE_ABC']].copy()
        all_predictions.append(df_pred)
        
        current_month = target_month
    
    if not all_predictions:
        print("\n⚠️  Nenhuma previsão gerada!")
        return pd.DataFrame()
    
    df_forecast = pd.concat(all_predictions, ignore_index=True)
    
    print(f"\n✅ Previsão concluída!")
    print(f"    Registros: {len(df_forecast):,}")
    print(f"    COMPONENTs: {df_forecast['COMPONENT'].nunique()}")
    print(f"    Volume total: {df_forecast['QUANTIDADE'].sum():,.0f}")
    
    return df_forecast

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# EXECUTAR PREVISÃO RECURSIVA (12 MESES)
# ==================================================================================
print("\n" + "="*80)
print("🔮 INICIANDO PREVISÃO RECURSIVA MULTI-HORIZONTE")
print("="*80)

n_HORIZONS = 12
base_date = df_final['MES'].max()

df_forecast_compras = recursive_prediction(
    df_base=df_final,
    model=ensemble_model,
    n_months=n_HORIZONS,
    base_month=base_date,
    product_class_map=product_class_map
)

print("\n✅ PREVISÃO CONCLUÍDA!")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def expandir_custo_por_mes(df, data_final):
    """Expande MES para cada COMPONENT + COD_FORNE até 'data_final'"""
    df = df.copy()
    df['MES'] = pd.to_datetime(df['MES'])
    data_final = pd.to_datetime(data_final)

    def expandir_grupo(g):
        g = g.sort_values('MES')
        g = g.drop_duplicates(subset='MES', keep='last')
        
        full_idx = pd.date_range(g['MES'].min(), data_final, freq='MS')
        g = g.set_index('MES').reindex(full_idx)
        g.index.name = 'MES'
        
        g['COMPONENT'] = g['COMPONENT'].ffill().bfill()
        g['COD_FORNE'] = g['COD_FORNE'].ffill().bfill()
        g['MOEDA'] = g['MOEDA'].ffill()
        g['PRECO_UNIT'] = g['PRECO_UNIT'].ffill()
        
        return g

    df_expanded = (
        df
        .groupby(['COMPONENT', 'COD_FORNE'], group_keys=False)
        .apply(expandir_grupo)
        .reset_index()
    )

    return df_expanded

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_forecast_compras['QUANTIDADE'] = np.ceil(df_forecast_compras['QUANTIDADE'])
max_month = df_forecast_compras['MES'].min() + pd.DateOffset(months=2)
df_hist_recent = df_historico_pedidos[['COMPONENT', 'COD_FORNE', 'MES', 'MOEDA', 'PRECO_UNIT']].sort_values(['COMPONENT', 'COD_FORNE', 'MES'], ascending=[True, True, False]).drop_duplicates(subset=['COMPONENT', 'COD_FORNE', 'MES'], keep='first')
df_hist_recent = expandir_custo_por_mes(
    df_historico_pedidos[['COMPONENT', 'COD_FORNE', 'MES', 'MOEDA', 'PRECO_UNIT']],
    data_final=max_month
)

df_forecast_compras = df_forecast_compras.merge(
    df_hist_recent,
    on=['COMPONENT', 'COD_FORNE', 'MES'],
    how='left'
)
df_forecast_compras = df_forecast_compras[~df_forecast_compras['PRECO_UNIT'].isna()] 
df_forecast_compras = df_forecast_compras[[*df_forecast_compras.drop(columns='CLASSE_ABC').columns, 'CLASSE_ABC']]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# COMPENSAÇÃO DE FATURAMENTO MÍNIMO
# ==================================================================================
def verificar_e_compensar_faturamento_minimo(df_forecast, df_faturamento_minimo, n_meses_output=3):
    """Compensa faturamento movendo COMPONENTES INTEIROS."""
    
    print("\n" + "="*80)
    print(f"💰 COMPENSAÇÃO INTELIGENTE - OUTPUT: {n_meses_output} MESES")
    print("="*80)
    
    df_work = df_forecast.copy()
    df_work['MES'] = pd.to_datetime(df_work['MES'])
    df_work['COD_FORNE'] = df_work['COD_FORNE'].astype(str)
    
    if 'PRECO_TOTAL' not in df_work.columns:
        df_work['PRECO_TOTAL'] = (df_work['QUANTIDADE'] * df_work['PRECO_UNIT']).round(2)
    
    meses_unicos = sorted(df_work['MES'].unique())
    meses_output = meses_unicos[:n_meses_output]
    
    # Identificar colunas faturamento
    col_map = {col.lower(): col for col in df_faturamento_minimo.columns}
    
    col_forn_fat = None
    for p in ['cod_forne', 'codforne', 'cod_fornecedor', 'fornecedor', 'supp_code']:
        if p in col_map:
            col_forn_fat = col_map[p]
            break
    
    col_fat = None
    for p in ['faturamento_minimo', 'faturamentominimo', 'faturamento', 'fatur.min.']:
        if p in col_map:
            col_fat = col_map[p]
            break
    
    if not col_forn_fat or not col_fat:
        print("⚠️ ERRO: Colunas não encontradas")
        return None, None
    
    df_fat = df_faturamento_minimo[[col_forn_fat, col_fat]].copy()
    df_fat = df_fat.rename(columns={col_forn_fat: 'COD_FORNE', col_fat: 'FATURAMENTO_MINIMO'})
    df_fat['COD_FORNE'] = df_fat['COD_FORNE'].astype(str)
    faturamento_map = df_fat.set_index('COD_FORNE')['FATURAMENTO_MINIMO'].to_dict()
    
    prioridade_abc = {'A': 1, 'B': 2, 'C': 3}
    
    print("\n📊 Aplicando compensação...")
    
    fornecedores = df_work['COD_FORNE'].unique()
    movimentos = []
    indices_para_remover = []
    
    for fornecedor in fornecedores:
        fat_min = faturamento_map.get(fornecedor, 0)
        if fat_min == 0:
            continue
        
        df_forn = df_work[df_work['COD_FORNE'] == fornecedor].copy()
        meses_forn = sorted(df_forn['MES'].unique())
        
        for mes_atual in meses_forn[:n_meses_output]:
            mask_atual = (df_work['COD_FORNE'] == fornecedor) & (df_work['MES'] == mes_atual)
            valor_atual = df_work.loc[mask_atual, 'PRECO_TOTAL'].sum()
            
            if valor_atual >= fat_min:
                continue
            
            deficit = fat_min - valor_atual
            meses_futuros = [m for m in meses_forn if m > mes_atual]
            
            for mes_futuro in meses_futuros:
                if deficit <= 0.01:
                    break
                
                mask_futuro = (df_work['COD_FORNE'] == fornecedor) & (df_work['MES'] == mes_futuro)
                df_futuro = df_work[mask_futuro].copy()
                
                if df_futuro.empty:
                    continue
                
                df_futuro['PRIORIDADE'] = df_futuro['CLASSE_ABC'].map(prioridade_abc).fillna(3)
                df_futuro = df_futuro.sort_values(['PRIORIDADE', 'PRECO_TOTAL'], ascending=[True, False])
                
                for idx, row in df_futuro.iterrows():
                    if deficit <= 0.01:
                        break
                    
                    qtd_total = row['QUANTIDADE']
                    preco_unit = row['PRECO_UNIT']
                    valor_total = row['PRECO_TOTAL']
                    
                    if valor_total <= deficit + 0.01:
                        qtd_mover = qtd_total
                        valor_mover = valor_total
                        indices_para_remover.append(idx)
                    else:
                        qtd_necessaria = deficit / preco_unit if preco_unit > 0 else 0
                        qtd_mover = np.ceil(qtd_necessaria)
                        qtd_mover = min(qtd_mover, qtd_total)
                        valor_mover = round(qtd_mover * preco_unit, 2)
                        
                        nova_qtd_futuro = qtd_total - qtd_mover
                        df_work.at[idx, 'QUANTIDADE'] = nova_qtd_futuro
                        df_work.at[idx, 'PRECO_TOTAL'] = round(nova_qtd_futuro * preco_unit, 2)
                    
                    mask_existe = (
                        (df_work['COD_FORNE'] == fornecedor) & 
                        (df_work['MES'] == mes_atual) & 
                        (df_work['COMPONENT'] == row['COMPONENT'])
                    )
                    
                    if mask_existe.any():
                        idx_existe = df_work[mask_existe].index[0]
                        nova_qtd = df_work.at[idx_existe, 'QUANTIDADE'] + qtd_mover
                        df_work.at[idx_existe, 'QUANTIDADE'] = nova_qtd
                        df_work.at[idx_existe, 'PRECO_TOTAL'] = round(nova_qtd * preco_unit, 2)
                    else:
                        nova_linha = {
                            'COMPONENT': row['COMPONENT'],
                            'COD_FORNE': fornecedor,
                            'MES': mes_atual,
                            'QUANTIDADE': qtd_mover,
                            'LEAD_TIME': row['LEAD_TIME'],
                            'MOEDA': row['MOEDA'],
                            'PRECO_UNIT': preco_unit,
                            'PRECO_TOTAL': valor_mover,
                            'CLASSE_ABC': row['CLASSE_ABC']
                        }
                        df_work = pd.concat([df_work, pd.DataFrame([nova_linha])], ignore_index=True)
                    
                    movimentos.append({
                        'COD_FORNE': fornecedor,
                        'COMPONENT': row['COMPONENT'],
                        'MES_ORIGEM': pd.to_datetime(mes_futuro).strftime('%Y-%m'),
                        'MES_DESTINO': pd.to_datetime(mes_atual).strftime('%Y-%m'),
                        'QUANTIDADE_MOVIDA': qtd_mover,
                        'VALOR_MOVIDO': valor_mover,
                        'CLASSE_ABC': row['CLASSE_ABC'],
                        'TIPO': 'INTEIRO' if qtd_mover == qtd_total else 'PARCIAL'
                    })
                    
                    deficit -= valor_mover
    
    if indices_para_remover:
        df_work = df_work.drop(index=indices_para_remover).reset_index(drop=True)
    
    df_work = df_work[df_work['QUANTIDADE'] > 0.01].copy()
    
    print(f"\n🔢 Arredondando quantidades...")
    df_work['QUANTIDADE'] = np.ceil(df_work['QUANTIDADE'])
    df_work['PRECO_TOTAL'] = (df_work['QUANTIDADE'] * df_work['PRECO_UNIT']).round(2)
    
    print(f"\n✂️  Filtrando {n_meses_output} primeiros meses...")
    df_output = df_work[df_work['MES'].isin(meses_output)].copy()
    
    print(f"\n📊 Agregando...")
    df_output_final = df_output.groupby(
        ['COMPONENT', 'COD_FORNE', 'MES'], 
        as_index=False
    ).agg({
        'QUANTIDADE': 'sum',
        'PRECO_UNIT': 'first',
        'LEAD_TIME': 'first',
        'MOEDA': 'first',
        'CLASSE_ABC': 'first'
    })
    
    df_output_final['QUANTIDADE'] = np.ceil(df_output_final['QUANTIDADE'])
    df_output_final['PRECO_TOTAL'] = (df_output_final['QUANTIDADE'] * df_output_final['PRECO_UNIT']).round(2)
    
    df_output_final = df_output_final[[
        'COMPONENT', 'COD_FORNE', 'MES', 'QUANTIDADE', 
        'LEAD_TIME', 'MOEDA', 'PRECO_TOTAL', 'CLASSE_ABC'
    ]]
    
    df_output_final = df_output_final.sort_values(['COD_FORNE', 'MES', 'COMPONENT'])
    
    print(f"   ✅ Linhas finais: {len(df_output_final):,}")
    print(f"\n✅ Compensação concluída! Movimentos: {len(movimentos)}")
    
    return None, df_output_final

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_faturamento_minimo = Helpers.getEntityData(context, 'faturamento_minimo').rename(columns={
    'Faturamento_M_nimo': 'FATURAMENTO_MINIMO',
    'cod._Fornecedor': 'COD_FORNE',
}).drop(['Fornecedor'], axis=1)

df_analise_faturamento, df_forecast_compras = verificar_e_compensar_faturamento_minimo(
    df_faturamento_minimo=df_faturamento_minimo,
    df_forecast=df_forecast_compras
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Adicionando os resultados da predicao
df_model_data = df_model_data[['COMPONENT', 'COD_FORNE', 'MES', 'LEAD_TIME', 'TARGET', 'QUANTIDADE']]
df_model_data = pd.merge(
    df_model_data, 
    df_predictions_compras[['COMPONENT', 'COD_FORNE', 'MES', 'PREDITO', 'CLASSE_ABC']], 
    on=['COMPONENT', 'COD_FORNE', 'MES'],
    how='left'
    ).rename(columns = {
    'TARGET':'QUANT_TARGET',
    'QUANTIDADE':'QUANT_REALIZADA',
    'PREDITO':'QUANT_PREDITA'
    })

df_model_data = pd.merge(
    df_model_data, 
    df_hist_recent[['COMPONENT', 'COD_FORNE', 'MES', 'MOEDA', 'PRECO_UNIT']], 
    on=['COMPONENT', 'COD_FORNE', 'MES'],
    how='left'
    )

df_model_data['PRECO_TARGET'] = df_model_data['QUANT_TARGET'] * df_model_data['PRECO_UNIT']
df_model_data['PRECO_REALIZADO'] = df_model_data['QUANT_REALIZADA'] * df_model_data['PRECO_UNIT']
df_model_data['PRECO_PREDITO'] = df_model_data['QUANT_PREDITA'] * df_model_data['PRECO_UNIT']

df_forecast_compras = df_forecast_compras.rename(columns = {
    'QUANTIDADE':'QUANT_PREDITA',
    'PRECO_TOTAL':'PRECO_PREDITO'
})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# SALVAR RESULTADOS
# ==================================================================================
print("\n" + "="*80)
print("💾 SALVANDO RESULTADOS")
print("="*80)

Helpers.save_output_dataset(
    context=context,
    output_name='df_historico_pedidos_realizados',
    data_frame=df_model_data
)
print("    ✅ df_historico_pedidos_realizados")

Helpers.save_output_dataset(
    context=context,
    output_name='df_historico_pedidos_previstos',
    data_frame=df_forecast_compras
)
print("    ✅ df_historico_pedidos_previstos")

print("\n" + "="*80)
print("✅ PROCESSAMENTO COMPLETO!")
print("="*80)