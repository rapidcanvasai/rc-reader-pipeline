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
from utils.libutils.vectorStores.utils import VectorStoreUtils

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ================================================================================
# RECIPE UNIFICADO: prepare_product_view_data
# ================================================================================
# MÓDULOS:
#     1. ENRIQUECIMENTO DE FORECAST DE COMPONENTES (Gera: component_forecast)
#     2. CONSOLIDAÇÃO DE HISTÓRICO DE COMPONENTES (Gera: component_history)
#     3. EXPLOSÃO DE CONSUMO POR PRODUTO (Gera: generate_component_consumption_by_products)
#
# DEPENDÊNCIAS GERAIS:
#     - new_multihorizon_components (Forecast Componentes)
#     - produtos (Cadastro Mestre)
#     - df_produto_fornecedor (Custos/Fornecedores)
#     - new_monthly_portalvendas_components (Histórico)
#     - new_multihorizon_products_abcxyz (Forecast Produtos Finais)
#     - df_estrutura_produto (BOM - Bill of Materials)
# ================================================================================

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 1. SETUP E IMPORTS GERAIS
# --------------------------------------------------------------------------------

import pandas as pd
import numpy as np

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 2. CARREGAMENTO DE TODOS OS INPUTS (INGESTÃO CENTRALIZADA)
# --------------------------------------------------------------------------------
# Inputs Bloco 1 (Forecast Comp)
df_new_multihorizon_components = Helpers.getEntityData(context, 'new_multihorizon_components')
df_produtos = Helpers.getEntityData(context, 'df_produtos')
df_produto_fornecedor = Helpers.getEntityData(context, 'df_produto_fornecedor')

# Inputs Bloco 2 (Histórico)
df_history_raw = Helpers.getEntityData(context, 'new_monthly_portalvendas_components')

# Inputs Bloco 3 (BOM Explosion)
df_forecast_prod_raw = Helpers.getEntityData(context, 'new_multihorizon_products_abcxyz')
df_estrutura = Helpers.getEntityData(context, 'df_estrutura_produto')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ================================================================================
# BLOCO 1: FORECAST DE COMPONENTES (ENRIQUECIMENTO)
# ================================================================================
# 1.1 Merge com produtos
df_fc_enriched = df_new_multihorizon_components.merge(
    df_produtos[[
        "B1_COD", "B1_DESC", "B1_GRUPO", "NOM_GRUP", "ORIGEM", "CURVA",
        "PRECO_VENDA_BRL", "PRECO_VENDA_USD", "PRECO_VENDA_EUR",
        "MULTIPLO_COMPRA", "KANBAN_MIN", "KANBAN_MAX", "B1_ATIVO", "DTA_CADASTRO"
    ]],
    left_on="Component",
    right_on="B1_COD",
    how="left"
)

# 1.2 Merge com fornecedor
df_fc_enriched = df_fc_enriched.merge(
    df_produto_fornecedor[[
        "PRODUTO", "COD_FORNE", "FORNEC_NOM", "COD_FABRI", "MOEDA", "CUSTO_PRODUTO"
    ]],
    left_on="Component",
    right_on="PRODUTO",
    how="left"
)

# 1.3 Limpeza
df_fc_enriched["COD_FORNE"] = df_fc_enriched["COD_FORNE"].fillna("MTE")
df_fc_enriched["FORNEC_NOM"] = df_fc_enriched["FORNEC_NOM"].fillna("MTE")
df_fc_enriched = df_fc_enriched.drop(columns=["B1_COD", "PRODUTO"], errors="ignore")

# 1.4 Conversão e Arredondamento
df_fc_enriched["DATA_PEDIDO"] = pd.to_datetime(df_fc_enriched["DATA_PEDIDO"])
df_fc_enriched["base_date"] = pd.to_datetime(df_fc_enriched["base_date"])

# Lógica Inteira: Fillna -> Round -> Int
df_fc_enriched["consumption_predicted_month"] = (
    df_fc_enriched["consumption_predicted_month"]
    .astype("float64")
    .fillna(0)
    .round(0)
    .astype(int)
)

# 1.5 Colunas Auxiliares
df_fc_enriched["forecast_horizon_months"] = (
    (df_fc_enriched["DATA_PEDIDO"].dt.year - df_fc_enriched["base_date"].dt.year) * 12 +
    (df_fc_enriched["DATA_PEDIDO"].dt.month - df_fc_enriched["base_date"].dt.month)
)
df_fc_enriched["forecast_month_str"] = df_fc_enriched["DATA_PEDIDO"].dt.strftime("%Y-%m")
df_fc_enriched["base_month_str"] = df_fc_enriched["base_date"].dt.strftime("%Y-%m")

# Otimização categórica
categ_cols_1 = ["Component", "B1_GRUPO", "NOM_GRUP", "ORIGEM", "CURVA", "B1_ATIVO", "COD_FORNE", "FORNEC_NOM", "MOEDA"]
for col in categ_cols_1:
    if col in df_fc_enriched.columns:
        df_fc_enriched[col] = df_fc_enriched[col].astype("category")

# 1.6 Output Final Bloco 1
df_output_1 = df_fc_enriched.sort_values(["Component", "base_date", "DATA_PEDIDO"], ignore_index=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ================================================================================
# BLOCO 2: HISTÓRICO DE CONSUMO (CONSOLIDAÇÃO)
# ================================================================================
# 2.1 Cópia e Conversão
df_hist_proc = df_history_raw.copy()
df_hist_proc["MONTH"] = pd.to_datetime(df_hist_proc["MONTH"])
df_hist_proc["Consumption"] = pd.to_numeric(df_hist_proc["Consumption"], errors='coerce').fillna(0)

# 2.2 Agregação (Soma por Componente/Mês)
df_hist_grouped = df_hist_proc.groupby(["Component", "MONTH"], as_index=False)["Consumption"].sum()

# 2.3 Tipagem
df_hist_grouped["Component"] = df_hist_grouped["Component"].astype("category")
df_hist_grouped["Consumption"] = df_hist_grouped["Consumption"].astype("float64")

# 2.4 Filtrar apenas os últimos 2 anos de histórico
data_corte = pd.Timestamp.now() - pd.DateOffset(years=2)
df_hist_grouped = df_hist_grouped[df_hist_grouped['MONTH'] >= data_corte]

# 2.5 Output Final Bloco 2
df_output_2 = df_hist_grouped.sort_values(["Component", "MONTH"], ignore_index=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ================================================================================
# BLOCO 3: EXPLOSÃO DE CONSUMO POR PRODUTO (BOM EXPLOSION)
# ================================================================================
# 3.1 Padronização e Limpeza
df_bom = df_estrutura.rename(columns={"COD_PRODUTO": "Codigo", "COD_COMPONENTE": "Componente", "G1_QUANT": "Quantidade"})
df_bom["Codigo"] = df_bom["Codigo"].astype(str)
df_bom["Componente"] = df_bom["Componente"].astype(str)
df_bom["Quantidade"] = pd.to_numeric(df_bom["Quantidade"], errors='coerce')

df_prod_fc = df_forecast_prod_raw.copy()
df_prod_fc["COD_MTE_COMP"] = df_prod_fc["COD_MTE_COMP"].astype(str)
df_prod_fc["DATA_PEDIDO"] = pd.to_datetime(df_prod_fc["DATA_PEDIDO"])
df_prod_fc["base_date"] = pd.to_datetime(df_prod_fc["base_date"])
df_prod_fc["QTDE_PEDIDA"] = pd.to_numeric(df_prod_fc["QTDE_PEDIDA"], errors='coerce')

# 3.2 Merge (Inner Join)
df_merged_bom = df_prod_fc.merge(df_bom, left_on="COD_MTE_COMP", right_on="Codigo", how="inner")

# 3.3 Cálculo de Consumo
df_merged_bom["ComponentConsumption"] = (df_merged_bom["QTDE_PEDIDA"] * df_merged_bom["Quantidade"]).fillna(0)
df_merged_bom.loc[df_merged_bom["ComponentConsumption"] < 0, "ComponentConsumption"] = 0

# 3.4 Agregação
df_bom_agg = df_merged_bom.groupby(
    ["Componente", "base_date", "Codigo", "DATA_PEDIDO"], as_index=False
).agg({
    "QTDE_PEDIDA": "sum", 
    "ComponentConsumption": "sum", 
    "Quantidade": "first"
})

# 3.5 Renomeação e Conversão Final (INTEIROS)
df_bom_agg = df_bom_agg.rename(columns={
    "Componente": "Component", "Codigo": "ProductCode",
    "QTDE_PEDIDA": "ForecastSales", "Quantidade": "QuantityPerUnit"
})

df_bom_agg["base_date"] = pd.to_datetime(df_bom_agg["base_date"])
df_bom_agg["DATA_PEDIDO"] = pd.to_datetime(df_bom_agg["DATA_PEDIDO"])
df_bom_agg["Component"] = df_bom_agg["Component"].astype("category")
df_bom_agg["ProductCode"] = df_bom_agg["ProductCode"].astype("category")

# Arredondamento para Inteiros (Sales, Consumption, Quantity)
cols_int = ["ForecastSales", "ComponentConsumption", "QuantityPerUnit"]
for col in cols_int:
    df_bom_agg[col] = df_bom_agg[col].round(0).astype("int64")

# 3.6 Output Final Bloco 3
df_output_3 = df_bom_agg.sort_values(["Component", "base_date", "ProductCode", "DATA_PEDIDO"], ignore_index=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ================================================================================
# 4. SALVAMENTO DOS OUTPUTS (PERSISTÊNCIA CENTRALIZADA)
# ================================================================================

# Output 1: Component Forecast
Helpers.save_output_dataset(
    context=context,
    output_name='component_forecast',
    data_frame=df_output_1
)

# Output 2: Component History
Helpers.save_output_dataset(
    context=context,
    output_name='component_history',
    data_frame=df_output_2
)

# Output 3: Component Consumption by Products
Helpers.save_output_dataset(
    context=context,
    output_name='components_consumption_by_products',
    data_frame=df_output_3
)