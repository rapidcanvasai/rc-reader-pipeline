"""
supplier_history_view Recipe
============================
Sanitiza e enriquece datasets para SupplierHistoryPage do frontend.

Inputs:
  - main (MAIN_DATA): Dados principais de componentes/fornecedores (37 cols)
  - df_historico_pedidos: Histórico de pedidos (9 cols)
  - lt_rp: Lead time por componente (4 cols, 1251 rows)

Outputs (3):
  - supplier_lead_time: Fornecedores únicos com lead time (21 rows)
  - order_history: Histórico + nome fornecedor (5065 rows)
  - supplier_group_history: Histórico agregado por fornecedor, grupo e mês

Note: Os 21 fornecedores únicos são idênticos em main.Supp_Cod e lt_rp.Supplier
      Cada fornecedor tem apenas 1 valor de RP/LT (componentes repetidos removidos)

Author: MTE Data Engineering Team
Date: 2024
Version: 2.1 (Added supplier_group_history)
"""

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
import pandas as pd

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Load datasets
df_main = Helpers.getEntityData(context, 'main')
df_historico = Helpers.getEntityData(context, 'df_historico_pedidos_lead_time')
df_lead_time = Helpers.getEntityData(context, 'lt_rp')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Utility functions
def remove_decimal_zero(series: pd.Series) -> pd.Series:
    """Remove .0 suffix from numeric strings (2880.0 → 2880)"""
    return series.astype(str).str.replace(r'\.0$', '', regex=True)

def sanitize_supplier_code(series: pd.Series) -> pd.Series:
    """Standardize supplier codes to string format without .0 suffix"""
    return series.fillna(0).astype('Int64').astype(str).str.replace(r'\.0$', '', regex=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 1. CREATE supplier_lead_time
df_supplier_lt = df_lead_time.groupby('Supplier').agg({
    'Review_Period': 'first',
    'Lead_Time': 'first',
    'Calculated_Lead_Time': 'first'
}).reset_index()

df_supplier_lt['Supplier'] = df_supplier_lt['Supplier'].astype(str).str.replace(r'\.0$', '', regex=True)
df_supplier_lt['Calculated_Lead_Time'] = df_supplier_lt['Calculated_Lead_Time'].round(0).astype('Int64')

df_names = df_main[['Supp_Cod', 'Supplier']].copy()
df_names['Supp_Cod'] = remove_decimal_zero(df_names['Supp_Cod'])
df_names = df_names.drop_duplicates(subset=['Supp_Cod'])
df_names = df_names[df_names['Supp_Cod'] != 'nan']
df_names = df_names[df_names['Supp_Cod'] != '']

df_supplier_lead_time = df_supplier_lt.merge(
    df_names.rename(columns={'Supp_Cod': 'Supp_Code', 'Supplier': 'Supplier_Name'}),
    left_on='Supplier',
    right_on='Supp_Code',
    how='left'
).drop(columns=['Supp_Code'])

df_supplier_lead_time = df_supplier_lead_time[[
    'Supplier', 'Supplier_Name', 'Review_Period', 'Lead_Time', 'Calculated_Lead_Time'
]]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 2. CREATE order_history
df_hist = df_historico.copy()
df_hist['COD_FORNE'] = sanitize_supplier_code(df_hist['COD_FORNE'])

df_hist = df_hist.merge(
    df_supplier_lead_time[['Supplier', 'Supplier_Name']].rename(
        columns={'Supplier_Name': 'NOME_FORNE'}
    ),
    left_on='COD_FORNE',
    right_on='Supplier',
    how='left'
).drop(columns=['Supplier'])

cols_order = ['PEDIDO', 'COD_FORNE', 'NOME_FORNE', 'PRODUTO', 'DATA_SI',
              'ENTREGA_EFET', 'QUANTIDADE', 'MOEDA', 'PRECO_TOTAL', 'LEAD_TIME_DIAS']
order_history = df_hist[[c for c in cols_order if c in df_hist.columns]]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 3. CREATE supplier_group_history
# Agregar histórico por fornecedor, grupo de produto e mês

# Get Component -> Group_name mapping from main
df_component_group = df_main[['Component', 'Group_name']].copy()
df_component_group = df_component_group.drop_duplicates(subset=['Component'])

# Merge order_history with component group
df_group_hist = order_history.merge(
    df_component_group,
    left_on='PRODUTO',
    right_on='Component',
    how='left'
).drop(columns=['Component'])

# Convert DATA_SI to month period for aggregation
df_group_hist['MES'] = pd.to_datetime(df_group_hist['DATA_SI']).dt.to_period('M').astype(str)

# Aggregate by supplier, group and month
supplier_group_history = df_group_hist.groupby(
    ['COD_FORNE', 'NOME_FORNE', 'Group_name', 'MES'], as_index=False
).agg({
    'QUANTIDADE': 'sum',
    'PRECO_TOTAL': 'sum'
})

# Remove rows without Group_name (products not found in main)
supplier_group_history = supplier_group_history[supplier_group_history['Group_name'].notna()]

# Sort by supplier, group and month
supplier_group_history = supplier_group_history.sort_values(
    ['COD_FORNE', 'Group_name', 'MES']
).reset_index(drop=True)

print(f"supplier_group_history: {supplier_group_history.shape}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Save outputs
Helpers.save_output_dataset(context=context, output_name='supplier_lead_time', data_frame=df_supplier_lead_time)
Helpers.save_output_dataset(context=context, output_name='order_history', data_frame=order_history)
Helpers.save_output_dataset(context=context, output_name='supplier_group_history', data_frame=supplier_group_history)