# ================================================================================
  # RECIPE: ruptura Recipe
  # ================================================================================
  # DEPENDÊNCIAS: Este recipe requer o seguinte dataset:
  #     1. df_main (sugestões de pedido com estoque, vendas e trânsito)
  # ================================================================================
  #
  # PROPÓSITO: Analisar risco de ruptura de estoque projetando cobertura para os
  #            próximos 4 meses, calculando pedidos emergenciais para 7 meses de
  #            cobertura e gerando flags de alerta por severidade de risco.
  #
  # INPUTS:
  #   - df_main (recomendações de pedido com métricas de estoque e vendas)
  #
  # OUTPUT:
  #   - df_ruptura (análise de risco de ruptura com projeções e flags de alerta)
  #
  # FILTROS APLICADOS:
  #   1. Últimas 4 colunas Transit_ (transit_cols[-4:]): Considera apenas os 4 meses mais recentes
  #      * Consequência: Chegadas em trânsito além de 4 meses não são consideradas na projeção
  #
  #   2. Venda_mensal > 0 para cálculo de cobertura: Componentes sem vendas recebem valor 999 (infinito)
  #      * Consequência: Itens sem movimentação não geram alertas de ruptura
  #
  #   3. Colunas Ruptura_ sem sufixo "Flag": Filtra apenas valores de dias de cobertura
  #      * Consequência: Flags de texto são processadas separadamente das métricas numéricas
  #
  #   4. Colunas numéricas (select_dtypes number/Float64): Converte apenas colunas numéricas
  #      * Consequência: Colunas de texto/string não são afetadas por conversões numéricas
  #
  # LÓGICA:
  #   Para cada componente:
  #     1. Calcula Participacao = (Sales_12M / Total_Sales_12M) × 100
  #     2. Calcula Final_order_ruptura = CEILING(MAX(0, (7 × Venda_mensal) - Total_Stock) / 10) × 10
  #     3. Projeta cobertura para 4 meses:
  #        - Mês i: Dias_cobertura = ((Total_Stock + Σ Transit até mês i) / (Venda_mensal × i)) × 30
  #     4. Gera flags de severidade por dias de cobertura:
  #        - < 30 dias: "Estoque até 30 dias" (Crítico)
  #        - < 60 dias: "Estoque até 60 dias" (Aviso)
  #        - ≤ 90 dias: "Estoque até 90 dias" (Atenção)
  #        - < 120 dias: "Estoque até 120 dias" (Monitorar)
  #        - ≥ 120 dias: "Cobertura maior que 4 meses" (Seguro)
  #     5. Reorganiza colunas por categoria (info, alcance, métricas, vendas, ruptura)
  #
  # EXEMPLO:
  #   Componente X: Total_Stock=500, Venda_mensal=100, Transit_M1=200
  #   → Final_order_ruptura = CEILING((7×100 - 500) / 10) × 10 = 200
  #   → Ruptura_M1 = ((500 + 200) / (100×1)) × 30 = 210 dias
  #   → Flag: "Cobertura maior que 4 meses"
  # ================================================================================


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
import numpy as np

# Mostrar todas as linhas
pd.set_option("display.max_rows", None)

# Mostrar todas as colunas
pd.set_option("display.max_columns", None)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_ruptura = Helpers.getEntityData(context, 'df_main')

# Garante que colunas de cálculo sejam numéricas
df_ruptura['Venda_mensal'] = pd.to_numeric(df_ruptura['Venda_mensal'], errors='coerce').fillna(0)
df_ruptura['Total_Stock'] = pd.to_numeric(df_ruptura['Total_Stock'], errors='coerce').fillna(0)
df_ruptura['Sales_12M'] = pd.to_numeric(df_ruptura['Sales_12M'], errors='coerce').fillna(0)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_ruptura['Participacao'] = (df_ruptura['Sales_12M']/df_ruptura['Sales_12M'].fillna(0).sum()*100).round(2)

df_ruptura['Alcance_-_Estoque_total_+_Novo_pedido_(Model)'] = (
    (df_ruptura["Total_Stock"].fillna(0) +  df_ruptura['Final_order'].fillna(0)) /
    (df_ruptura["Venda_mensal"].replace(0, np.nan))
).round(0).fillna(0).astype(int)

df_ruptura['Alcance_-_Estoque_total_+_Novo_pedido_(Baseline)'] = (
    (df_ruptura["Total_Stock"].fillna(0) +  df_ruptura['Final_order_baseline'].fillna(0)) /
    (df_ruptura["Venda_mensal"].replace(0, np.nan))
).round(0).fillna(0).astype(int)

df_ruptura['Final_order_ruptura'] = np.ceil(np.maximum(0, (7 * df_ruptura['Venda_mensal']) - df_ruptura['Total_Stock']) / 10) * 10
df_ruptura['Final_order_ruptura'] = df_ruptura['Final_order_ruptura'].fillna(0)

df_ruptura['Nova_cobertura_Ruptura'] = np.rint((df_ruptura['Total_Stock'] + df_ruptura['Final_order_ruptura'])/df_ruptura['Venda_mensal'])
df_ruptura['Nova_cobertura_Ruptura'] = df_ruptura['Nova_cobertura_Ruptura'].astype(float).fillna(0)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
transit_cols = [col for col in df_ruptura.columns if col.startswith('Transit_')]
transit_cols = transit_cols[-4:] # Pega as 4 últimas

ruptura_cols = ['Ruptura_' + col.split('_')[1] for col in transit_cols]
cumsum_transit = pd.Series(0, index=df_ruptura.index)

for i, (t_col, r_col) in enumerate(zip(transit_cols, ruptura_cols), start=1):
    cumsum_transit += df_ruptura[t_col].fillna(0)
    
    venda_mensal_x_i = df_ruptura['Venda_mensal'] * i
    df_ruptura[r_col] = np.where(
        venda_mensal_x_i > 0,
        np.rint(((df_ruptura['Total_Stock'] + cumsum_transit) / venda_mensal_x_i) * 30),
        999 # Se venda é 0, cobertura é "infinita"
    )
    df_ruptura[r_col] = df_ruptura[r_col].apply(pd.to_numeric, errors="coerce").round(0)

num_cols = df_ruptura.select_dtypes(include=['number', 'Float64']).columns
df_ruptura[num_cols] = df_ruptura[num_cols].astype(float).fillna(0)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
ruptura_cols = [col for col in df_ruptura.columns if col.startswith('Ruptura_')]
for r_col in ruptura_cols:
    flag_col = r_col.replace('Ruptura_', 'Ruptura_Flag_') # Nome da coluna de flag
    
    col_values = df_ruptura[r_col].fillna(9999).astype(float)
    
    conditions = [
        col_values < 30,
        col_values < 60,
        col_values <= 90,
        col_values < 120
    ]
    choices = [
        "🚫 Estoque até 30 dias",
        "⚠️ Estoque até 60 dias",
        "📦 Estoque até 90 dias",
        "🛒 Estoque até 120 dias"
    ]
    
    df_ruptura[flag_col] = np.select(conditions, choices, default="✅ Cobertura maior que 4 meses")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
main_info = ['Component', 'Cod_X', 'Description', 'Group_code', 'Group_name', 
             'Supp_Cod', 'Supplier', 'ABC', 'Participacao', 'New_ABC', 'check_abc']

metrics = ['Stock', 'Transit', 'Inspection', 'Reserved', 'Total_Stock', 'Sales_12M',
           'Venda_mensal', 'Cobertura', 
           'Final_order', 'Final_order_baseline', 'Final_order_ruptura', 
           'Nova_cobertura_Ruptura',
           'Unfulfilled_12M', 'KanBan_Min', 'KanBan_Max', 
           'Alert', 'Obs', '%_Export_12M', 'Safety_stock', 'RP', 'LT', 
           'LT+RP', 'Inventory_level', 'Demand_(LT+RP)', 'On_demand', 'Origin', 'Cost', 
           'Currency', 'Multiplier', 'Total_Cost']

sales_cols = [c for c in df_ruptura.columns if c.startswith('Sales-M')]
alcance_cols = ['Alcance_-_Estoque_Atual', 'Alcance_-_Estoque_Total', 
                'Alcance_-_Estoque_total_+_Novo_pedido_(Model)', 
                'Alcance_-_Estoque_total_+_Novo_pedido_(Baseline)', 'Flag']
transit_cols = [c for c in df_ruptura.columns if c.startswith('Transit_')]
ruptura_cols = [c for c in df_ruptura.columns if c.startswith('Ruptura_') and not c.endswith(' Flag')]
ruptura_flag_cols = [c for c in df_ruptura.columns if c.startswith('Ruptura_') and c.endswith(' Flag')]

other_info = ['NewProduct', 'IsException', 'Check_Suggestion', 'Min_Order_Value_Warning']

known_order = (
    main_info + alcance_cols + metrics + sales_cols + 
    transit_cols + ruptura_cols + ruptura_flag_cols + other_info
)

all_cols = list(df_ruptura.columns)

ordered_cols = [col for col in known_order if col in all_cols]
other_cols = [col for col in all_cols if col not in ordered_cols]
final_order = ordered_cols + other_cols

df_ruptura = df_ruptura[final_order]

print("Reorganização concluída. Todas as colunas do df_main foram mantidas.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def safe_convert_to_int64(series):
    try:
        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_series = numeric_series.replace([np.inf, -np.inf], np.nan)
        return numeric_series.round(0).astype("Int64")
    except Exception as e:
        print(f"Erro na conversão: {e}")
        return pd.to_numeric(series, errors="coerce")

# 1) Colunas de string
string_cols = [
    "Component", "Cod_X", "Description", "Group_code", "Group_name",
    "Supplier", "ABC", "Obs", 'New_ABC', 'Flag'
]
ruptura_flag_cols = [c for c in df_ruptura.columns if c.startswith('Ruptura_') and c.endswith(' Flag')]
string_cols.extend(ruptura_flag_cols)

for col in string_cols:
    if col in df_ruptura.columns:
        df_ruptura[col] = df_ruptura[col].astype("string").fillna("")

# 2) Colunas de inteiros (fixas)
int_cols = [
    "Supp_Cod", "Stock", "Transit", "Inspection", "Reserved", "Total_Stock",
    "Sales_12M", "Unfulfilled_12M", "KanBan_Min", "KanBan_Max",
    "Final_order", "Final_order_baseline", "Final_order_ruptura",
    "Alert", "Safety_stock", "RP", "LT", "LT+RP", "Inventory_level", "Demand_(LT+RP)",
    'Venda_mensal', 'Nova_cobertura_Ruptura',
    'Alcance_-_Estoque_Atual', 'Alcance_-_Estoque_Total', 
    'Alcance_-_Estoque_total_+_Novo_pedido_(Model)',
    'Alcance_-_Estoque_total_+_Novo_pedido_(Baseline)'
]

for col in int_cols:
    if col in df_ruptura.columns:
        df_ruptura[col] = safe_convert_to_int64(df_ruptura[col])

df_ruptura["Supp_Cod"] = df_ruptura["Supp_Cod"].astype(str)

# 3) Colunas de float (fixas)
float_cols = ["Participacao", "%_Export_12M", "Cost", "Total_Cost", "Cobertura"]
for col in float_cols:
    if col in df_ruptura.columns:
        df_ruptura[col] = pd.to_numeric(df_ruptura[col], errors="coerce")
        df_ruptura[col] = df_ruptura[col].replace([np.inf, -np.inf], np.nan).round(2)

# 4) Colunas de inteiros (dinâmicas)
transit_cols = [col for col in df_ruptura.columns if col.startswith("Transit_")]
sales_m_cols = [col for col in df_ruptura.columns if col.startswith("Sales-M")]
ruptura_cols = [col for col in df_ruptura.columns if col.startswith("Ruptura_") and not col.endswith("Flag")]

all_dynamic_int_cols = transit_cols + sales_m_cols + ruptura_cols

if all_dynamic_int_cols:
    for col in all_dynamic_int_cols:
        df_ruptura[col] = safe_convert_to_int64(df_ruptura[col])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
Helpers.save_output_dataset(context=context, output_name='df_ruptura', data_frame=df_ruptura)