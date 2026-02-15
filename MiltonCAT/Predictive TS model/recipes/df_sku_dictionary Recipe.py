# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Required imports
from utils.notebookhelpers.helpers import Helpers
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Carregar os dados
raw_sales = Helpers.getEntityData(context, 'Machine Sales New')
df_sku_dictionary = Helpers.getEntityData(context, 'df_sku_dictionary')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Filtrar apenas registros onde Fleet == 'New'
raw_sales = raw_sales[raw_sales['Fleet'] == 'New'].reset_index(drop=True)

# Remover espaços da coluna Model
raw_sales['Model'] = raw_sales['Model'].str.replace(' ', '', regex=False)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Limpeza - remover colunas desnecessárias
df_sales = raw_sales.copy()

to_drop_cols = [
    'Sales_ID','Salesman', 'Sales_Manager', 'Customer_Name',
    'SUPPLEMENTALOF', 'Fleet','Eq_ID','Serial',
    'Book_Value', 'Milton_List_Price',
    'Item','Year','Hours','Group_Code','Make',
    'Overallowance', 'Merchandising', 'Parts_Work_Orders', 'PDI_Charges',
    'Net_Sales_Price', 'Estimated_GP', 'GP_%', '%FMV', 'Attached_To',
    'Fiscal_Year', 'Period', 'Estimated_Credits', 'Estimated_Commitments',
    'Gross_Margin', 'Est_GP_Gross_Margin', 'Act_vs_Est_Gross_Margin',
    'Estimated_Support_Dollars',
    'source_file', 'General_Manager','PIN_Date', 'Product_Family'
]

df_sales = df_sales.drop(columns=to_drop_cols)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Agregação mensal por Model (SKU)
df_sales['month'] = df_sales['Sell_Date'].dt.to_period('M').dt.to_timestamp()

df_sales_monthly = (
    df_sales
    .groupby(['month', 'Model'])
    .size()
    .reset_index(name='sales_count')
)

df_sales_monthly.columns = ['date','sku','quantity']
df_sales_monthly = df_sales_monthly[['sku','date','quantity']]
df_sales_monthly.sort_values(by=['sku','date'], inplace=True)
df_sales_monthly = df_sales_monthly.reset_index(drop=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Merge para adicionar Group e SubGroup
df_sales_monthly_v1 = df_sales_monthly.copy()

df_sales_monthly_v1 = df_sales_monthly_v1.merge(
    df_sku_dictionary[['Model','SKU_Group','SKU_SubGroup']],
    left_on='sku',
    right_on='Model',
    how='left'
)

# Remover SKUs sem grupo definido
df_sales_monthly_v1.dropna(subset=['SKU_Group','SKU_SubGroup'], inplace=True)
df_sales_monthly_v1 = df_sales_monthly_v1.reset_index(drop=True)

# Remover coluna duplicada do merge
df_sales_monthly_v1.drop(columns=['Model'], inplace=True)

# Renomear colunas
df_sales_monthly_v1.columns = ['sku','date','quantity','group','subgroup']

# Reordenar colunas
df_sales_monthly_v1 = df_sales_monthly_v1[['group','subgroup','sku','date','quantity']]

# Ordenar dados
df_sales_monthly_v1 = df_sales_monthly_v1.sort_values(by=['group','subgroup','sku','date']).reset_index(drop=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Salvar output para o framework (obrigatório para a recipe funcionar)
Helpers.save_output_dataset(context=context, output_name='clean', data_frame=df_sales_monthly_v1)