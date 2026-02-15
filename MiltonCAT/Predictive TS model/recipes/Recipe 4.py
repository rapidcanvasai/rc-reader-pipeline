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

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from difflib import get_close_matches
import warnings
warnings.filterwarnings('ignore')

# Initialize or fetch the execution context
context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Configuration (entity names inside RapidCanvas)
GAP_ENTITY_NAME = "df_gap"          # Dataset com Model_Eq., Stock, Pred, Sell_Date
LEADTIME_ENTITY_NAME = "Lead Times" # Dataset com Sales_Model, Lead Times, etc.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read input datasets from RapidCanvas entities

df_gap = Helpers.getEntityData(context, GAP_ENTITY_NAME)
df_lead_times = Helpers.getEntityData(context, LEADTIME_ENTITY_NAME)

# Make sure we are working with pandas DataFrames
df_gap = pd.DataFrame(df_gap).copy()
df_lead_times = pd.DataFrame(df_lead_times).copy()

# Standardize column names (trim spaces)
df_gap.columns = [str(c).strip() for c in df_gap.columns]
df_lead_times.columns = [str(c).strip() for c in df_lead_times.columns]

# Convert dates to datetime
df_gap['Sell_Date'] = pd.to_datetime(df_gap['Sell_Date'], errors='coerce')

print(f"✅ Dados carregados:")
print(f"   - df_gap: {len(df_gap)} registros")
print(f"   - lead_times: {len(df_lead_times)} registros")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Helper Functions

def calculate_avg_lead_time_days(lead_times_df):
    """
    Calcula o lead time médio em dias para cada equipamento
    Converte semanas para dias quando necessário
    """
    lead_times_processed = []
    
    for idx, row in lead_times_df.iterrows():
        # Identificar coluna de modelo (Sales_Model ou Sales_Model_Code)
        model_col = None
        for col in ['Sales_Model', 'Sales_Model_Code']:
            if col in row.index and pd.notna(row[col]):
                model_col = col
                break
        
        if model_col is None:
            continue
            
        # Converter para dias se estiver em semanas
        unit = str(row.get('Lead_Time_Units', 'WK')).upper()
        
        if unit == 'WK':
            min_days = row['Min_Lead_Time'] * 7
            max_days = row['Max_Lead_Time'] * 7
        else:  # já está em dias (DY)
            min_days = row['Min_Lead_Time']
            max_days = row['Max_Lead_Time']
        
        # Calcular média
        avg_days = (min_days + max_days) / 2
        
        lead_times_processed.append({
            'Equipment': row[model_col],
            'Min_Lead_Time_Days': min_days,
            'Avg_Lead_Time_Days': avg_days,
            'Max_Lead_Time_Days': max_days
        })
    
    return pd.DataFrame(lead_times_processed)

def calculate_global_avg_lead_times(lead_times_df):
    """
    Calcula as médias globais de todos os lead times do arquivo
    """
    return {
        'Min_Days': lead_times_df['Min_Lead_Time_Days'].mean(),
        'Avg_Days': lead_times_df['Avg_Lead_Time_Days'].mean(),
        'Max_Days': lead_times_df['Max_Lead_Time_Days'].mean()
    }

def match_equipment_lead_times(equipment, lead_times_df, global_avg):
    """
    Faz match do equipamento com lead times por prefixo
    """
    # Buscar todos os modelos que começam com o equipment
    matches = lead_times_df[lead_times_df['Equipment'].str.startswith(str(equipment), na=False)]
    
    if len(matches) > 0:
        # Calcular médias dos matches encontrados
        return {
            'Min_Days': matches['Min_Lead_Time_Days'].mean(),
            'Avg_Days': matches['Avg_Lead_Time_Days'].mean(),
            'Max_Days': matches['Max_Lead_Time_Days'].mean(),
            'Match_Type': f'Prefix ({len(matches)} matches)',
            'Has_Lead_Time_Info': True
        }
    else:
        # Usar média global
        return {
            'Min_Days': global_avg['Min_Days'],
            'Avg_Days': global_avg['Avg_Days'],
            'Max_Days': global_avg['Max_Days'],
            'Match_Type': 'Global Average',
            'Has_Lead_Time_Info': False
        }

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Process Lead Times

print("\n📊 Processando lead times...")
lead_times_processed = calculate_avg_lead_time_days(df_lead_times)
print(f"   - Lead times processados: {len(lead_times_processed)} equipamentos")

# Calcular médias globais
global_avg = calculate_global_avg_lead_times(lead_times_processed)
print(f"   - Médias globais (dias): Min={global_avg['Min_Days']:.1f}, Avg={global_avg['Avg_Days']:.1f}, Max={global_avg['Max_Days']:.1f}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Calculate Net Demand (Sequential Stock Consumption)

def calculate_net_demand(df_gap):
    """
    Calcula a demanda líquida considerando o consumo sequencial do estoque
    """
    net_demand_list = []
    
    # Agrupar por equipamento
    for equipment in df_gap['Model_Eq.'].unique():
        equipment_data = df_gap[df_gap['Model_Eq.'] == equipment].sort_values('Sell_Date').reset_index(drop=True)
        
        # Pegar estoque inicial
        initial_stock = equipment_data['Stock'].iloc[0] if len(equipment_data) > 0 else 0
        initial_stock = pd.to_numeric(initial_stock, errors='coerce')
        initial_stock = 0 if pd.isna(initial_stock) else initial_stock
        
        remaining_stock = initial_stock
        
        for idx, row in equipment_data.iterrows():
            demand = pd.to_numeric(row['Pred'], errors='coerce')
            demand = 0 if pd.isna(demand) else demand
            
            # GUARDAR O ESTOQUE ANTES DE CONSUMIR
            stock_before_demand = remaining_stock
            
            # Calcular quanto do estoque será usado
            stock_used = min(remaining_stock, demand)
            net_demand = max(0, demand - stock_used)
            
            # Atualizar estoque remanescente
            remaining_stock = max(0, remaining_stock - stock_used)
            
            net_demand_list.append({
                'Equipment': equipment,
                'Sell_Date': row['Sell_Date'],
                'Predicted_Demand': demand,
                'Stock_Before_Demand': stock_before_demand,
                'Stock_Used': stock_used,
                'Net_Demand': net_demand,
                'Remaining_Stock': remaining_stock
            })
    
    return pd.DataFrame(net_demand_list)

print("\n🔄 Calculando demanda líquida (consumo sequencial do estoque)...")
net_demand_df = calculate_net_demand(df_gap)
print(f"   - Demanda líquida calculada: {len(net_demand_df)} registros")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Calculate Order Dates and Generate Orders

def calculate_order_dates(net_demand_df, lead_times_df, global_avg):
    """
    Calcula as datas de pedido baseadas no lead time de cada equipamento
    """
    orders = []
    
    # Rastrear tipos de match
    match_stats = {'Prefix': 0, 'Global Average': 0}
    
    for idx, row in net_demand_df.iterrows():
        equipment = row['Equipment']
        
        # Se há demanda líquida, criar ordem
        if row['Net_Demand'] > 0:
            # Fazer match do equipamento
            lead_info = match_equipment_lead_times(equipment, lead_times_df, global_avg)
            
            # Contar tipos de match
            if 'Prefix' in lead_info['Match_Type']:
                match_stats['Prefix'] += 1
            else:
                match_stats['Global Average'] += 1
            
            # Calcular data do pedido (data de venda - lead time médio)
            if pd.notna(row['Sell_Date']):
                order_by_date = row['Sell_Date'] - timedelta(days=int(lead_info['Avg_Days']))
            else:
                order_by_date = None
            
            # Determinar prioridade baseada no estoque e lead time
            stock = row['Stock_Before_Demand']
            lead_time_days = lead_info['Avg_Days']
            gap = row['Net_Demand']
            
            if stock == 0:
                priority = "High"
            elif gap > 0 and lead_time_days >= 120:
                priority = "High"
            elif gap > 0 and lead_time_days >= 75:
                priority = "Medium"
            else:
                priority = "Low"
            
            orders.append({
                'Equipment': equipment,
                'Priority': priority,
                'Stock': stock,
                'Recommended_Order_(Gap)': row['Net_Demand'],
                'Min_Lead_Time': int(lead_info['Min_Days']),
                'Max_Lead_Time': int(lead_info['Max_Days']),
                'Lead_Time': int(lead_info['Avg_Days']),
                'Order_by_Date': order_by_date.date() if pd.notna(order_by_date) else None,
                'For_Demand_Date': row['Sell_Date'].date() if pd.notna(row['Sell_Date']) else None,
                'Match_Type': lead_info['Match_Type']
            })
    
    print(f"\n   ✅ Matches por prefixo: {match_stats['Prefix']} ordens")
    print(f"   📊 Usando média global: {match_stats['Global Average']} ordens")
    
    return pd.DataFrame(orders)

print("\n📅 Calculando datas de pedido e prioridades...")
orders_df = calculate_order_dates(net_demand_df, lead_times_processed, global_avg)
print(f"   - {len(orders_df)} ordens de compra geradas")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Final Selection and Sorting

final_cols = [
    "Equipment",
    "Priority",
    "Stock",
    "Recommended_Order_(Gap)",
    "Min_Lead_Time",
    "Max_Lead_Time",
    "Lead_Time",
    "Order_by_Date",
]

# Ensure all final columns exist
for col in final_cols:
    if col not in orders_df.columns:
        orders_df[col] = pd.NA

# Select final columns
output_df = orders_df[final_cols].copy()

# Create a categorical type for Priority to ensure correct sorting order
priority_order = pd.CategoricalDtype(categories=["High", "Medium", "Low"], ordered=True)
output_df["Priority"] = output_df["Priority"].astype(priority_order)

# Sort by Priority (High to Low) and then by Order_by_Date (most recent first)
output_df = output_df.sort_values(
    by=["Priority", "Order_by_Date"],
    ascending=[True, False],
    na_position="last"
).reset_index(drop=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Save Output to RapidCanvas

Helpers.save_output_dataset(
    context=context, 
    output_name='inventory_gap_with_lead_times', 
    data_frame=output_df
)

# Print Summary
print("\n" + "="*80)
print("RESUMO FINAL")
print("="*80)
print(f"\n📦 Total de ordens geradas: {len(output_df)}")

if len(output_df) > 0:
    print(f"\n📊 Por Prioridade:")
    print(output_df['Priority'].value_counts().sort_index())
    
    print(f"\n🔝 Top 10 Ordens:")
    print(output_df.head(10)[['Equipment', 'Priority', 'Recommended_Order_(Gap)', 'Lead_Time', 'Order_by_Date']].to_string(index=False))
else:
    print("\n⚠️  Nenhuma ordem de compra foi gerada")
    print("   Possíveis razões:")
    print("   1. Não há demanda líquida (estoque suficiente)")

print("\n✅ Processo concluído!")