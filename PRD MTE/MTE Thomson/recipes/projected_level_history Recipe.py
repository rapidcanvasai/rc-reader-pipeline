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
from dateutil.relativedelta import relativedelta
import holidays
import datetime 
import zipfile
import io

def print_df_info(df_name, df):
    """Imprime informações básicas de um DataFrame"""
    print(f"\n{'='*70}")
    print(f"📊 DataFrame: {df_name}")
    print(f"{'='*70}")
    print(f"   Shape: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    print(f"   Colunas: {list(df.columns)}")
    print(f"{'='*70}\n")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# df_inventory_histories
df_inventory_histories = Helpers.getEntityData(context, 'new_inventory_histories2')
print_df_info("df_inventory_histories", df_inventory_histories)

# Guardar cópia original para output inventory_stock_history
inventory_stock_history = df_inventory_histories.copy()

# df_main
df_main = Helpers.getEntityData(context, 'main')
print_df_info("df_main", df_main)

print("\n🔧 Preparando df_main...")

# ✅ VERIFICAÇÃO INTELIGENTE: Verifica qual coluna usar
# Detecta qual variante do nome da coluna existe (Cod_X ou Cod X)
cod_x_col = 'Cod_X' if 'Cod_X' in df_main.columns else ('Cod X' if 'Cod X' in df_main.columns else None)

if 'Component' in df_main.columns and cod_x_col:
    print(f"   ✅ Colunas 'Component' e '{cod_x_col}' existem no dataset")
elif 'Component' in df_main.columns:
    print("   ✅ Coluna 'Component' já existe no dataset")
elif cod_x_col:
    df_main['Component'] = df_main[cod_x_col]
    print(f"   ✅ Coluna 'Component' criada a partir de '{cod_x_col}' (mantendo ambas)")
else:
    print("   ❌ ERRO CRÍTICO: Nenhuma coluna 'Component' ou 'Cod_X'/'Cod X' encontrada!")
    print(f"   Colunas disponíveis: {df_main.columns.tolist()}")
    raise ValueError("Coluna 'Component' não encontrada em df_main")

# Indexar por Component (se ainda não estiver indexado)
if 'Component' not in df_main.index.names:
    df_main = df_main.set_index('Component', drop=False)
    print("   ✅ df_main indexado por 'Component'")
else:
    print("   ✅ df_main já está indexado por 'Component'")

print(f"   Shape final de df_main: {df_main.shape}")
print(f"   Componentes únicos: {df_main['Component'].nunique()}")

# df_pedidos_pendentes
df_pedidos_pendentes = Helpers.getEntityData(context, 'df_pedidos_pendentes')
print_df_info("df_pedidos_pendentes", df_pedidos_pendentes)

# df_main_confidence_interval (para inventory_stock_metrics)
df_main_confidence_interval = Helpers.getEntityData(context, 'df_main_confidence_interval')
print_df_info("df_main_confidence_interval", df_main_confidence_interval)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Transformação: Pivot do histórico de estoque
print("\n🔄 Transformando histórico de estoque para formato WIDE...")

df_inventory_histories = df_inventory_histories.pivot_table(
    index='date',
    columns='Componente',
    values='QTD_ESTOQUE',
    fill_value=0
)

print("✅ df_inventory_histories transformado")
print(f"   Shape: {df_inventory_histories.shape[0]:,} datas × {df_inventory_histories.shape[1]:,} componentes")
print(f"   Período: {df_inventory_histories.index.min().date()} a {df_inventory_histories.index.max().date()}")
print(f"   Primeiros 5 componentes: {df_inventory_histories.columns[:5].tolist()}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_in_transit_orders(df_pedidos_pendentes: pd.DataFrame, df_artifact=pd.DataFrame()):

    today = datetime.datetime.now()

    df_pedidos_pendentes = df_pedidos_pendentes.replace("nan", None)
    df_pedidos_pendentes["DT. Prod. Real"] = ""
    df_pedidos_pendentes["OBS 1"] = ""
    df_pedidos_pendentes["OBS 2"] = ""

    df_pedidos_pendentes["DT. Prod. Prev."] = df_pedidos_pendentes["EMBARQUE_PREVISTO_PO"] - pd.Timedelta(15, unit="D")
    df_pedidos_pendentes["Status Pedido"] = np.where(df_pedidos_pendentes["PROCESSO"].isna(), "Negociação", "Transito")
    
    df_pedidos_pendentes["Alerta"] = np.where(
        (today > df_pedidos_pendentes["DT. Prod. Prev."]) & (pd.isna(df_pedidos_pendentes['DT. Prod. Real'])), 
        "Prod. Atrasada", 
        "Em Progresso")

    col_names = {
        "Alerta":"Alerta", "DATA_SI":"Data SI", "NUMERO_SI":"SI",
        "PRODUTO":"MTE", "CODIGO_X_MTE": "MTE X", "QTDE_NAO_ENTREGUE":"Qtd",
        "ENTREGA_PREVISTA_PO": "DT. Entrega PO", "CHEG_PORTO_ETA_15": "Entrega Prevista",
        "FORNEC_NOM": "Fornecedor", "COD_PROD_FOR": "Cod Prod. Forn.", "PROFORMA": "Proforma",
        "PEDIDO": "PO", "INVOICE": "Invoice", "PROCESSO": "Código Embarque",
        "CONFIRMACAO_PEDIDO": "Conf. PO", "DT. Prod. Prev.": "DT. Prod. Prev.",
        "DT. Prod. Real": "DT. Prod. Real", "EMBARQUE_EFET": "Data Embarque",
        "CHEG_PORTO_ETA": "Data Prevista Chegada Porto", "Status Pedido": "Status Pedido",
        "OBS 1": "OBS 1", "OBS 2": "OBS 2"
    }

    df_pedidos_pendentes.rename(columns=col_names, inplace=True)
    df_pedidos_pendentes = df_pedidos_pendentes[col_names.values()]
    
    if not df_artifact.empty:
        df_artifact['key'] = df_artifact['MTE'].astype(str) + df_artifact['PO'].astype(str)
        df_pedidos_pendentes['key'] = df_pedidos_pendentes['MTE'].astype(str) + df_pedidos_pendentes['PO'].astype(str)
        df_artifact['Data Prod Artifact'] = df_artifact["DT. Prod. Real"]
        df_artifact = df_artifact[['key', 'Data Prod Artifact']]
        df_pedidos_pendentes = pd.merge(df_pedidos_pendentes, df_artifact, on='key', how='left')
        df_pedidos_pendentes["DT. Prod. Real"] = df_pedidos_pendentes['Data Prod Artifact']
        df_pedidos_pendentes = df_pedidos_pendentes.drop(columns=['key', 'Data Prod Artifact'])
        
        df_pedidos_pendentes['Alerta'] = np.where(
            pd.notna(df_pedidos_pendentes['DT. Prod. Real']),
            np.where(pd.to_datetime(df_pedidos_pendentes["DT. Prod. Real"], format="%d/%m/%Y") > df_pedidos_pendentes['DT. Prod. Prev.'],
            "Produzido após prazo", "Produzido"),
            np.where(today > df_pedidos_pendentes["DT. Prod. Prev."], "Prod. Atrasada", "Em Progresso")
        )
        
    # Format dates
    for col in ["DT. Prod. Prev.", "Data SI", "DT. Entrega PO", "Entrega Prevista", "Data Embarque", "Data Prevista Chegada Porto"]:
        df_pedidos_pendentes[col] = df_pedidos_pendentes[col].dt.strftime("%d-%b-%Y")
    
    return df_pedidos_pendentes

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("\n🔄 Processando pedidos pendentes...")
df_pending_orders = get_in_transit_orders(df_pedidos_pendentes)
print(f"   ✅ {len(df_pending_orders)} pedidos processados")

df_pending_orders['DT. Entrega PO'] = pd.to_datetime(df_pending_orders['DT. Entrega PO'])
df_pending_orders['year_month'] = df_pending_orders['DT. Entrega PO'].dt.strftime('%Y-%m')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("\n📅 Configurando datas de referência...")

if not pd.api.types.is_datetime64_any_dtype(df_inventory_histories.index):
    df_inventory_histories.index = pd.to_datetime(df_inventory_histories.index)

usage_date = df_inventory_histories.index.max()
base_date = usage_date

print(f"   Base date: {pd.Timestamp(base_date).date()}")
print(f"   Usage date: {pd.Timestamp(usage_date).date()}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Garantir indexação
if 'Component' not in df_main.index.names:
    df_main = df_main.set_index('Component', drop=False)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("\n🔍 Identificando componentes para processar...")

df_main_filtered = df_main.copy()
components = df_main['Component'].unique().tolist()
component_selector = sorted(set(df_inventory_histories.columns).intersection(set(components)))

print(f"   Componentes em df_main: {len(components)}")
print(f"   Componentes no histórico: {len(df_inventory_histories.columns)}")
print(f"   Componentes para processar: {len(component_selector)}")
print(f"   Taxa de cobertura: {len(component_selector)/len(components)*100:.1f}%")

if len(component_selector) == 0:
    print("\n❌ ERRO CRÍTICO: Nenhum componente para processar!")
    print(f"   Exemplos df_main: {components[:10]}")
    print(f"   Exemplos histórico: {df_inventory_histories.columns[:10].tolist()}")
    raise ValueError("Nenhum componente para processar. Verifique os nomes dos componentes.")

print(f"   ✅ Primeiros 5: {component_selector[:5]}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def calculate_projected_level(usage_date, lt, rp, current_level, final_order, demand, df_intransit):
    first_date = usage_date + pd.Timedelta(1, unit="D")
    arrival_date_current_order = usage_date + pd.Timedelta(lt, unit="D")
    final_date = usage_date + pd.Timedelta(lt+rp, unit="D")

    demand_per_day = demand/(lt+rp) if (lt+rp) > 0 else 0
    date_range = pd.date_range(first_date, final_date, freq="D")
    projected_level = current_level
    df_projected_level_history = pd.DataFrame(columns=["Date", "Projected level"])
    total_lost = 0
    
    df_intransit["DT. Entrega PO"] = pd.to_datetime(df_intransit["DT. Entrega PO"], format="%d-%b-%Y")
    
    for date in date_range:
        projected_level += df_intransit[df_intransit["DT. Entrega PO"] == date]["Qtd"].sum()
        if date == arrival_date_current_order:
            projected_level += final_order
        if projected_level > demand_per_day:
            projected_level -= demand_per_day
        else:
            total_lost += demand_per_day
            projected_level = 0
        df_projected_level_history = pd.concat([
            df_projected_level_history,
            pd.DataFrame({"Date": [date], "Projected level": [projected_level]})
        ], ignore_index=True)
    
    return df_projected_level_history, total_lost

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_projected_level_history_list = []

print(f"\n🔄 Processando {len(component_selector)} componentes...")
processed_count = 0
skipped_count = 0
errors_shown = 0
MAX_ERRORS_TO_SHOW = 5

for idx, temp_comp in enumerate(component_selector):
    if (idx + 1) % 500 == 0:
        print(f"   ✅ Processados: {idx + 1}/{len(component_selector)} (sucesso: {processed_count}, pulados: {skipped_count})")
    
    try:
        # ✅ CORREÇÃO: Garantir que sempre pegamos um único valor (primeira ocorrência)
        component_data = df_main.loc[temp_comp]
        
        # Se retornar Series (múltiplas linhas), pegar apenas a primeira
        if isinstance(component_data, pd.DataFrame):
            component_data = component_data.iloc[0]
        
        component_lead_time = component_data["LT"]
        component_review_period = component_data["RP"]
        component_inventory_level = component_data["Inventory_level"]
        component_final_order = component_data["Final_order"]
        component_demand = component_data["Demand_(LT+RP)"]
        
    except KeyError as e:
        skipped_count += 1
        if errors_shown < MAX_ERRORS_TO_SHOW:
            print(f"   ⚠️ Componente '{temp_comp}' não encontrado em df_main")
            errors_shown += 1
        continue

    # ✅ CORREÇÃO: Verificar NaN com tratamento adequado para escalares
    variables_to_check = [
        component_lead_time, component_review_period, component_inventory_level, 
        component_final_order, component_demand
    ]
    
    # Garantir que são valores escalares antes de verificar NaN
    has_nan = any(
        pd.isna(v) if np.isscalar(v) else pd.isna(v).any() 
        for v in variables_to_check
    )

    temp_pending_orders = df_pending_orders[df_pending_orders["MTE"] == temp_comp]
    
    if has_nan:
        skipped_count += 1
        if errors_shown < MAX_ERRORS_TO_SHOW:
            print(f"   ⚠️ Componente '{temp_comp}' tem valores NaN")
            errors_shown += 1
    else:
        try:
            history_df, _ = calculate_projected_level(
                usage_date,
                component_lead_time,
                component_review_period,
                component_inventory_level,
                component_final_order,
                component_demand,
                temp_pending_orders
            )

            if not history_df.empty:
                history_df["component"] = temp_comp
                df_projected_level_history_list.append(history_df)
                processed_count += 1
        except Exception as e:
            skipped_count += 1
            if errors_shown < MAX_ERRORS_TO_SHOW:
                print(f"   ⚠️ Erro ao processar '{temp_comp}': {str(e)[:80]}")
                errors_shown += 1

# Concatenação final
if df_projected_level_history_list:
    df_projected_level_history = pd.concat(df_projected_level_history_list, ignore_index=True)
else:
    df_projected_level_history = pd.DataFrame()

print(f"\n{'='*70}")
print(f"✅ Processamento concluído!")
print(f"{'='*70}")
print(f"   Componentes processados com sucesso: {processed_count}")
print(f"   Componentes pulados: {skipped_count}")
print(f"   df_projected_level_history: {len(df_projected_level_history)} registros")
print(f"{'='*70}\n")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def clean_whitespace(df, column):
    """Remove espaços em branco no início e fim de uma coluna de string."""
    if column not in df.columns:
        return df
    return df[column].astype(str).str.strip()
    
def filter_t_components(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Filtra linhas onde a coluna especificada começa com 'T' seguida por um ou mais dígitos."""
    if df.empty or column not in df.columns:
        return df
    return df[df[column].astype(str).str.match(r'^T\d+')]

def remove_double_dot_lines(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Remove linhas em que a coluna selecionada contém exatamente dois pontos "."."""
    if df.empty or column not in df.columns:
        return df
    return df[df[column].astype(str).str.count(r'\.') != 2]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Limpeza de df_projected_level_history
if not df_projected_level_history.empty and 'component' in df_projected_level_history.columns:
    print("🧹 Aplicando limpeza em df_projected_level_history...")
    inicial = len(df_projected_level_history)
    
    df_projected_level_history["component"] = clean_whitespace(df_projected_level_history, "component")
    df_projected_level_history = filter_t_components(df_projected_level_history, 'component')
    df_projected_level_history = remove_double_dot_lines(df_projected_level_history, 'component')
    
    final = len(df_projected_level_history)
    print(f"   Registros: {inicial:,} → {final:,} (removidos: {inicial - final:,})")
else:
    print("❌ ERRO: df_projected_level_history está vazio ou sem coluna 'component'!")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Converter forecast para dados mensais (último dia de cada mês)
print("\n📅 Convertendo forecast para frequência MENSAL...")

inicial_forecast = len(df_projected_level_history)
df_projected_level_history['Date'] = pd.to_datetime(df_projected_level_history['Date'])

# Pegar o último registro de cada mês por componente
df_projected_level_history = df_projected_level_history.groupby(
    ['component', pd.Grouper(key='Date', freq='M')]
).last().reset_index()

final_forecast = len(df_projected_level_history)
meses_forecast = df_projected_level_history['Date'].nunique()

print(f"   Registros: {inicial_forecast:,} → {final_forecast:,}")
print(f"   Meses no forecast: {meses_forecast}")
print(f"   ✅ Forecast convertido para frequência mensal (último dia de cada mês)")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Filtrar inventory_stock_history para ter apenas os componentes do forecast
print("\n🔍 Filtrando inventory_stock_history...")

forecast_components = set(df_projected_level_history['component'].unique())
inicial_history = len(inventory_stock_history)
inicial_components = inventory_stock_history['Componente'].nunique()

inventory_stock_history = inventory_stock_history[
    inventory_stock_history['Componente'].isin(forecast_components)
]

final_history = len(inventory_stock_history)
final_components = inventory_stock_history['Componente'].nunique()

print(f"   Registros: {inicial_history:,} → {final_history:,}")
print(f"   Componentes: {inicial_components:,} → {final_components:,}")
print(f"   ✅ Agora inventory_stock_history tem os mesmos {final_components:,} componentes do forecast")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Converter para dados mensais (último dia de cada mês)
print("\n📅 Convertendo histórico para frequência MENSAL...")

inicial_registros = len(inventory_stock_history)
inventory_stock_history['date'] = pd.to_datetime(inventory_stock_history['date'])

# Pegar o último registro de cada mês por componente
inventory_stock_history = inventory_stock_history.groupby(
    ['Componente', pd.Grouper(key='date', freq='M')]
).last().reset_index()

final_registros = len(inventory_stock_history)
meses = inventory_stock_history['date'].nunique()

print(f"   Registros: {inicial_registros:,} → {final_registros:,}")
print(f"   Meses no histórico: {meses}")
print(f"   ✅ Histórico convertido para frequência mensal (último dia de cada mês)")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Criar inventory_stock_metrics a partir de df_main_confidence_interval
print("\n📊 Criando inventory_stock_metrics...")

metrics_columns = [
    'Component', 'Description', 'Group_code', 'Supp_Cod', 'Safety_stock',
    'LT', 'RP', 'LT+RP', 'Inventory_level', 'Final_order', 'Demand_(LT+RP)',
    'Transit', 'Inspection', 'min', 'max', 'precision'
]

# Verificar se todas as colunas existem
missing_cols = [col for col in metrics_columns if col not in df_main_confidence_interval.columns]
if missing_cols:
    print(f"   ⚠️ Colunas faltando em df_main_confidence_interval: {missing_cols}")
    available_cols = [col for col in metrics_columns if col in df_main_confidence_interval.columns]
    inventory_stock_metrics = df_main_confidence_interval[available_cols].copy()
else:
    inventory_stock_metrics = df_main_confidence_interval[metrics_columns].copy()

print(f"   ✅ inventory_stock_metrics criado: {len(inventory_stock_metrics):,} registros, {len(inventory_stock_metrics.columns)} colunas")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ❌ VALIDAÇÃO FINAL - BLOQUEAR SALVAMENTO SE VAZIO
print(f"\n{'='*70}")
print("💾 VALIDAÇÃO FINAL")
print(f"{'='*70}")

errors = []

if df_projected_level_history.empty:
    errors.append("inventory_stock_forecast está VAZIO")

if inventory_stock_history.empty:
    errors.append("inventory_stock_history está VAZIO")

if inventory_stock_metrics.empty:
    errors.append("inventory_stock_metrics está VAZIO")

if errors:
    print("❌ ERRO CRÍTICO: Datasets vazios detectados!")
    for error in errors:
        print(f"   • {error}")
    print(f"{'='*70}")
    raise ValueError(f"Não é possível salvar datasets vazios: {', '.join(errors)}")

print(f"✅ Validação aprovada!")
print(f"   inventory_stock_forecast: {len(df_projected_level_history):,} registros")
print(f"   inventory_stock_history: {len(inventory_stock_history):,} registros")
print(f"   inventory_stock_metrics: {len(inventory_stock_metrics):,} registros")
print(f"{'='*70}\n")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Padronização de colunas (lowercase, inglês)
print("🔧 Padronizando nomes das colunas...")

# inventory_stock_history
inventory_stock_history = inventory_stock_history.rename(columns={
    'Componente': 'component',
    'QTD_ESTOQUE': 'stock_quantity'
})

# inventory_stock_forecast (df_projected_level_history)
df_projected_level_history = df_projected_level_history.rename(columns={
    'Date': 'date',
    'Projected level': 'stock_forecast'
})

# inventory_stock_metrics (todas para minúsculo, tratar caracteres especiais)
inventory_stock_metrics = inventory_stock_metrics.rename(columns={
    'Component': 'component',
    'Description': 'description',
    'Group_code': 'group_code',
    'Supp_Cod': 'supp_cod',
    'Safety_stock': 'safety_stock',
    'LT': 'lt',
    'RP': 'rp',
    'LT+RP': 'lt_rp',
    'Inventory_level': 'inventory_level',
    'Final_order': 'final_order',
    'Demand_(LT+RP)': 'demand_lt_rp',
    'Transit': 'transit',
    'Inspection': 'inspection'
})

print("   ✅ Colunas padronizadas")

# Salvamento
print("💾 Salvando datasets...")

Helpers.save_output_dataset(context=context, output_name='inventory_stock_forecast', data_frame=df_projected_level_history)
print(f"   ✅ inventory_stock_forecast salvo")

Helpers.save_output_dataset(context=context, output_name='inventory_stock_history', data_frame=inventory_stock_history)
print(f"   ✅ inventory_stock_history salvo")

Helpers.save_output_dataset(context=context, output_name='inventory_stock_metrics', data_frame=inventory_stock_metrics)
print(f"   ✅ inventory_stock_metrics salvo")

print(f"\n🎉 SUCESSO! 3 datasets salvos.\n")