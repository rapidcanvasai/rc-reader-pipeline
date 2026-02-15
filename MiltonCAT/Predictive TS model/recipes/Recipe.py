# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Inicializar contexto
from utils.notebookhelpers.helpers import Helpers
from utils.dtos.templateOutputCollection import TemplateOutputCollection
from utils.dtos.templateOutput import TemplateOutput
from utils.dtos.templateOutput import OutputType
from utils.dtos.templateOutput import ChartType
from utils.dtos.variable import Metadata
from utils.rcclient.commons.variable_datatype import VariableDatatype
from utils.dtos.templateOutput import FileType
from utils.dtos.rc_ml_model import RCMLModel
import pandas as pd
import numpy as np

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Parâmetros configuráveis
MIN_MAPE_THRESHOLD = 5  # apenas calcular MAPE mensal quando Count >= threshold
ROUND_PRED = True       # arredondar demanda para inteiro no cálculo do gap
SAFETY_STOCK = 0        # estoque de segurança por modelo (unidades não consumíveis)
LEAD_TIME_MONTHS = 0    # deslocar demanda forward por lead time em meses

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Função para normalizar MODEL do inventário para corresponder ao Model_Eq. do forecast
def normalize_model_eq_from_inventory(model: str) -> str:
    """Normalize inventory MODEL to match forecast Model_Eq. as best as possible.
    Heuristics:
    - keep as-is if it already matches common patterns (e.g., '249D 3', '262D')
    - otherwise, take the first token before space (e.g., '313 GC' -> '313', '255 DCA2' -> '255')
    - strip quotes and whitespace
    """
    if pd.isna(model):
        return ''
    s = str(model).strip().strip('"')
    # If it contains a letter-number plus trailing generation like '249D 3', keep as-is
    # Otherwise, reduce to the first token
    if ' ' in s:
        head = s.split(' ')[0]
        # If head is short (e.g., '249D') but original looks like '249D 3', it's safer to keep full
        # because forecasts may use '249D 3'. Try to detect patterns 'D ' or numeric + space + digit
        if any(tag in s for tag in ['D ', ' XE', ' GC', ' LGP']) or any(ch.isdigit() for ch in s[len(head):]):
            # Special cases like '255 DCA2' should collapse to '255'
            # If head is purely digits (e.g., '255'), use head
            if head.replace('.', '').isdigit():
                return head
            # Else try full string
            return s
        return head
    return s

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Função para calcular gap de inventário
def compute_inventory_gap(
    demand_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    key_col: str,
    date_col: str = 'Sell_Date',
    demand_col: str = 'Pred',
    stock_col: str = 'Stock',
    safety_stock: float = 0,
    lead_time_months: int = 0,
    round_pred: bool = True,
) -> pd.DataFrame:
    """
    Gap by month per item (key_col) comparing initial stock vs future demand.

    Returns columns:
      [key_col, Sell_Date, Pred, CumDemand, Stock, Remaining, Gap_Month, Backlog, Served]
    """
    df = demand_df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.to_period('M').dt.to_timestamp()

    if lead_time_months:
        df[date_col] = df[date_col] + pd.DateOffset(months=lead_time_months)

    df = df.groupby([key_col, date_col], as_index=False)[demand_col].sum()
    if round_pred:
        df[demand_col] = df[demand_col].round().astype(int)

    inv = inventory_df[[key_col, stock_col]].drop_duplicates(subset=[key_col]).copy()
    out = df.merge(inv, on=key_col, how='left')
    out[stock_col] = out[stock_col].fillna(0).astype(float)

    out['_avail_start'] = (out[stock_col] - float(safety_stock)).clip(lower=0)

    def per_item(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(date_col).copy()
        S = float(g['_avail_start'].iloc[0]) if len(g) else 0.0
        g['CumDemand'] = g[demand_col].cumsum()
        g['Backlog'] = (g['CumDemand'] - S).clip(lower=0)
        g['Gap_Month'] = g['Backlog'].diff().clip(lower=0)
        g['Gap_Month'] = g['Gap_Month'].fillna(g['Backlog'])
        g['Remaining'] = (S - g['CumDemand']).clip(lower=0)
        g['Served'] = g[demand_col] - g['Gap_Month']
        return g

    out = out.groupby(key_col, group_keys=False).apply(per_item)
    cols = [key_col, date_col, demand_col, 'CumDemand', stock_col, 'Remaining', 'Gap_Month', 'Backlog', 'Served']
    out = out[cols].rename(columns={date_col: 'Sell_Date', demand_col: 'Pred'})
    return out

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Carregar os datasets necessários
future_df = Helpers.getEntityData(context, 'df_future_count_sales')
results_df = Helpers.getEntityData(context, 'df_results_count_sales')
inv_raw = Helpers.getEntityData(context, 'Inventory')

# FILTRAR APENAS O ENSEMBLE
future_df = future_df[future_df['Model'] == 'ensemble']
results_df = results_df[results_df['Model'] == 'ensemble']

print(f"Carregados {len(future_df):,} registros de demanda futura")
print(f"Carregados {len(results_df):,} registros de resultados históricos")
print(f"Carregados {len(inv_raw):,} itens no inventário")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Normalizar modelo do inventário para alinhar com Model_Eq. do forecast
inv_norm = inv_raw.copy()
inv_norm['Model_Eq.'] = inv_norm['MODEL'].apply(normalize_model_eq_from_inventory)

# Construir contagens de inventário por Model_Eq.
inv_counts = (
    inv_norm.groupby('Model_Eq.')
    .size()
    .reset_index(name='Stock')
)

print(f"\nModelos únicos no inventário: {len(inv_counts)}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Preparar dados de demanda futura
future_use = future_df[['Sell_Date', 'Model_Eq.', 'Model', 'Pred']].copy()
future_use['Sell_Date'] = pd.to_datetime(future_use['Sell_Date'])
future_use['Pred'] = pd.to_numeric(future_use['Pred'], errors='coerce').fillna(0.0)

# Criar grid completo: todos os modelos x todos os meses (preencher Pred=0 quando ausente)
models_all = pd.Index(sorted(set(future_use['Model_Eq.'].unique()) | set(inv_counts['Model_Eq.'].unique())))
months_all = pd.date_range(future_use['Sell_Date'].min(), future_use['Sell_Date'].max(), freq='MS')

full_grid = (
    pd.MultiIndex.from_product([models_all, months_all], names=['Model_Eq.', 'Sell_Date'])
    .to_frame(index=False)
)

print(f"Grid completo: {len(models_all)} modelos × {len(months_all)} meses = {len(full_grid):,} combinações")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Agregar future_use e fazer merge com o grid completo
agg_future = (
    future_use.groupby(['Model_Eq.', 'Sell_Date'], as_index=False)['Pred'].sum()
)

future_full = full_grid.merge(agg_future, on=['Model_Eq.', 'Sell_Date'], how='left')
future_full['Pred'] = future_full['Pred'].fillna(0.0)
future_full['Model'] = 'ensemble'  # placeholder

print(f"Demanda futura agregada: {len(future_full):,} linhas")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Calcular gap de inventário usando o grid completo
print("\nCalculando gap de inventário...")
gap_df = compute_inventory_gap(
    demand_df=future_full[['Sell_Date', 'Model_Eq.', 'Model', 'Pred']],
    inventory_df=inv_counts,
    key_col='Model_Eq.',
    stock_col='Stock',
    safety_stock=SAFETY_STOCK,
    lead_time_months=LEAD_TIME_MONTHS,
    round_pred=ROUND_PRED,
)

print(f"✓ Gap calculado: {len(gap_df):,} linhas")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Calcular MAPE mensal por modelo usando results_df (histórico/validação)
valid_agg = (
    results_df.groupby(['Model_Eq.', 'Sell_Date'], as_index=False)
    .agg(Count=('Count', 'sum'), Pred=('Pred', 'sum'))
)

valid_agg['Sell_Date'] = pd.to_datetime(valid_agg['Sell_Date'])
valid_agg['Count'] = pd.to_numeric(valid_agg['Count'], errors='coerce').fillna(0.0)
valid_agg['Pred'] = pd.to_numeric(valid_agg['Pred'], errors='coerce').fillna(0.0)

valid_agg['MAPE_month'] = np.where(
    valid_agg['Count'] >= MIN_MAPE_THRESHOLD,
    (np.abs(valid_agg['Pred'] - valid_agg['Count']) / valid_agg['Count']) * 100,
    np.nan,
)

print(f"\nMAPE calculado para {len(valid_agg):,} períodos históricos")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Fazer merge do MAPE no resultado do gap
final_df = gap_df.merge(
    valid_agg[['Model_Eq.', 'Sell_Date', 'MAPE_month']],
    on=['Model_Eq.', 'Sell_Date'],
    how='left'
)

# Ordenar colunas e linhas
final_df = final_df[
    ['Model_Eq.', 'Sell_Date', 'Pred', 'Served', 'Gap_Month', 'Remaining', 'Stock', 'CumDemand', 'Backlog', 'MAPE_month']
].sort_values(['Model_Eq.', 'Sell_Date']).reset_index(drop=True)

print(f"\n✓ Dataset final preparado: {len(final_df):,} linhas")
print(f"Colunas: {list(final_df.columns)}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Salvar dataset final
Helpers.save_output_dataset(context=context, output_name='df_gap', data_frame=final_df)
print(f"\n✓ Dataset salvo: 'df_gap' com {len(final_df):,} linhas")
print("\nPrimeiras linhas do resultado:")
print(final_df.head(20))