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
# from utils.libutils.vectorStores.utils import VectorStoreUtils  # Não utilizado

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# CONFIGURAÇÕES
FUTURE_PERIOD = 6           # Meses para prever no futuro
TEST_MONTHS = 6             # Meses para validação (holdout)
BLEND_WEIGHT_AI = 0.5       # Peso do modelo XGBoost no blend
BLEND_WEIGHT_AVG = 0.5      # Peso da média histórica no blend
LONGTAIL_BOOST_FACTOR = 1.25  # Fator de ajuste para previsões long-tail

# Critérios de elegibilidade para XGBoost
ZERO_RATIO_THRESHOLD = 0.2  # Máximo de zeros permitido (20%)
MEAN_VOLUME_THRESHOLD = 1.0 # Volume médio mínimo
ELIGIBILITY_WINDOW_MONTHS = 18  # Janela de meses para avaliar elegibilidade

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import pandas as pd
import numpy as np
import warnings
import xgboost as xgb

warnings.filterwarnings('ignore')


def filter_complete_months(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Remove o mês atual (incompleto) dos dados.
    Garante corte no 1º dia do mês À MEIA-NOITE.
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    first_day_current_month = pd.Timestamp.today().normalize().replace(day=1)
    df_complete = df[df[date_column] < first_day_current_month].copy()

    rows_removed = len(df) - len(df_complete)
    if rows_removed > 0:
        print(f"Filtro de meses completos: {rows_removed} registros do mês atual removidos.")

    return df_complete
pd.options.display.float_format = '{:.2f}'.format
pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', None)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Funções utilitárias

def get_data_summary(df: pd.DataFrame):
    """Gera tabela resumo dos dados."""
    data_summary = df.dtypes.to_frame('dtypes')
    data_summary['non_null'] = df.count()
    data_summary['unique_values'] = df.apply(lambda srs: len(srs.unique()))
    data_summary['first_row'] = df.iloc[0]
    data_summary['last_row'] = df.iloc[-1]
    return data_summary


def is_xgboost_eligible(df_subset: pd.DataFrame, target_col: str = 'quantity') -> tuple:
    """
    Verifica se uma série temporal é elegível para XGBoost.

    Critérios (baseados nos últimos 18 meses):
    1. Zero Ratio < 20% (não pode ter muitos zeros)
    2. Mean Volume > 1.0 (média mensal mínima)

    Returns:
        tuple: (elegível: bool, métricas: dict)
    """
    # Ordenar por data e pegar apenas últimos 18 meses
    df_sorted = df_subset.sort_values('date')
    if len(df_sorted) > ELIGIBILITY_WINDOW_MONTHS:
        df_recent = df_sorted.iloc[-ELIGIBILITY_WINDOW_MONTHS:]
    else:
        df_recent = df_sorted

    ts = df_recent.groupby('date')[target_col].sum()
    zero_ratio = (ts == 0).sum() / len(ts)
    mean_val = ts.mean()

    eligible = (zero_ratio < ZERO_RATIO_THRESHOLD) and (mean_val > MEAN_VOLUME_THRESHOLD)

    return eligible, {
        'zero_ratio': zero_ratio,
        'mean_val': mean_val
    }


def mape(y_true, y_pred):
    """Calcula Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Preparação dos dados

def prepare_sales_data(df_sales: pd.DataFrame, cutoff_date: str = None) -> pd.DataFrame:
    """
    Prepara os dados de vendas garantindo que cada SKU tenha todos os meses.

    Args:
        df_sales: DataFrame com colunas [group, subgroup, sku, date, quantity]
        cutoff_date: Data de corte (formato 'YYYY-MM-DD'). Se None, usa dados até o mês anterior.

    Returns:
        DataFrame preparado com todos os meses preenchidos
    """
    df = df_sales.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Aplicar cutoff se especificado
    if cutoff_date:
        df = df[df['date'] < pd.Timestamp(cutoff_date)].reset_index(drop=True)

    # Garantir que cada SKU tenha todos os meses
    all_skus = df['sku'].unique()
    all_dates = pd.date_range(df['date'].min(), df['date'].max(), freq='MS')

    full_index = pd.MultiIndex.from_product([all_skus, all_dates], names=['sku', 'date'])
    df_full = pd.DataFrame(index=full_index).reset_index()

    df_merged = df_full.merge(df, on=['sku', 'date'], how='left', suffixes=('', '_orig'))

    # Preencher group/subgroup
    for col in ['group', 'subgroup']:
        if col in df_merged.columns:
            df_merged[col] = df_merged.groupby('sku')[col].transform(lambda x: x.ffill().bfill())

    # Preencher quantity com 0
    df_merged['quantity'] = df_merged['quantity'].fillna(0).astype(int)

    df_merged = df_merged[['group', 'subgroup', 'sku', 'date', 'quantity']]
    df_merged = df_merged.sort_values(['group', 'subgroup', 'sku', 'date']).reset_index(drop=True)

    return df_merged


def create_hierarchical_aggregations(df_sales: pd.DataFrame) -> tuple:
    """
    Cria agregações em diferentes níveis hierárquicos.

    Returns:
        tuple: (df_group, df_subgroup, df_sku)
    """
    df_group = df_sales.groupby(['group', 'date']).agg({'quantity': 'sum'}).reset_index()
    df_subgroup = df_sales.groupby(['subgroup', 'date']).agg({'quantity': 'sum'}).reset_index()
    df_sku = df_sales.groupby(['sku', 'date']).agg({'quantity': 'sum'}).reset_index()

    return df_group, df_subgroup, df_sku


def scan_and_split_series(df_group: pd.DataFrame, df_subgroup: pd.DataFrame, df_sku: pd.DataFrame) -> tuple:
    """
    Escaneia séries SKU e separa entre elegíveis para XGBoost e long-tail.
    (V2: apenas SKU é escaneado, GROUP e SUBGROUP não são mais usados)

    Returns:
        tuple: (df_global para XGBoost, lista de candidatos top-down)
    """
    xgboost_dfs = []
    top_down_candidates = []

    # Escanear apenas SKUs (V2: removido GROUP e SUBGROUP)
    print("Scanning SKUs...")
    for item in df_sku['sku'].unique():
        subset = df_sku[df_sku['sku'] == item].copy()
        eligible, metrics = is_xgboost_eligible(subset)

        if eligible:
            subset['series_id'] = f"SKU_{item}"
            subset['level'] = 'SKU'
            subset['original_id'] = item
            xgboost_dfs.append(subset[['date', 'series_id', 'level', 'original_id', 'quantity']])
        else:
            top_down_candidates.append({'level': 'SKU', 'id': item, 'metrics': metrics})

    # Consolidar dataset XGBoost
    if xgboost_dfs:
        df_global = pd.concat(xgboost_dfs, ignore_index=True)
    else:
        df_global = pd.DataFrame()

    print(f"Resultado: {len(xgboost_dfs)} séries para XGBoost, {len(top_down_candidates)} para Top-Down")

    return df_global, top_down_candidates


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Feature Engineering

#TIME_WINDOW_LIST = [1, 2, 3, 6, 12, 24]
TIME_WINDOW_LIST = [1, 2, 3, 6, 12]


def create_features(df_agg: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """
    Cria features para o modelo XGBoost.

    Args:
        df_agg: DataFrame com colunas [date, category, sales_current_month]
        include_target: Se True, cria coluna target (sales_next_1_month)

    Returns:
        DataFrame com features criadas
    """
    df = df_agg.copy()

    # Time Features
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Lag features
    for lag in TIME_WINDOW_LIST:
        df[f"lag_{lag}"] = df.groupby("category")["sales_current_month"].shift(lag)

    # Difference features
    for diff in TIME_WINDOW_LIST:
        df[f"diff_{diff}"] = df.groupby("category")["sales_current_month"].diff(diff)

    # Rolling statistics
    for roll in TIME_WINDOW_LIST:
        df[f"roll_mean_{roll}"] = (
            df.groupby("category")["sales_current_month"]
            .rolling(roll).mean()
            .reset_index(level=0, drop=True)
        )

    # Encoding categorical features
    category_series = df['category'].copy()
    df = pd.get_dummies(df, columns=['category'])
    df['category'] = category_series

    # Backfill missing values por categoria
    df = df.groupby('category').apply(lambda group: group.bfill()).reset_index(drop=True)

    # Drop NA
    df = df.dropna().reset_index(drop=True)

    # Trend feature (sequência temporal)
    df['month_of_sequence'] = df.groupby(['category'])['date'].rank(method='dense')
    df['month_of_sequence'] = df['month_of_sequence'].astype(np.int64)

    # Target: vendas do próximo mês
    if include_target:
        df["sales_next_1_month"] = df.groupby("category")["sales_current_month"].shift(-1, fill_value=np.nan)

    return df


def get_features_for_date(df_agg: pd.DataFrame, cutoff_date, target_col: str = 'sales_next_1_month') -> pd.DataFrame:
    """
    Gera features para uma data específica (usado na previsão iterativa).
    """
    df = df_agg.copy()

    # Time Features
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Lag features
    for lag in TIME_WINDOW_LIST:
        df[f"lag_{lag}"] = df.groupby("category")["sales_current_month"].shift(lag)

    # Difference features
    for diff in TIME_WINDOW_LIST:
        df[f"diff_{diff}"] = df.groupby("category")["sales_current_month"].diff(diff)

    # Rolling statistics
    for roll in TIME_WINDOW_LIST:
        df[f"roll_mean_{roll}"] = (
            df.groupby("category")["sales_current_month"]
            .rolling(roll).mean()
            .reset_index(level=0, drop=True)
        )

    # Encoding categorical features
    category_series = df['category'].copy()
    df = pd.get_dummies(df, columns=['category'])
    df['category'] = category_series

    # Backfill missing values
    df = df.groupby('category').apply(lambda group: group.bfill()).reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)

    # Trend feature
    df['month_of_sequence'] = df.groupby(['category'])['date'].rank(method='dense')
    df['month_of_sequence'] = df['month_of_sequence'].astype(np.int64)

    # Filtrar apenas a data de corte
    df = df[df['date'] == cutoff_date]
    df[target_col] = None

    return df


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Modelo XGBoost e Previsão

def train_xgboost_model(df_agg: pd.DataFrame, test_months: int = TEST_MONTHS) -> tuple:
    """
    Treina modelo XGBoost com split temporal.

    Returns:
        tuple: (df_train, df_test, df_valid_results)
    """
    target_col = "sales_next_1_month"
    non_feature_cols = ['date', 'category', target_col]

    # Train-test split baseado em sequência temporal
    first_month_of_test = df_agg['month_of_sequence'].max() - test_months
    print(f"Split: meses de teste a partir da sequência {first_month_of_test}")

    df_train = df_agg[df_agg['month_of_sequence'] < first_month_of_test].reset_index(drop=True)
    df_test = df_agg[df_agg['month_of_sequence'] >= first_month_of_test].reset_index(drop=True)

    # Preparar features
    X_train = df_train.drop(non_feature_cols, axis=1)
    y_train = df_train[target_col]

    X_test = df_test.drop(non_feature_cols, axis=1)
    # y_test usado apenas para referência no df_valid

    # Treinar modelo
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)

    # Validação
    y_pred = model.predict(X_test)

    df_valid = df_test.copy()
    df_valid['y_pred'] = y_pred
    df_valid['y_test'] = df_test[target_col]
    df_valid['y_pred'] = np.where(df_valid['y_pred'] <= 0, 0, df_valid['y_pred'])

    return df_train, df_test, df_valid


def generate_future_predictions_xgboost(df_agg: pd.DataFrame, feature_columns: list,
                                         future_period: int = FUTURE_PERIOD) -> pd.DataFrame:
    """
    Gera previsões iterativas para meses futuros usando XGBoost.
    """
    target_col = 'sales_next_1_month'
    non_feature_cols = ['date', 'category', target_col]

    # Retreinar modelo com todos os dados disponíveis
    df_all_data = df_agg[df_agg[target_col].notnull()]
    X_all = df_all_data.drop(non_feature_cols, axis=1)
    y_all = df_all_data[target_col]

    model_final = xgb.XGBRegressor(objective='reg:squarederror')
    model_final.fit(X_all, y_all)

    # Primeira previsão (próximo mês)
    df_last = df_agg[df_agg[target_col].isnull()]
    X_last = df_last.drop(non_feature_cols, axis=1)
    y_pred = model_final.predict(X_last)

    # Gerar datas futuras
    category_list = df_agg['category'].unique().tolist()
    future_dates = pd.date_range(
        start=df_agg['date'].max() + pd.DateOffset(months=1),
        periods=future_period,
        freq='MS'
    )

    # DataFrame para acumular previsões
    df_agg_future = df_agg[['date', 'category', 'sales_current_month']].copy()

    # Previsão iterativa
    for i, future_date in enumerate(future_dates):
        # Criar DataFrame para a data futura
        df_future_step = pd.DataFrame({
            'date': [future_date] * len(category_list),
            'category': category_list,
            'sales_current_month': y_pred if i == 0 else this_y_pred
        })
        df_future_step['sales_current_month'] = np.maximum(df_future_step['sales_current_month'], 0)

        # Concatenar com histórico
        df_agg_future = pd.concat([df_agg_future, df_future_step])
        df_agg_future = df_agg_future.sort_values(by=['category', 'date']).reset_index(drop=True)

        # Gerar features para próxima iteração
        if i < len(future_dates) - 1:
            df_features = get_features_for_date(df_agg_future, future_date, target_col)
            X_next = df_features.drop(non_feature_cols, axis=1, errors='ignore')
            # Garantir mesmas colunas
            for col in feature_columns:
                if col not in X_next.columns:
                    X_next[col] = 0
            X_next = X_next[feature_columns]
            this_y_pred = model_final.predict(X_next)

    # Marcar tipo (actual vs future)
    df_agg_future['type'] = 'actual'
    df_agg_future.loc[df_agg_future['date'].isin(future_dates), 'type'] = 'future'

    return df_agg_future


def apply_blend_with_historical_avg(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica blend 50/50 entre previsão XGBoost e média histórica de 3 anos.
    """
    # Separar histórico e futuro
    df_history = df_pred[df_pred['type'] == 'actual'].reset_index(drop=True)
    df_history['month'] = df_history['date'].dt.month

    # Calcular média sazonal por categoria/mês
    seasonal_profile = (
        df_history
        .groupby(['category', 'month'])['sales_current_month']
        .mean()
        .reset_index()
        .rename(columns={'sales_current_month': 'hist_avg_sales'})
    )

    # Preparar previsões futuras
    df_future = df_pred[df_pred['type'] != 'actual'].copy().reset_index(drop=True)
    df_future['month'] = df_future['date'].dt.month

    # Merge com média histórica
    df_blend = df_future.merge(seasonal_profile, on=['category', 'month'], how='left')

    # Aplicar blend
    df_blend['sales_blended'] = (
        df_blend['sales_current_month'] * BLEND_WEIGHT_AI +
        df_blend['hist_avg_sales'].fillna(0) * BLEND_WEIGHT_AVG
    )

    # Atualizar previsões com valores blendados
    df_future_blended = df_blend[['date', 'category', 'sales_blended', 'type']].copy()
    df_future_blended = df_future_blended.rename(columns={'sales_blended': 'sales_current_month'})

    # Reconstruir DataFrame final
    df_history_clean = df_history[['date', 'category', 'sales_current_month', 'type']]
    df_final = pd.concat([df_history_clean, df_future_blended])
    df_final = df_final.sort_values(by=['category', 'date']).reset_index(drop=True)

    return df_final


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Previsão para Long-Tail (Top-Down)

def validate_longtail_forecast(df_rest: pd.DataFrame, test_months: int = TEST_MONTHS) -> pd.DataFrame:
    """
    Valida previsões long-tail usando holdout temporal.
    Como cada mês é independente, basta:
    1. Separar holdout
    2. Calcular componentes com dados de treino
    3. Prever cada mês do holdout
    4. Comparar com real
    """
    results = []

    for cat_id in df_rest['category'].unique():
        history = df_rest[df_rest['category'] == cat_id].sort_values('date').copy()

        if len(history) <= test_months:
            continue

        # Holdout: últimos test_months
        cutoff_date = history['date'].iloc[-(test_months + 1)]

        df_train = history[history['date'] <= cutoff_date].copy()
        df_holdout = history[history['date'] > cutoff_date].copy()

        if df_train.empty or df_holdout.empty:
            continue

        # Média mensal (apenas treino)
        df_train['month'] = df_train['date'].dt.month
        monthly_avg_map = df_train.groupby('month')['quantity'].mean().to_dict()
        overall_avg = df_train['quantity'].mean()

        # Quarterly upscale (apenas treino)
        history_q = df_train.set_index('date')['quantity'].resample('QS').sum()

        if len(history_q) >= 4:
            next_q_vol = history_q.rolling(window=4).mean().iloc[-1]
        elif len(history_q) > 0:
            next_q_vol = history_q.mean()
        else:
            next_q_vol = overall_avg * 3

        model_monthly_val = next_q_vol / 3.0

        # Prever cada mês do holdout (independente) com fator de boost (V2: 1.15)
        for _, row in df_holdout.iterrows():
            m = row['date'].month
            val_baseline = monthly_avg_map.get(m, overall_avg)
            val_pred = LONGTAIL_BOOST_FACTOR * (BLEND_WEIGHT_AI * model_monthly_val + BLEND_WEIGHT_AVG * val_baseline)

            results.append({
                'date': row['date'],
                'category': cat_id,
                'y_test': row['quantity'],
                'y_pred': round(val_pred, 4)
            })

    return pd.DataFrame(results)


def generate_blended_forecast_longtail(df_rest: pd.DataFrame, start_date: str, periods: int = FUTURE_PERIOD) -> pd.DataFrame:
    """
    Gera previsões para séries long-tail usando blend de:
    1. Quarterly Upscale (tendência suavizada)
    2. Média mensal histórica (sazonalidade)
    """
    results = []
    future_dates = pd.date_range(start=start_date, periods=periods, freq='MS')

    for cat_id in df_rest['category'].unique():
        history = df_rest[df_rest['category'] == cat_id].sort_values('date').copy()

        # Componente A: Média mensal histórica (baseline sazonal)
        history['month'] = history['date'].dt.month
        monthly_avg_map = history.groupby('month')['quantity'].mean().to_dict()

        # Componente B: Quarterly Upscale (tendência suavizada)
        history_q = history.set_index('date')['quantity'].resample('QS').sum()

        if len(history_q) >= 4:
            next_q_vol = history_q.rolling(window=4).mean().iloc[-1]
        elif len(history_q) > 0:
            next_q_vol = history_q.mean()
        else:
            next_q_vol = 0

        model_monthly_val = next_q_vol / 3.0

        # Blend 50/50 com fator de boost (V2: 1.15)
        for date in future_dates:
            m = date.month
            val_baseline = monthly_avg_map.get(m, 0)
            val_model = model_monthly_val
            val_final = LONGTAIL_BOOST_FACTOR * (BLEND_WEIGHT_AI * val_model + BLEND_WEIGHT_AVG * val_baseline)

            results.append({
                'date': date,
                'category': cat_id,
                'sales_current_month': round(val_final, 4),
                'type': 'future'
            })

    return pd.DataFrame(results)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
############################
# EXECUÇÃO PRINCIPAL
############################

# 1. Carregar dados (formato: group, subgroup, sku, date, quantity)
df_sales_raw = Helpers.getEntityData(context, 'clean')
df_sales_raw['date'] = pd.to_datetime(df_sales_raw['date'])

# 1.1 Filtrar apenas meses completos (remove mês atual incompleto)
df_sales_raw = filter_complete_months(df_sales_raw, 'date')

# 2. Preparar dados
print("Preparando dados...")
df_sales = prepare_sales_data(df_sales_raw)

# 3. Criar agregações hierárquicas
print("Criando agregações hierárquicas...")
df_group, df_subgroup, df_sku = create_hierarchical_aggregations(df_sales)

# 4. Separar séries elegíveis para XGBoost vs Long-tail
print("Classificando séries...")
df_global, top_down_candidates = scan_and_split_series(df_group, df_subgroup, df_sku)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 5. Processar séries XGBoost

if not df_global.empty:
    print("\n=== Processando séries XGBoost ===")

    # Preparar formato para modelo
    df_agg = df_global[['date', 'series_id', 'quantity']].copy()
    df_agg.columns = ['date', 'category', 'sales_current_month']

    # Feature engineering
    df_agg = create_features(df_agg)

    # Treinar e validar
    df_train, df_test, df_valid = train_xgboost_model(df_agg)

    # Obter colunas de features
    target_col = 'sales_next_1_month'
    non_feature_cols = ['date', 'category', target_col]
    feature_columns = [c for c in df_train.columns if c not in non_feature_cols]

    # Gerar previsões futuras
    df_pred_xgb = generate_future_predictions_xgboost(df_agg, feature_columns)

    # Aplicar blend com média histórica
    df_pred_xgb = apply_blend_with_historical_avg(df_pred_xgb)
    df_pred_xgb['bucket'] = 'xgboost'

    print(f"XGBoost: {df_pred_xgb['category'].nunique()} séries processadas")
else:
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_pred_xgb = pd.DataFrame()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 6. Processar séries Long-tail (Top-Down)

print("\n=== Processando séries Long-tail ===")

# Preparar dados long-tail (V2: apenas SKU)
df_sku_lt = df_sku.copy()
df_sku_lt.rename(columns={'sku': 'category'}, inplace=True)
df_sku_lt['category'] = 'SKU_' + df_sku_lt['category'].astype(str)

df_rest = df_sku_lt.copy()

# Remover séries já processadas pelo XGBoost
if not df_pred_xgb.empty:
    xgb_categories = df_pred_xgb['category'].unique()
    df_rest = df_rest[~df_rest['category'].isin(xgb_categories)]

df_rest = df_rest.reset_index(drop=True)

if not df_rest.empty:
    # Validação long-tail
    print("Validando séries Long-tail...")
    df_valid_lt = validate_longtail_forecast(df_rest)
    print(f"Long-tail validação: {len(df_valid_lt)} registros")

    # Gerar previsões long-tail
    start_date = df_rest['date'].max() + pd.DateOffset(months=1)
    df_forecast_lt = generate_blended_forecast_longtail(df_rest, start_date.strftime('%Y-%m-%d'))

    # Combinar histórico com previsões
    df_rest['type'] = 'actual'
    df_rest.rename(columns={'quantity': 'sales_current_month'}, inplace=True)
    df_pred_lt = pd.concat([df_rest, df_forecast_lt])
    df_pred_lt = df_pred_lt.sort_values(by=['category', 'date']).reset_index(drop=True)
    df_pred_lt['bucket'] = 'longtail'

    print(f"Long-tail: {df_pred_lt['category'].nunique()} séries processadas")
else:
    df_pred_lt = pd.DataFrame()
    df_valid_lt = pd.DataFrame()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 7. Consolidar resultados finais

print("\n=== Consolidando resultados ===")

# Merge XGBoost + Long-tail
df_merged_pred = pd.concat([df_pred_xgb, df_pred_lt])
df_merged_pred = df_merged_pred.sort_values(by=['category', 'date']).reset_index(drop=True)
df_merged_pred['date'] = pd.to_datetime(df_merged_pred['date']).dt.strftime('%Y-%m-%d')
df_merged_pred['sales_current_month'] = df_merged_pred['sales_current_month'].round().astype(int)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 8. Preparar outputs separados (usando nomenclatura antiga)

# Output 1: Histórico (df_historical)
df_historical = df_merged_pred[df_merged_pred['type'] == 'actual'].copy()
df_historical = df_historical[['date', 'category', 'sales_current_month', 'bucket']]
df_historical.columns = ['Sell_Date', 'Model Eq.', 'Count', 'Model']

# Output 2: Validação (df_validation) - Consolidar XGBoost + Long-tail
validation_dfs = []

# Validação XGBoost
if not df_valid.empty:
    df_valid_xgb = df_valid[['date', 'category', 'y_test', 'y_pred']].copy()
    df_valid_xgb = df_valid_xgb.dropna(subset=['y_test'])
    # Ajustar data para o mês PREVISTO (não o mês do registro)
    df_valid_xgb['date'] = pd.to_datetime(df_valid_xgb['date']) + pd.DateOffset(months=1)
    df_valid_xgb['y_pred'] = df_valid_xgb['y_pred'].round().astype(int)
    df_valid_xgb['y_test'] = df_valid_xgb['y_test'].round().astype(int)
    df_valid_xgb['Model'] = 'xgboost'
    validation_dfs.append(df_valid_xgb)

# Validação Long-tail
if not df_valid_lt.empty:
    df_valid_lt_out = df_valid_lt[['date', 'category', 'y_test', 'y_pred']].copy()
    df_valid_lt_out['y_pred'] = df_valid_lt_out['y_pred'].round().astype(int)
    df_valid_lt_out['y_test'] = df_valid_lt_out['y_test'].round().astype(int)
    df_valid_lt_out['Model'] = 'longtail'
    validation_dfs.append(df_valid_lt_out)

# Consolidar validações
if validation_dfs:
    df_validation = pd.concat(validation_dfs, ignore_index=True)
    df_validation.columns = ['Sell_Date', 'Model Eq.', 'Count', 'Pred', 'Model']
    df_validation['Sell_Date'] = pd.to_datetime(df_validation['Sell_Date']).dt.strftime('%Y-%m-%d')
else:
    df_validation = pd.DataFrame(columns=['Sell_Date', 'Model Eq.', 'Count', 'Pred', 'Model'])

# Output 3: Futuro (df_future)
df_future = df_merged_pred[df_merged_pred['type'] == 'future'].copy()
df_future = df_future[['date', 'category', 'sales_current_month', 'bucket']]
df_future.columns = ['Sell_Date', 'Model Eq.', 'Pred', 'Model']

# Remover prefixo "SKU_" da coluna Model Eq. em todos os outputs
df_historical['Model Eq.'] = df_historical['Model Eq.'].str.replace('SKU_', '', regex=False)
df_validation['Model Eq.'] = df_validation['Model Eq.'].str.replace('SKU_', '', regex=False)
df_future['Model Eq.'] = df_future['Model Eq.'].str.replace('SKU_', '', regex=False)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 9. Salvar outputs

print("\n=== Salvando outputs ===")
print(f"Histórico: {len(df_historical)} registros")
print(f"Validação: {len(df_validation)} registros")
print(f"Futuro: {len(df_future)} registros")

Helpers.save_output_dataset(context=context, output_name='df_count_machine_sales', data_frame=df_historical)
Helpers.save_output_dataset(context=context, output_name='df_results_count_sales', data_frame=df_validation)
Helpers.save_output_dataset(context=context, output_name='df_future_count_sales', data_frame=df_future)

print("\nProcessamento concluído!")