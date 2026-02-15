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

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import logging
import holidays
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    make_scorer, mean_absolute_error, mean_squared_error, 
    r2_score, mean_absolute_percentage_error, median_absolute_error
)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
pd.set_option("display.max_columns", None)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def wmape(y_true, y_pred):
    """Weighted Mean Absolute Percentage Error"""
    return np.sum(np.abs(y_true - y_pred))/np.sum(np.abs(y_true))

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error com tratamento de infinitos"""
    ape = np.abs(y_true - y_pred)/np.maximum(y_true, 1e-10)
    
    if np.isscalar(ape):
        if np.isfinite(ape):
            return ape
        else:
            return 1
    else:
        ape[~np.isfinite(ape)] = 1
    return np.mean(ape)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def predict_ensemble(model_dict, df_features):
    """
    Faz previsões usando modelos ABC/XYZ agrupados por cluster.
    """
    
    # Tipo 1: Modelo ABC-Pareto
    if 'product_class_map' in model_dict and 'feature_names' in model_dict:
        print("     -> Usando modelo ABC-Pareto (classificação dinâmica)")
        
        features = model_dict['feature_names']
        product_class_map = model_dict['product_class_map']
        
        missing_features = set(features) - set(df_features.columns)
        if missing_features:
            raise ValueError(f"Features faltando no dataframe: {missing_features}")
        
        X = df_features[features].fillna(0).astype('float64')
        
        if 'COD_MTE_COMP' not in df_features.columns:
            raise ValueError("COD_MTE_COMP não encontrado. É necessário para roteamento ABC.")
        
        df_features['classe_abc'] = df_features['COD_MTE_COMP'].map(product_class_map).fillna('C')
        
        predictions = np.zeros(len(df_features))
        
        for classe in ['A', 'B', 'C']:
            mask = (df_features['classe_abc'] == classe)
            if mask.sum() == 0:
                continue
            
            X_classe = X.loc[mask]
            if X_classe.empty:
                continue
            
            model_classe = model_dict['models'][classe]
            
            if model_classe is None:
                predictions[mask.values] = 0.0
            else:
                pred_classe = model_classe.predict(X_classe)
                predictions[mask.values] = pred_classe
        
        return np.maximum(predictions, 0)
    
    # Tipo 2: Modelo ABC-XYZ
    elif 'ABC_XYZ' in df_features.columns:
        print("     -> Usando modelo ABC-XYZ (coluna pré-existente)")
        
        predictions = np.zeros(len(df_features))
        
        for cluster in df_features['ABC_XYZ'].unique():
            if cluster not in model_dict:
                print(f"⚠️ Cluster {cluster} não tem modelo treinado, usando predição zero")
                mask = (df_features['ABC_XYZ'] == cluster)
                predictions[mask] = 0.0
                continue
            
            mask = (df_features['ABC_XYZ'] == cluster)
            cluster_model = model_dict[cluster]['model']
            cluster_features = model_dict[cluster]['features']
            
            missing_features = set(cluster_features) - set(df_features.columns)
            if missing_features:
                print(f"⚠️ Features faltando para cluster {cluster}: {missing_features}")
                predictions[mask] = 0.0
                continue
            
            X_cluster = df_features.loc[mask, cluster_features].fillna(0)
            
            if len(X_cluster) > 0:
                cluster_predictions = cluster_model.predict(X_cluster)
                predictions[mask] = np.maximum(cluster_predictions, 0)
        
        return predictions
    
    else:
        raise ValueError("Formato de modelo não reconhecido. Esperado 'product_class_map' ou 'ABC_XYZ' em df_features")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def train_abc_models(df_train, df_test, features, user_param_grid=None, n_iter_search=50, 
                     cod_mte_comp='COD_MTE_COMP'):
    """
    Treina um modelo XGBoost separado para cada classe ABC usando classificação Pareto.
    """
    
    print("=" * 80)
    print("🚀 TREINAMENTO DE MODELOS XGBOOST SEPARADOS POR CLASSE ABC (PARETO)")
    print("=" * 80)
    
    if user_param_grid is None:
        user_param_grid = {}

    print("📊 Classificando produtos (ABC) com base no volume total de treino...")
    
    df_train_agg = df_train.groupby(cod_mte_comp)['QTDE_PEDIDA'].sum().reset_index()
    df_train_agg = df_train_agg.sort_values('QTDE_PEDIDA', ascending=False).reset_index(drop=True)
    
    df_train_agg['cumsum'] = df_train_agg['QTDE_PEDIDA'].cumsum()
    total_sales = df_train_agg['QTDE_PEDIDA'].sum()
    df_train_agg['cumsum_pct'] = (df_train_agg['cumsum'] / total_sales) * 100
    
    def get_abc_class(cumsum_pct):
        if cumsum_pct <= 80: return 'A'
        elif cumsum_pct <= 95: return 'B'
        else: return 'C'
    
    df_train_agg['classe_abc'] = df_train_agg['cumsum_pct'].apply(get_abc_class)
    
    print("📊 Classificando produtos (XYZ) com base na volatilidade...")
    
    df_train_volatility = df_train.groupby(cod_mte_comp)['QTDE_PEDIDA'].agg(['mean', 'std']).reset_index()
    df_train_volatility['cv'] = df_train_volatility['std'] / df_train_volatility['mean'].replace(0, np.nan)
    df_train_volatility['cv'] = df_train_volatility['cv'].fillna(0)
    
    def get_xyz_class(cv):
        if cv < 0.5: return 'X'
        elif cv < 1.0: return 'Y'
        else: return 'Z'
    
    df_train_volatility['classe_xyz'] = df_train_volatility['cv'].apply(get_xyz_class)
    
    df_train_agg = pd.merge(df_train_agg, 
                            df_train_volatility[[cod_mte_comp, 'cv', 'classe_xyz']], 
                            on=cod_mte_comp, 
                            how='left')
    
    df_train_agg['classe_abc_xyz'] = df_train_agg['classe_abc'] + df_train_agg['classe_xyz']
    
    product_class_map = df_train_agg.set_index(cod_mte_comp)['classe_abc'].to_dict()
    product_xyz_map = df_train_agg.set_index(cod_mte_comp)['classe_xyz'].to_dict()
    product_abc_xyz_map = df_train_agg.set_index(cod_mte_comp)['classe_abc_xyz'].to_dict()
    
    classes_train = df_train[cod_mte_comp].map(product_class_map).fillna('C').values
    classes_test = df_test[cod_mte_comp].map(product_class_map).fillna('C').values
    
    classes_xyz_train = df_train[cod_mte_comp].map(product_xyz_map).fillna('Z').values
    classes_abc_xyz_train = df_train[cod_mte_comp].map(product_abc_xyz_map).fillna('CZ').values
    
    n_A = (classes_train == 'A').sum()
    n_B = (classes_train == 'B').sum()
    n_C = (classes_train == 'C').sum()
    
    n_X = (classes_xyz_train == 'X').sum()
    n_Y = (classes_xyz_train == 'Y').sum()
    n_Z = (classes_xyz_train == 'Z').sum()
    
    print(f"📊 Distribuição de LINHAS de treino (produto-mês):")
    print(f"\n    ABC (Volume):")
    print(f"       Classe A: {n_A} linhas ({n_A/len(classes_train)*100:.1f}%)")
    print(f"       Classe B: {n_B} linhas ({n_B/len(classes_train)*100:.1f}%)")
    print(f"       Classe C: {n_C} linhas ({n_C/len(classes_train)*100:.1f}%)")
    print(f"\n    XYZ (Volatilidade):")
    print(f"       Classe X: {n_X} linhas ({n_X/len(classes_xyz_train)*100:.1f}%)")
    print(f"       Classe Y: {n_Y} linhas ({n_Y/len(classes_xyz_train)*100:.1f}%)")
    print(f"       Classe Z: {n_Z} linhas ({n_Z/len(classes_xyz_train)*100:.1f}%)")
    
    print(f"\n    Matriz ABC-XYZ (Top 5 combinações):")
    abc_xyz_counts = pd.Series(classes_abc_xyz_train).value_counts().head(5)
    for abc_xyz, count in abc_xyz_counts.items():
        print(f"       {abc_xyz}: {count} linhas ({count/len(classes_abc_xyz_train)*100:.1f}%)")
    print()

    target_col = 'sales_next_1_month' if 'sales_next_1_month' in df_train.columns else 'sales_next_month'
    
    X_train = df_train[features].fillna(0).astype('float64')
    y_train = df_train[target_col].fillna(0).astype('float64')
    X_test = df_test[features].fillna(0).astype('float64')
    y_test = df_test[target_col].fillna(0).astype('float64')

    abc_models = {
        'models': {}, 
        'feature_names': features, 
        'product_class_map': product_class_map,
        'product_xyz_map': product_xyz_map,
        'product_abc_xyz_map': product_abc_xyz_map
    }
    
    y_pred_train = np.zeros(len(y_train))
    y_pred_test = np.zeros(len(y_test))

    default_xgb_params = {
        'n_estimators': 500,
        'max_depth': 10,
        'learning_rate': 0.03,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_weight': 5,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'gamma': 0,
        'random_state': 42,
        'objective': 'reg:squarederror',
        'n_jobs': -1
    }

    for classe in ['A', 'B', 'C']:
        print("-" * 60)
        print(f"🧠 Treinando Modelo para CLASSE {classe}")
        print("-" * 60)
        
        mask_train = (classes_train == classe)
        X_train_classe = X_train[mask_train]
        y_train_classe = y_train[mask_train]
        
        if len(X_train_classe) == 0:
            print(f"    ⚠️ Sem dados de treino para Classe {classe}. Pulando.")
            abc_models['models'][classe] = None
            continue
        
        model_classe = None
        
        if classe in user_param_grid:
            print(f"    🔍 Executando RandomizedSearchCV para Classe {classe} (n_iter={n_iter_search})...")
            param_grid = user_param_grid[classe]
            
            base_model = xgb.XGBRegressor(n_jobs=-1, random_state=42)
            
            n_splits_safe = 3
            train_window_size = 12
            
            print(f"    CV: Janela Deslizante (Treino={train_window_size} meses, n_splits={n_splits_safe})")
            
            tscv = TimeSeriesSplit(
                n_splits=n_splits_safe, 
                max_train_size=train_window_size,
                test_size=3
            )
            
            search = RandomizedSearchCV(
                base_model, 
                param_grid, 
                scoring=make_scorer(mape, greater_is_better=False), 
                cv=tscv, 
                n_iter=n_iter_search, 
                n_jobs=-1, 
                verbose=1,
                random_state=42
            )
            
            try:
                search.fit(X_train_classe, y_train_classe)
                model_classe = search.best_estimator_
                print(f"    ✅ Melhores parâmetros: {search.best_params_}")
            except Exception as e:
                print(f"    ❌ ERRO no RandomizedSearchCV (Classe {classe}): {e}")
                print("    Voltando para parâmetros padrão...")
                model_classe = xgb.XGBRegressor(**default_xgb_params)
                model_classe.fit(X_train_classe, y_train_classe)
        
        else:
            print(f"    ⚙️  Treinando Classe {classe} com parâmetros padrão...")
            model_classe = xgb.XGBRegressor(**default_xgb_params)
            model_classe.fit(X_train_classe, y_train_classe)
        
        abc_models['models'][classe] = model_classe
        
        pred_train_classe = model_classe.predict(X_train_classe)
        y_pred_train[mask_train] = pred_train_classe
        
        mask_test = (classes_test == classe)
        X_test_classe = X_test[mask_test]
        if not X_test_classe.empty:
            pred_test_classe = model_classe.predict(X_test_classe)
            y_pred_test[mask_test] = pred_test_classe
        
        mae_train_classe = mean_absolute_error(y_train_classe, pred_train_classe)
        mape_train_classe = mape(y_train_classe, pred_train_classe)
        print(f"    📈 Classe {classe} (Treino) - MAE: {mae_train_classe:.2f} | MAPE: {mape_train_classe:.4f}")

    y_pred_train = np.maximum(y_pred_train, 0)
    y_pred_test = np.maximum(y_pred_test, 0)
    
    y_pred_train = np.nan_to_num(y_pred_train, nan=y_train.mean())
    y_pred_test = np.nan_to_num(y_pred_test, nan=y_train.mean())

    print("\n" + "=" * 80)
    print("📊 MÉTRICAS FINAIS (COMBINADAS)")
    print("=" * 80)
    
    for classe in ['A', 'B', 'C']:
        mask_train = (classes_train == classe)
        mask_test = (classes_test == classe)
        
        if mask_train.sum() > 0:
            mae_train = mean_absolute_error(y_train[mask_train], y_pred_train[mask_train])
            mape_train = mape(y_train[mask_train], y_pred_train[mask_train])
            wmape_train = wmape(y_train[mask_train], y_pred_train[mask_train])
            print(f"    TREINO Classe {classe} - MAE: {mae_train:.2f} | MAPE: {mape_train:.4f} | WMAE: {wmape_train:.4f}")
        
        if mask_test.sum() > 0:
            mae_test = mean_absolute_error(y_test[mask_test], y_pred_test[mask_test])
            mape_test = mape(y_test[mask_test], y_pred_test[mask_test])
            wmape_test = wmape(y_test[mask_test], y_pred_test[mask_test])
            print(f"    TESTE  Classe {classe} - MAE: {mae_test:.2f} | MAPE: {mape_test:.4f} | WMAE: {wmape_test:.4f}")
    
    print("-" * 40)
    mae_geral_train = mean_absolute_error(y_train, y_pred_train)
    mae_geral_test = mean_absolute_error(y_test, y_pred_test)
    wmape_geral_train = wmape(y_train, y_pred_train)
    wmape_geral_test = wmape(y_test, y_pred_test)
    print(f"    GERAL (Treino) - MAE: {mae_geral_train:.2f} | WMAE: {wmape_geral_train:.4f}")
    print(f"    GERAL (Teste)  - MAE: {mae_geral_test:.2f} | WMAE: {wmape_geral_test:.4f}")
    print("=" * 80)
    
    return (
        abc_models,
        y_train.values,
        y_pred_train,
        y_test.values,
        y_pred_test
    )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_estrutura = Helpers.getEntityData(context, 'df_estrutura_produto')
df_estrutura.columns = ["Codigo", "Componente", "Quantidade"] 
df_sku_abc_xyz = Helpers.getEntityData(context, 'sku_abc_xyz')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_portalvendas = Helpers.getEntityData(context, 'portal_vendas')

print("Shape do DataFrame:", df_portalvendas.shape)
print("Primeiras linhas:")
print(df_portalvendas.head())
print("Colunas:", df_portalvendas.columns.tolist())

if 'Msg_' in df_portalvendas.columns[0]:
    raise ValueError("Arquivo portal_vendas contém mensagens de erro SQL ao invés de dados. Por favor, re-exporte o arquivo corretamente.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
lista_dfs_removidos = []

row_tail = df_portalvendas.tail(1).copy()
row_tail['motivo'] = 'Linha de rodape/invalida'

lista_dfs_removidos.append(row_tail)

df_portalvendas = df_portalvendas.drop(df_portalvendas.tail(1).index)

df_portalvendas.columns = (
    df_portalvendas.columns.str.replace(" ", "_")
    .str.replace("._", "_", regex=False)
    .str.replace("_R\$", "", regex=True)
)
print(df_portalvendas.columns)

dict_columns_types = {
    "NR_DO_PEDIDO_MTE": "object",
    "DATA_PEDIDO": "datetime64[ns]",
    "CODIGO_MTE": "category",
    "QTDE_PEDIDA": "float64",
    "QTDE_ENTREGUE": "float64",
    "QTDE_SALDO": "float64",
    "VALOR_UNITARIO": "float64",
    "VALOR_FATURADO": "float64",
    "VALOR_SALDO": "float64",
    "VALOR_PEDIDO": "float64",
    "NR_PEDIDO_CLIENTE": "object",
    "NOTA_FISCAL": "object",
    "DATA_NOTA_FISCAL": "datetime64[ns]",
    "CODIGO_REPRESENTANTE": "category",
    "CODIGO_MERCADO": "category",
    "CODIGO_CLIENTE": "category",
    "COD_MTE_COMP": "category",
    "CODIGO_MTE_ORIG_EXP": "category",
}

print("Before transformation")

df_portalvendas = df_portalvendas.astype(dict_columns_types)
print("After transformation")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
filter_last_2years = False

if filter_last_2years:
    last_date = df_portalvendas['DATA_PEDIDO'].max()
    last_date_first_day_of_month = pd.Timestamp(f"{last_date.year}-{last_date.month:02d}-01")
    start_date = last_date_first_day_of_month - pd.DateOffset(years=2)
    
    mask_antigos = df_portalvendas['DATA_PEDIDO'] < start_date
    df_sem_match_antigo = df_portalvendas[mask_antigos].copy()
    df_sem_match_antigo['motivo'] = 'Dados antigos (> 2 anos)'
    lista_dfs_removidos.append(df_sem_match_antigo)
    
    mask_mes_incompleto = df_portalvendas['DATA_PEDIDO'] >= last_date_first_day_of_month
    df_sem_match_futuro = df_portalvendas[mask_mes_incompleto].copy()
    df_sem_match_futuro['motivo'] = 'Mes corrente incompleto'
    lista_dfs_removidos.append(df_sem_match_futuro)
    
    df_portalvendas = df_portalvendas[
        (df_portalvendas['DATA_PEDIDO'] >= start_date) & 
        (df_portalvendas['DATA_PEDIDO'] < last_date_first_day_of_month)
    ]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
set_components_all = set(df_estrutura["Componente"].unique())
set_components_imported = set(df_estrutura[df_estrutura["Componente"].str.startswith("T")]["Componente"].unique())

set_products_use_imported = set()
for component in set_components_imported:
    set_products_use_imported = set_products_use_imported.union(
        set(df_estrutura[df_estrutura["Componente"] == component]["Codigo"].unique())
    )

set_products_all = set()
for component in set_components_all:
    set_products_all = set_products_all.union(
        set(df_estrutura[df_estrutura["Componente"] == component]["Codigo"].unique())
    )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def refresh_categories(df, column_name):
    df.loc[:, column_name] = df[column_name].cat.set_categories(df[column_name].cat.remove_unused_categories().unique())
    return df

mask_nao_importados = ~df_portalvendas['COD_MTE_COMP'].isin(set_products_use_imported)

df_sem_match_skus = df_portalvendas[mask_nao_importados].copy()
df_sem_match_skus['motivo'] = 'Nao utiliza componentes importados (T)'
lista_dfs_removidos.append(df_sem_match_skus)

df_portalvendas_filtered = df_portalvendas[~mask_nao_importados].copy()
df_portalvendas_filtered = refresh_categories(df_portalvendas_filtered, 'COD_MTE_COMP')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_sem_match = pd.concat(lista_dfs_removidos, ignore_index=True)

df_sem_match['origem'] = 'portal_vendas (script: New 01 Process ABCXYZ)'

df_sem_match.rename(columns={'COD_MTE_COMP': 'Cod_component'}, inplace=True)

df_sem_match = df_sem_match[['Cod_component', 'origem', 'motivo']]
Helpers.save_output_dataset(context=context, output_name='df_sem_match_atual_2', data_frame=df_sem_match)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_portalvendas_filtered = df_portalvendas_filtered[['COD_MTE_COMP', 'DATA_PEDIDO', 'QTDE_PEDIDA', 'VALOR_UNITARIO']]
df_preprocessed = df_portalvendas_filtered.copy()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_preprocessed['YEAR_MONTH'] = df_preprocessed['DATA_PEDIDO'].dt.strftime('%Y-%m')
df_preprocessed['DATE_MONTH'] = df_preprocessed['YEAR_MONTH'].astype(str) + '-01'
df_preprocessed['DATE_MONTH'] = pd.to_datetime(df_preprocessed['DATE_MONTH'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_agg = df_preprocessed[['COD_MTE_COMP','DATE_MONTH','QTDE_PEDIDA']].copy()
df_agg = df_agg.groupby(['COD_MTE_COMP','DATE_MONTH']).sum().reset_index()
df_agg = df_agg.set_index(['DATE_MONTH','COD_MTE_COMP']).unstack(['COD_MTE_COMP']).fillna(0).stack().reset_index()
df_agg = df_agg.sort_values(by=['COD_MTE_COMP','DATE_MONTH'], ascending=True).reset_index(drop=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_agg_abcxyz = pd.merge(df_agg, df_sku_abc_xyz[['SKU_ID', 'ABC_XYZ']], left_on='COD_MTE_COMP', right_on='SKU_ID', how='left')
df_agg_abcxyz.drop(columns=['SKU_ID'], inplace=True)
df_agg_abcxyz['ABC_XYZ'].fillna('CZ', inplace=True)

print("Distribuição ABC_XYZ:")
print(df_agg_abcxyz[['COD_MTE_COMP', 'ABC_XYZ']].drop_duplicates()['ABC_XYZ'].value_counts())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def create_train_dataset(df_agg):
    """
    Cria dataset de treino com features de lag, diff, rolling e FERIADOS CHINESES.
    """
    df_agg['YEAR'] = df_agg['DATE_MONTH'].dt.year
    df_agg['MONTH'] = df_agg['DATE_MONTH'].dt.month
    
    unique_dates = df_agg[['YEAR', 'MONTH']].drop_duplicates()
    holiday_map = {}
    
    print("Calculando feriados chineses para o histórico...")
    for _, row in unique_dates.iterrows():
        y, m = int(row['YEAR']), int(row['MONTH'])
        cn_holidays = holidays.China(years=y)
        count = sum(1 for date in cn_holidays if date.month == m)
        holiday_map[(y, m)] = count
    
    df_agg['chinese_holidays_count'] = df_agg.set_index(['YEAR', 'MONTH']).index.map(holiday_map)

    df_agg['month_of_sequence'] = df_agg.groupby(['COD_MTE_COMP'])['DATE_MONTH'].rank(method='dense')
    df_agg['month_of_sequence'] = df_agg['month_of_sequence'].astype(np.int64)
    
    for lag in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        df_agg[f"lag_{lag}"] = df_agg.groupby("COD_MTE_COMP")["QTDE_PEDIDA"].shift(lag)
    
    for diff in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        df_agg[f"diff_{diff}"] = df_agg.groupby("COD_MTE_COMP")["QTDE_PEDIDA"].diff(diff)
    
    for roll in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        shifted_series = df_agg.groupby("COD_MTE_COMP")["QTDE_PEDIDA"].shift(1)
        
        df_agg[f"roll_mean_{roll}"] = (shifted_series.rolling(roll, min_periods=1).mean().reset_index(level=0, drop=True))
        df_agg[f"roll_std_{roll}"] = (shifted_series.rolling(roll, min_periods=1).std().reset_index(level=0, drop=True))
        df_agg[f"roll_min_{roll}"] = (shifted_series.rolling(roll, min_periods=1).min().reset_index(level=0, drop=True))
        df_agg[f"roll_max_{roll}"] = (shifted_series.rolling(roll, min_periods=1).max().reset_index(level=0, drop=True))
    
    rev_type_series = df_agg['COD_MTE_COMP']
    df_agg = pd.get_dummies(df_agg, columns=['COD_MTE_COMP'])
    df_agg['COD_MTE_COMP'] = rev_type_series
    
    df_agg = df_agg.groupby('COD_MTE_COMP').apply(lambda group: group.bfill()).reset_index(drop=True)
    
    horizons = [1]
    for horizon in horizons:
        df_agg[f"sales_next_{horizon}_month"] = df_agg.groupby("COD_MTE_COMP")["QTDE_PEDIDA"].shift(-horizon, fill_value=np.nan)
    
    df_agg['is_next_month_carnival'] = np.where(df_agg['MONTH'] == 2, 1, 0)
    
    return df_agg

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def create_future_dataset(df_agg):
    future_period = 12
    future_date = df_agg['DATE_MONTH'] + pd.DateOffset(months=future_period)
    future_date = future_date[-future_period:]
    
    category_list = df_agg['COD_MTE_COMP'].unique().tolist()
    
    df_future_0 = pd.DataFrame({'DATE_MONTH': future_date}).reset_index(drop=True)
    
    df_future = pd.DataFrame()
    for category in category_list:
        this_df_future = df_future_0.copy()
        this_df_future['COD_MTE_COMP'] = category
        df_future = pd.concat([df_future, this_df_future])
    
    df_future['QTDE_PEDIDA'] = -999
    df_future = df_future.reset_index(drop=True)
    
    return df_future

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
USE_EXISTING_ABC_XYZ = False
USE_ABC_PARETO = True

if USE_ABC_PARETO and USE_EXISTING_ABC_XYZ:
    raise ValueError("Escolha apenas um método: USE_EXISTING_ABC_XYZ ou USE_ABC_PARETO")

print("="*80)
print("🚀 INICIANDO TREINAMENTO DE MODELOS")
print("="*80)

if USE_ABC_PARETO:
    print("📊 Método: Classificação ABC-Pareto dinâmica (RECOMENDADO)")
    print("    - Classes definidas por volume acumulado (80-95-100%)")
    print()
    
    df_train_full = create_train_dataset(df_agg_abcxyz.drop(columns=['ABC_XYZ']))
    
    n_months_test = 3
    split_point = df_train_full['DATE_MONTH'].max() - pd.DateOffset(months=n_months_test)
    
    df_train_processed = df_train_full[df_train_full['DATE_MONTH'] < split_point].copy()
    df_test_processed = df_train_full[df_train_full['DATE_MONTH'] >= split_point].copy()
    
    target_col = 'sales_next_1_month' if 'sales_next_1_month' in df_train_processed.columns else 'sales_next_month'
    
    non_feature_cols = ['DATE_MONTH', 'COD_MTE_COMP', target_col]
    features = [col for col in df_train_processed.columns if col not in non_feature_cols]
    
    grid_A = {
        "n_estimators": [300, 500],
        "max_depth": [4, 5, 6],                
        "learning_rate": [0.03, 0.05],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
        "min_child_weight": [10, 15, 20],
        "gamma": [1, 2],
        "reg_alpha": [0.5, 1],
        "reg_lambda": [0.5, 1]
    }
    
    grid_B = {
        "n_estimators": [400, 500, 600],
        "max_depth": [5, 6, 7, 8],           
        "learning_rate": [0.03],             
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
        "min_child_weight": [5, 10, 15],
        "gamma": [0.5, 1],
        "reg_alpha": [0.1, 0.5, 1.0],
        "reg_lambda": [0.1, 0.5, 1.0]
    }
    
    grid_C = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.7, 0.8]
    }
    
    user_param_grid = {'A': grid_A, 'B': grid_B, 'C': grid_C}
    
    model_dict, y_train, y_pred_train, y_test, y_pred_test = train_abc_models(
        df_train_processed,
        df_test_processed,
        features,
        user_param_grid=user_param_grid,
        n_iter_search=30
    )
    
elif USE_EXISTING_ABC_XYZ:
    print("📊 Método: Classificação ABC-XYZ pré-existente")
    print("    - Usando coluna 'ABC_XYZ' do dataset sku_abc_xyz")
    print()
    
    model_dict = {}
    
    for cluster in df_agg_abcxyz['ABC_XYZ'].unique():
        print(f"\n🎯 Treinando modelo para cluster: {cluster}")
        df_cluster = df_agg_abcxyz[df_agg_abcxyz['ABC_XYZ']==cluster].copy()
        df_cluster_copy = df_cluster.drop(columns=['ABC_XYZ'])
        
        df_train_cluster = create_train_dataset(df_cluster_copy)
        
        df_train_cluster = df_train_cluster[df_train_cluster['sales_next_1_month'].notna()]
        
        non_feature_cols = ['DATE_MONTH','COD_MTE_COMP', 'sales_next_1_month']
        
        X_train = df_train_cluster.drop(non_feature_cols, axis=1)
        y_train = df_train_cluster['sales_next_1_month']
        
        feature_names = X_train.columns.tolist()
        
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, max_depth=10, learning_rate=0.03)
        model.fit(X_train, y_train)
        
        model_dict[cluster] = {
            'model': model,
            'features': feature_names
        }
        
        print(f"✅ Modelo treinado para cluster {cluster} - Features: {len(feature_names)}")

else:
    raise ValueError("Você deve escolher um método de treinamento")

print("\n" + "="*80)
print("✅ TREINAMENTO CONCLUÍDO")
print("="*80)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_features(df_agg, cutoff_date):
    """
    Gera features para um cutoff_date específico.
    """
    df_agg_v1 = df_agg.copy()
    target_col = 'sales_next_1_month'
    
    df_agg_v1['YEAR'] = df_agg_v1['DATE_MONTH'].dt.year
    df_agg_v1['MONTH'] = df_agg_v1['DATE_MONTH'].dt.month
    
    cutoff_ts = pd.to_datetime(cutoff_date)
    
    y = cutoff_ts.year
    m = cutoff_ts.month
    
    cn_holidays = holidays.China(years=y)
    cn_count = sum(1 for date in cn_holidays if date.month == m)
    
    df_agg_v1['chinese_holidays_count'] = cn_count
    
    df_agg_v1['month_of_sequence'] = df_agg_v1.groupby(['COD_MTE_COMP'])['DATE_MONTH'].rank(method='dense')
    df_agg_v1['month_of_sequence'] = df_agg_v1['month_of_sequence'].astype(np.int64)
    
    for lag in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        df_agg_v1[f"lag_{lag}"] = df_agg_v1.groupby("COD_MTE_COMP")["QTDE_PEDIDA"].shift(lag)
    
    for diff in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        df_agg_v1[f"diff_{diff}"] = df_agg_v1.groupby("COD_MTE_COMP")["QTDE_PEDIDA"].diff(diff)

    for roll in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        shifted_series = df_agg_v1.groupby("COD_MTE_COMP")["QTDE_PEDIDA"].shift(1)
        
        df_agg_v1[f"roll_mean_{roll}"] = (shifted_series.rolling(roll, min_periods=1).mean().reset_index(level=0, drop=True))
        df_agg_v1[f"roll_std_{roll}"] = (shifted_series.rolling(roll, min_periods=1).std().reset_index(level=0, drop=True))
        df_agg_v1[f"roll_min_{roll}"] = (shifted_series.rolling(roll, min_periods=1).min().reset_index(level=0, drop=True))
        df_agg_v1[f"roll_max_{roll}"] = (shifted_series.rolling(roll, min_periods=1).max().reset_index(level=0, drop=True))
    
    rev_type_series = df_agg_v1['COD_MTE_COMP']
    df_agg_v1 = pd.get_dummies(df_agg_v1, columns=['COD_MTE_COMP'])
    df_agg_v1['COD_MTE_COMP'] = rev_type_series    
    
    df_agg_v1 = df_agg_v1.groupby('COD_MTE_COMP').apply(lambda group: group.bfill())
    
    df_agg_v1 = df_agg_v1[df_agg_v1['DATE_MONTH'] == cutoff_ts]
    df_agg_v1[target_col] = None
    
    df_agg_v1['is_next_month_carnival'] = np.where(df_agg_v1['MONTH'] == 2, 1, 0)
    
    return df_agg_v1

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def recursive_prediction(df_agg, df_future, model_dict, use_abc_xyz_column=False):
    """
    Previsão recursiva usando predict_ensemble
    """
    target_col = 'sales_next_1_month'
    non_feature_cols = ['DATE_MONTH', 'COD_MTE_COMP', target_col]
    
    df_test_next_1_month = df_agg[df_agg['sales_next_1_month'].isna()].copy()
    
    if use_abc_xyz_column and 'ABC_XYZ' not in df_test_next_1_month.columns:
        df_test_next_1_month = pd.merge(
            df_test_next_1_month, 
            df_agg_abcxyz[['COD_MTE_COMP', 'ABC_XYZ']].drop_duplicates(), 
            on='COD_MTE_COMP', 
            how='left'
        )
    
    y_pred = predict_ensemble(model_dict, df_test_next_1_month)
    
    df_future = df_future.sort_values(by=['DATE_MONTH','COD_MTE_COMP']).reset_index(drop=True)
    future_date_list = df_future['DATE_MONTH'].unique()
    
    df_agg_future = df_agg[['DATE_MONTH', 'COD_MTE_COMP', 'QTDE_PEDIDA']].copy()
    
    this_y_pred = y_pred.copy()
    
    for i in range(len(future_date_list)):
        print(f"Predicting for period: {future_date_list[i]}")
        
        this_df_future = df_future[df_future['DATE_MONTH'] == future_date_list[i]].copy()
        this_df_future['QTDE_PEDIDA'] = this_y_pred
        this_df_future['QTDE_PEDIDA'] = np.where(this_df_future['QTDE_PEDIDA'] <= 0, 0, this_df_future['QTDE_PEDIDA'])
        
        df_agg_future = pd.concat([df_agg_future, this_df_future])
        df_agg_future = df_agg_future.sort_values(by=['COD_MTE_COMP', 'DATE_MONTH']).reset_index(drop=True)
        
        this_df_future_w_feature = get_features(df_agg_future, future_date_list[i])
        
        if use_abc_xyz_column and 'ABC_XYZ' not in this_df_future_w_feature.columns:
            this_df_future_w_feature = pd.merge(
                this_df_future_w_feature, 
                df_agg_abcxyz[['COD_MTE_COMP', 'ABC_XYZ']].drop_duplicates(), 
                on='COD_MTE_COMP', 
                how='left'
            )
        
        this_y_pred = predict_ensemble(model_dict, this_df_future_w_feature)
    
    return df_agg_future

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_all_predictions = pd.DataFrame()

print("="*80)
print("🚀 GERANDO PREVISÕES RECURSIVAS")
print("="*80)

if USE_ABC_PARETO:
    print("\n📊 Usando modelo ABC-Pareto unificado...")
    
    df_fe = df_agg_abcxyz.drop(columns=['ABC_XYZ'] if 'ABC_XYZ' in df_agg_abcxyz.columns else [])
    df_train = create_train_dataset(df_fe)
    df_future = create_future_dataset(df_train)
    
    df_predictions = recursive_prediction(df_train, df_future, model_dict, use_abc_xyz_column=False)
    df_all_predictions = df_predictions.copy()
    
    print(f"✅ Previsões geradas: {len(df_all_predictions)} registros")
    
elif USE_EXISTING_ABC_XYZ:
    for cluster in df_agg_abcxyz['ABC_XYZ'].unique():
        print(f"\n🔮 Gerando previsões para cluster: {cluster}")
        df_fe_cluster = df_agg_abcxyz[df_agg_abcxyz['ABC_XYZ']==cluster].copy()
        df_fe_cluster = df_fe_cluster.drop(columns=['ABC_XYZ'])
        
        df_train_cluster = create_train_dataset(df_fe_cluster)
        df_future_cluster = create_future_dataset(df_train_cluster)
        
        df_cluster_predictions = recursive_prediction(df_train_cluster, df_future_cluster, model_dict, use_abc_xyz_column=True)
        
        df_all_predictions = pd.concat([df_all_predictions, df_cluster_predictions])
        
        print(f"✅ Previsões geradas para cluster {cluster}: {len(df_cluster_predictions)} registros")

print("\n" + "="*80)
print("✅ TODAS AS PREVISÕES CONCLUÍDAS")
print("="*80)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_multi_horizon_pred = df_all_predictions.copy()
df_multi_horizon_pred.rename(columns={'DATE_MONTH': 'DATA_PEDIDO'}, inplace=True)
df_multi_horizon_pred['base_date'] = df_agg_abcxyz['DATE_MONTH'].max()

# Adicionar classificações ABC-XYZ aos PRODUTOS primeiro
if USE_ABC_PARETO and 'product_class_map' in model_dict:
    print("\n📊 Adicionando classificações ABC-XYZ aos produtos...")
    
    product_class_map = model_dict['product_class_map']
    product_xyz_map = model_dict.get('product_xyz_map', {})
    product_abc_xyz_map = model_dict.get('product_abc_xyz_map', {})
    
    df_multi_horizon_pred['classe_abc'] = df_multi_horizon_pred['COD_MTE_COMP'].map(product_class_map).fillna('C')
    df_multi_horizon_pred['classe_xyz'] = df_multi_horizon_pred['COD_MTE_COMP'].map(product_xyz_map).fillna('Z')
    df_multi_horizon_pred['classe_abc_xyz'] = df_multi_horizon_pred['COD_MTE_COMP'].map(product_abc_xyz_map).fillna('CZ')
    
    print(f"   ✅ Classificações adicionadas")
    
elif USE_EXISTING_ABC_XYZ:
    print("\n📊 Usando classificação ABC-XYZ original...")
    df_multi_horizon_pred = pd.merge(
        df_multi_horizon_pred,
        df_agg_abcxyz[['COD_MTE_COMP', 'ABC_XYZ']].drop_duplicates(),
        on='COD_MTE_COMP',
        how='left'
    )
    df_multi_horizon_pred['ABC_XYZ'] = df_multi_horizon_pred['ABC_XYZ'].fillna('CZ')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_product_componentes(end_prod_cod, df_estrutura):
    '''Returns a dictionary with the components of the input end product and their respective quantities'''
    dict_components = {}
    components = df_estrutura[df_estrutura["Codigo"] == end_prod_cod][["Componente", "Quantidade"]].values
    for component in components:
        dict_components[component[0]] = component[1]
    return dict_components

def create_component_consumption_dataframe(df,
                                           df_estrutura,
                                           sales_column = "sales_next_month",
                                           consumption_column = "consumption_next_month", 
                                           code_column = "COD_MTE_COMP", 
                                           period_column="_next_month",
                                           other_columns_to_keep = []):
    '''Takes a dataframe with end product sales for each product and month and returns a dataframe with the consumptions for each component for each month'''
    df_components = pd.DataFrame()
    
    # Filtrar apenas colunas necessárias (SEM classificações no groupby)
    base_columns_needed = [period_column, sales_column]
    
    # Se base_date está em other_columns_to_keep, adicionar
    if 'base_date' in other_columns_to_keep:
        base_columns_needed.append('base_date')
    
    for end_prod_cod in df[code_column].unique():
        dict_components = get_product_componentes(end_prod_cod, df_estrutura)
        for component_cod, component_quantity in dict_components.items():
            new_row = pd.DataFrame()
            new_row = df[df[code_column] == end_prod_cod].copy()[base_columns_needed]
            new_row["Component"] = component_cod
            new_row[consumption_column] = new_row[sales_column] * component_quantity
            df_components = pd.concat([df_components, new_row], axis=0)
    
    df_components = df_components.drop(columns=[sales_column])
    
    # CORREÇÃO CRÍTICA: Groupby APENAS por período e componente (sem classificações)
    groupby_cols = [period_column, "Component"]
    if 'base_date' in df_components.columns:
        groupby_cols.append('base_date')
    
    df_components = df_components.groupby(groupby_cols)[consumption_column].sum().reset_index()
    
    return df_components

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Apenas base_date é necessário no groupby
other_columns_to_keep = ['base_date']

df_multi_horizon_pred_component = create_component_consumption_dataframe(
    df_multi_horizon_pred,
    df_estrutura,
    'QTDE_PEDIDA',
    "consumption_predicted_month",
    code_column='COD_MTE_COMP',
    period_column='DATA_PEDIDO',
    other_columns_to_keep=other_columns_to_keep
)

# Agora adicionar classificações APÓS a agregação (1 classificação por componente)
if USE_ABC_PARETO and 'product_class_map' in model_dict:
    print("\n📊 Adicionando classificações ABC-XYZ aos componentes...")
    
    product_class_map = model_dict['product_class_map']
    product_xyz_map = model_dict.get('product_xyz_map', {})
    
    # Criar mapeamento de componente -> produtos que o utilizam
    component_to_products = {}
    for _, row in df_estrutura.iterrows():
        produto = row['Codigo']
        componente = row['Componente']
        
        if componente not in component_to_products:
            component_to_products[componente] = []
        component_to_products[componente].append(produto)
    
    def get_best_classification(componente, product_map, priority_order):
        """
        Retorna a MELHOR classificação entre os produtos que usam o componente.
        Prioriza: A > B > C (para ABC) e X > Y > Z (para XYZ)
        """
        if componente not in component_to_products:
            return priority_order[-1]  # Retorna pior classe se não encontrado
        
        produtos = component_to_products[componente]
        classes = [product_map.get(p) for p in produtos if p in product_map]
        
        if not classes:
            return priority_order[-1]
        
        # Retornar primeira classe na ordem de prioridade
        for priority_class in priority_order:
            if priority_class in classes:
                return priority_class
        
        return priority_order[-1]
    
    # Criar mapeamento único de componente -> classificação
    component_abc_map = {
        comp: get_best_classification(comp, product_class_map, ['A', 'B', 'C'])
        for comp in df_multi_horizon_pred_component['Component'].unique()
    }
    
    component_xyz_map = {
        comp: get_best_classification(comp, product_xyz_map, ['X', 'Y', 'Z'])
        for comp in df_multi_horizon_pred_component['Component'].unique()
    }
    
    # Adicionar classificações (UMA por componente)
    df_multi_horizon_pred_component['classe_abc'] = df_multi_horizon_pred_component['Component'].map(component_abc_map)
    df_multi_horizon_pred_component['classe_xyz'] = df_multi_horizon_pred_component['Component'].map(component_xyz_map)
    df_multi_horizon_pred_component['classe_abc_xyz'] = df_multi_horizon_pred_component['classe_abc'] + df_multi_horizon_pred_component['classe_xyz']
    
    print(f"   ✅ Classificações únicas adicionadas aos componentes")
    print(f"   Total de componentes: {df_multi_horizon_pred_component['Component'].nunique()}")
    print(f"   Linhas totais: {len(df_multi_horizon_pred_component)}")
    
elif USE_EXISTING_ABC_XYZ:
    print("\n📊 Usando classificação ABC-XYZ original para componentes...")
    # Para ABC-XYZ existente, usar a mesma lógica de herdar do melhor produto
    df_multi_horizon_pred_component = pd.merge(
        df_multi_horizon_pred_component,
        df_agg_abcxyz[['COD_MTE_COMP', 'ABC_XYZ']].drop_duplicates(),
        left_on='Component',
        right_on='COD_MTE_COMP',
        how='left'
    )
    df_multi_horizon_pred_component['ABC_XYZ'] = df_multi_horizon_pred_component['ABC_XYZ'].fillna('CZ')
    if 'COD_MTE_COMP' in df_multi_horizon_pred_component.columns:
        df_multi_horizon_pred_component.drop(columns=['COD_MTE_COMP'], inplace=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_multi_horizon_pred['QTDE_PEDIDA'] = np.ceil(df_multi_horizon_pred['QTDE_PEDIDA'])
df_multi_horizon_pred_component['consumption_predicted_month'] = np.ceil(df_multi_horizon_pred_component['consumption_predicted_month'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("\n" + "="*80)
print("💾 SALVANDO OUTPUTS")
print("="*80)

# VERIFICAÇÃO DE QUALIDADE: Checar duplicatas
print("\n🔍 VERIFICAÇÃO DE QUALIDADE:")
duplicatas_componentes = df_multi_horizon_pred_component.groupby(['DATA_PEDIDO', 'Component']).size()
duplicatas_encontradas = duplicatas_componentes[duplicatas_componentes > 1]

if len(duplicatas_encontradas) > 0:
    print(f"   ⚠️ ATENÇÃO: {len(duplicatas_encontradas)} combinações (DATA_PEDIDO, Component) com duplicatas!")
    print(f"   Exemplos:")
    print(duplicatas_encontradas.head(5))
else:
    print(f"   ✅ Nenhuma duplicata encontrada!")
    print(f"   Cada (DATA_PEDIDO, Component) tem exatamente 1 linha")

print(f"\n📊 Estatísticas Finais:")
print(f"   Componentes: {len(df_multi_horizon_pred_component)} linhas")
print(f"   Componentes únicos: {df_multi_horizon_pred_component['Component'].nunique()}")
print(f"   Datas únicas: {df_multi_horizon_pred_component['DATA_PEDIDO'].nunique()}")
print(f"   Linhas esperadas: {df_multi_horizon_pred_component['Component'].nunique() * df_multi_horizon_pred_component['DATA_PEDIDO'].nunique()}")

Helpers.save_output_dataset(context=context, output_name='new_multihorizon_products_abcxyz', data_frame=df_multi_horizon_pred)
Helpers.save_output_dataset(context=context, output_name='new_multihorizon_components_abcxyz', data_frame=df_multi_horizon_pred_component)

print("✅ Outputs salvos com sucesso!")
print("   - new_multihorizon_products_abcxyz")
print("   - new_multihorizon_components_abcxyz")

print("\n" + "="*80)
print("🎉 SCRIPT INTEGRADO CONCLUÍDO COM SUCESSO!")
print("="*80)
print(f"\n📊 RESUMO:")
print(f"   Método usado: {'ABC-Pareto + XYZ (Volatilidade)' if USE_ABC_PARETO else 'ABC-XYZ pré-existente'}")
print(f"   Produtos previstos: {df_multi_horizon_pred['COD_MTE_COMP'].nunique()}")
print(f"   Componentes previstos: {df_multi_horizon_pred_component['Component'].nunique()}")
print(f"   Período de previsão: {df_multi_horizon_pred['DATA_PEDIDO'].min()} até {df_multi_horizon_pred['DATA_PEDIDO'].max()}")

if USE_ABC_PARETO and 'classe_abc_xyz' in df_multi_horizon_pred.columns:
    print(f"\n📈 CLASSIFICAÇÃO DOS PRODUTOS:")
    print(f"   ABC (Volume):")
    for classe in ['A', 'B', 'C']:
        count = (df_multi_horizon_pred['classe_abc'] == classe).sum()
        produtos = df_multi_horizon_pred[df_multi_horizon_pred['classe_abc'] == classe]['COD_MTE_COMP'].nunique()
        print(f"      Classe {classe}: {produtos} produtos ({count} registros)")
    
    print(f"\n   XYZ (Volatilidade):")
    for classe in ['X', 'Y', 'Z']:
        count = (df_multi_horizon_pred['classe_xyz'] == classe).sum()
        produtos = df_multi_horizon_pred[df_multi_horizon_pred['classe_xyz'] == classe]['COD_MTE_COMP'].nunique()
        print(f"      Classe {classe}: {produtos} produtos ({count} registros)")
    
    print(f"\n   Top 5 Combinações ABC-XYZ:")
    top_combinations = df_multi_horizon_pred.groupby('classe_abc_xyz')['COD_MTE_COMP'].nunique().sort_values(ascending=False).head(5)
    for abc_xyz, produtos in top_combinations.items():
        count = (df_multi_horizon_pred['classe_abc_xyz'] == abc_xyz).sum()
        print(f"      {abc_xyz}: {produtos} produtos ({count} registros)")

print(f"\n💾 OUTPUTS GERADOS:")
print(f"   ✅ new_multihorizon_products_abcxyz")
print(f"   ✅ new_multihorizon_components_abcxyz")
print("="*80)

