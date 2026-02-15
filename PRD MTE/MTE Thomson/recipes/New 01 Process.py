# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Importações proprietárias e configuração do contexto
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
# Importações padrão
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
from importlib import reload
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    make_scorer, mean_absolute_error, mean_squared_error, 
    r2_score, mean_absolute_percentage_error, median_absolute_error
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Funções auxiliares originais (mantidas)
def remove_letters_in_beginning_of_string(string):
    return re.sub(r"^[a-z A-Z]+", "", string)

def remove_letters_in_end_of_string(string):
    return re.sub(r"[a-z A-Z]+$", "", string)

def remove_letters_from_string(string):
    return remove_letters_in_beginning_of_string(remove_letters_in_end_of_string(string))

def get_metrics(df_actual, df_prediction, index=None):
    """Get the metrics for the predictions"""
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

def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred))/np.sum(np.abs(y_true))

def mape(y_true, y_pred):
    ape = np.abs(y_true - y_pred)/y_true
    
    if np.isscalar(ape):
        if np.isfinite(ape):
            return ape
        else:
            return 1
    else:
        ape[~np.isfinite(ape)] = 1
    return np.mean(ape)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# MODIFICADO: predict_ensemble
# Esta função agora lida com o novo formato de modelo ABC (que é XGBoost)
# ou um modelo XGBoost legado.
# ==================================================================================
def predict_ensemble(ensemble_model, df_features):
    """
    Faz previsões usando o modelo (ou modelos) treinado.
    Detecta automaticamente se é um ensemble ABC ou um modelo legado.
    
    Parameters:
    -----------
    ensemble_model : dict
        Dicionário com modelos treinados
    df_features : DataFrame
        Features para fazer previsão
        
    Returns:
    --------
    predictions : numpy array
        Previsões do modelo
    """
    
    # Verificar se temos as features necessárias
    if 'feature_names' not in ensemble_model:
        raise ValueError("ensemble_model não tem 'feature_names'")
    
    features = ensemble_model['feature_names']
    
    # Verificar se todas as features estão no dataframe
    missing_features = set(features) - set(df_features.columns)
    if missing_features:
        raise ValueError(f"Features faltando no dataframe: {missing_features}")
        
    # Preparar dados de features
    X = df_features[features].fillna(0).astype('float64')

    # ============================================================================
    # LÓGICA 1: NOVO MODELO ABC (um modelo XGBoost por classe)
    # ============================================================================
    if 'product_class_map' in ensemble_model:
        # print("     -> Aplicando modelos ABC...")
        product_class_map = ensemble_model['product_class_map']
        
        if COD_MTE_COMP not in df_features.columns:
            raise ValueError(f"{COD_MTE_COMP} não encontrado. É necessário para roteamento ABC.")

        # Classificar linhas de entrada
        df_features['classe_abc'] = df_features[COD_MTE_COMP].map(product_class_map).fillna('C')
        
        predictions = np.zeros(len(df_features))
        
        for classe in ['A', 'B', 'C']:
            mask = (df_features['classe_abc'] == classe)
            if mask.sum() == 0:
                continue
            
            X_classe = X.loc[mask]
            if X_classe.empty:
                continue
                
            model_classe = ensemble_model['models'][classe]
            
            # Se a classe não teve modelo (ex: sem dados), prevê 0
            if model_classe is None:
                predictions[mask.values] = 0.0
            else:
                pred_classe = model_classe.predict(X_classe)
                predictions[mask.values] = pred_classe
            
        return np.maximum(predictions, 0) # Garantir não-negatividade

    # ============================================================================
    # LÓGICA 2: MODELO LEGADO (um único XGBoost)
    # ============================================================================
    else:
        # print("     -> Aplicando modelo único (legado)...")
        if 'xgboost' in ensemble_model['models']:
            predictions = ensemble_model['models']['xgboost'].predict(X.values)
            return np.maximum(predictions, 0) # Garantir não-negatividade
        else:
            raise ValueError("Modelo legado 'xgboost' não disponível no ensemble.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Funções de produto e componentes (mantidas do original)
def get_product_componentes(end_prod_cod, df_estrutura):
    '''Returns a dictionary with the components of the input end product and their respective quantities'''
    dict_components = {}
    components = df_estrutura[df_estrutura["Codigo"] == end_prod_cod][["Componente", "Quantidade"]].values
    for component in components:
        dict_components[component[0]] = component[1]
    return dict_components

def create_component_consumption_dataframe(df,
                                           df_estrutura,
                                           sales_column="sales_next_month",
                                           consumption_column="consumption_next_month",
                                           code_column="COD_MTE_COMP",
                                           period_column="_next_month",
                                           other_columns_to_keep=None):
    if other_columns_to_keep is None:
        other_columns_to_keep = []
    
    component_rows = []
    
    for end_prod_cod in df[code_column].unique():
        dict_components = get_product_componentes(end_prod_cod, df_estrutura)
        
        for component_cod, component_quantity in dict_components.items():
            new_row = df[df[code_column] == end_prod_cod].copy()[[period_column, sales_column] + other_columns_to_keep]
            new_row["Component"] = component_cod
            new_row[consumption_column] = new_row[sales_column] * component_quantity
            component_rows.append(new_row)
    
    df_components = pd.concat(component_rows, axis=0)
    df_components = df_components.drop(columns=[sales_column])
    
    # MUDANÇA: Agregar TODAS as colunas numéricas
    agg_dict = {consumption_column: 'sum'}
    for col in other_columns_to_keep:
        agg_dict[col] = 'sum'
    
    df_components = df_components.groupby([period_column, "Component"]).agg(agg_dict).reset_index()
    
    return df_components
    
def refresh_categories(df, column_name):
    df.loc[:, column_name] = df[column_name].cat.set_categories(df[column_name].cat.remove_unused_categories().unique())
    return df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# ETAPA 3: FUNÇÃO DE TREINO 
# ==================================================================================
def train_abc_models(df_train, df_test, features, user_param_grid=None, n_iter_search=50):
    """
    Treina um modelo XGBoost separado para cada classe ABC.
    (Versão 2: Usa TimeSeriesSplit com JANELA DESLIZANTE)
    """
    
    print("=" * 80)
    print("🚀 TREINAMENTO DE MODELOS XGBOOST SEPARADOS POR CLASSE ABC")
    print("=" * 80)
    
    if user_param_grid is None:
        user_param_grid = {}

    # ============================================================================
    # 1. CLASSIFICAÇÃO ABC (Baseada no volume total de treino por PRODUTO)
    # ============================================================================
    print("📊 Classificando produtos (ABC) com base no volume total de treino...")
    
    # Agrega vendas totais por produto no período de treino
    df_train_agg = df_train.groupby(COD_MTE_COMP)['QTDE_PEDIDA'].sum().reset_index()
    df_train_agg = df_train_agg.sort_values('QTDE_PEDIDA', ascending=False).reset_index(drop=True)
    
    # Calcular Pareto
    df_train_agg['cumsum'] = df_train_agg['QTDE_PEDIDA'].cumsum()
    total_sales = df_train_agg['QTDE_PEDIDA'].sum()
    df_train_agg['cumsum_pct'] = (df_train_agg['cumsum'] / total_sales) * 100
    
    def get_abc_class(cumsum_pct):
        if cumsum_pct <= 80: return 'A'
        elif cumsum_pct <= 95: return 'B'
        else: return 'C'
        
    df_train_agg['classe_abc'] = df_train_agg['cumsum_pct'].apply(get_abc_class)
    
    # Criar mapa de Produto -> Classe
    product_class_map = df_train_agg.set_index(COD_MTE_COMP)['classe_abc'].to_dict()
    
    # Mapear classes para os dataframes de treino e teste
    classes_train = df_train[COD_MTE_COMP].map(product_class_map).fillna('C').values
    classes_test = df_test[COD_MTE_COMP].map(product_class_map).fillna('C').values
    
    n_A = (classes_train == 'A').sum()
    n_B = (classes_train == 'B').sum()
    n_C = (classes_train == 'C').sum()
    
    print(f"📊 Distribuição de LINHAS de treino (produto-mês):")
    print(f"    Classe A: {n_A} linhas ({n_A/len(classes_train)*100:.1f}%)")
    print(f"    Classe B: {n_B} linhas ({n_B/len(classes_train)*100:.1f}%)")
    print(f"    Classe C: {n_C} linhas ({n_C/len(classes_train)*100:.1f}%)")
    print()

    # ============================================================================
    # 2. PREPARAR DADOS E MODELOS
    # ============================================================================
    X_train = df_train[features].fillna(0).astype('float64')
    y_train = df_train["sales_next_month"].fillna(0).astype('float64')
    X_test = df_test[features].fillna(0).astype('float64')
    y_test = df_test["sales_next_month"].fillna(0).astype('float64')

    abc_models = {
        'models': {}, 
        'feature_names': features, 
        'product_class_map': product_class_map
    }
    
    y_pred_train = np.zeros(len(y_train))
    y_pred_test = np.zeros(len(y_test))

    # Parâmetros default (robustos)
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

    # ============================================================================
    # 3. TREINAR MODELO PARA CADA CLASSE
    # ============================================================================
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
        
        # --- Lógica do GridSearch ---
        if classe in user_param_grid:
            print(f"    🔍 Executando RandomizedSearchCV para Classe {classe} (n_iter={n_iter_search})...")
            param_grid = user_param_grid[classe]
            
            base_model = xgb.XGBRegressor(n_jobs=-1, random_state=42)

            # Total de meses únicos no df de treino
            n_months_total = df_train.loc[mask_train, '_next_month'].nunique()

            n_splits_safe = 3 # Fixo em 3 para ser robusto
            train_window_size = 12 # 12 meses de treino
            
            print(f"    CV: Janela Deslizante (Treino={train_window_size} meses, n_splits={n_splits_safe})")

            tscv = TimeSeriesSplit(
                n_splits=n_splits_safe, 
                max_train_size=train_window_size,
                test_size=3 
            )
            # =====================================================================

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
            
        # --- Lógica de Treino Padrão ---
        else:
            print(f"    ⚙️  Treinando Classe {classe} com parâmetros padrão...")
            model_classe = xgb.XGBRegressor(**default_xgb_params)
            model_classe.fit(X_train_classe, y_train_classe)
            
        # Armazenar modelo
        abc_models['models'][classe] = model_classe
        
        # Prever no treino
        pred_train_classe = model_classe.predict(X_train_classe)
        y_pred_train[mask_train] = pred_train_classe
        
        # Prever no teste
        mask_test = (classes_test == classe)
        X_test_classe = X_test[mask_test]
        if not X_test_classe.empty:
            pred_test_classe = model_classe.predict(X_test_classe)
            y_pred_test[mask_test] = pred_test_classe
            
        # Métricas de treino da classe
        mae_train_classe = mean_absolute_error(y_train_classe, pred_train_classe)
        mape_train_classe = mape(y_train_classe, pred_train_classe)
        print(f"    📈 Classe {classe} (Treino) - MAE: {mae_train_classe:.2f} | MAPE: {mape_train_classe:.4f}")

    # ============================================================================
    # 4. MÉTRICAS FINAIS (COMBINADAS) - [WMAE ADICIONADO AQUI]
    # ============================================================================
    
    y_pred_train = np.maximum(y_pred_train, 0)
    y_pred_test = np.maximum(y_pred_test, 0)
    
    # Preencher NaNs que podem ter surgido de predições falhas
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
            # --- CÁLCULO WMAPE ADICIONADO ---
            wmape_train = wmape(y_train[mask_train], y_pred_train[mask_train])
            print(f"    TREINO Classe {classe} - MAE: {mae_train:.2f} | MAPE: {mape_train:.4f} | WMAE: {wmape_train:.4f}")
            
        if mask_test.sum() > 0:
            mae_test = mean_absolute_error(y_test[mask_test], y_pred_test[mask_test])
            mape_test = mape(y_test[mask_test], y_pred_test[mask_test])
            # --- CÁLCULO WMAPE ADICIONADO ---
            wmape_test = wmape(y_test[mask_test], y_pred_test[mask_test])
            print(f"    TESTE  Classe {classe} - MAE: {mae_test:.2f} | MAPE: {mape_test:.4f} | WMAE: {wmape_test:.4f}")
            
    print("-" * 40)
    mae_geral_train = mean_absolute_error(y_train, y_pred_train)
    mae_geral_test = mean_absolute_error(y_test, y_pred_test)
    # --- CÁLCULO WMAPE ADICIONADO ---
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
def analyze_by_class(y_true, y_pred, classes, sales_12m):
    """
    Análise detalhada de performance por classe ABC.
    """
    
    print("\n" + "=" * 80)
    print("📊 ANÁLISE DETALHADA POR CLASSE ABC")
    print("=" * 80)
    
    results = []
    
    for classe in ['A', 'B', 'C']:
        mask = (classes == classe)
        if mask.sum() == 0:
            continue
        
        y_true_classe = y_true[mask]
        y_pred_classe = y_pred[mask]
        sales_classe = sales_12m[mask]
        
        # Métricas
        mae = mean_absolute_error(y_true_classe, y_pred_classe)
        rmse = np.sqrt(np.mean((y_true_classe - y_pred_classe) ** 2))
        mape = np.mean(np.abs((y_true_classe - y_pred_classe) / np.maximum(y_true_classe, 1))) * 100
        bias = np.mean(y_pred_classe - y_true_classe)
        
        # Estatísticas
        n_produtos = mask.sum()
        pct_produtos = (n_produtos / len(y_true)) * 100
        vendas_total = sales_classe.sum()
        pct_vendas = (vendas_total / sales_12m.sum()) * 100
        
        # Acurácia por faixa
        errors_pct = np.abs((y_true_classe - y_pred_classe) / np.maximum(y_true_classe, 1)) * 100
        dentro_10 = (errors_pct <= 10).sum() / len(errors_pct) * 100
        dentro_20 = (errors_pct <= 20).sum() / len(errors_pct) * 100
        dentro_30 = (errors_pct <= 30).sum() / len(errors_pct) * 100
        
        result = {
            'Classe': classe,
            'N_Produtos': n_produtos,
            'Pct_Produtos': pct_produtos,
            'Pct_Vendas': pct_vendas,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Bias': bias,
            'Dentro_10pct': dentro_10,
            'Dentro_20pct': dentro_20,
            'Dentro_30pct': dentro_30
        }
        results.append(result)
        
        print(f"\n{'='*60}")
        print(f"CLASSE {classe}")
        print(f"{'='*60}")
        print(f"  Produtos: {n_produtos:,} ({pct_produtos:.1f}% do total)")
        print(f"  Vendas: {pct_vendas:.1f}% do total")
        print(f"\n  Métricas de Erro:")
        print(f"     MAE:  {mae:.2f}")
        print(f"     RMSE: {rmse:.2f}")
        print(f"     MAPE: {mape:.2f}%")
        print(f"     Bias: {bias:+.2f}")
        print(f"\n  Acurácia:")
        print(f"     Dentro de ±10%: {dentro_10:.1f}%")
        print(f"     Dentro de ±20%: {dentro_20:.1f}%")
        print(f"     Dentro de ±30%: {dentro_30:.1f}%")
    
    return pd.DataFrame(results)

def calcular_media_movel(df, group_col, date_col, value_col, window=3):
    """
    Calcula a média móvel para cada produto/componente.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame com dados observados
    group_col : str
        Coluna de agrupamento (produto ou componente)
    date_col : str
        Coluna de data
    value_col : str
        Coluna com valores a calcular média
    window : int
        Janela da média móvel (default: 3 meses)
    
    Returns:
    --------
    df_with_ma : DataFrame
        DataFrame com coluna adicional 'media_movel'
    """
    df_sorted = df.sort_values([group_col, date_col]).copy()
    
    # Calcular média móvel por grupo
    # A média móvel usa a janela (window) dos valores passados (shift(1))
    df_sorted['media_movel'] = df_sorted.groupby(group_col)[value_col].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    
    return df_sorted

def count_chinese_holidays_in_month(year, month):
    """
    Conta quantos dias de feriado chinês existem em um determinado mês/ano.
    Foca nos feriados oficiais da China (incluindo o Ano Novo Lunar e Golden Weeks).
    """
    # Instancia feriados da China para o ano específico
    cn_holidays = holidays.China(years=[year])
    
    # Filtra apenas os feriados que caem no mês específico
    holidays_in_month = [date for date in cn_holidays if date.month == month]
    
    return len(holidays_in_month)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Carregar dados usando Helpers
df_datas_reajustes = Helpers.getEntityData(context, 'datas_reajustes')
df_estrutura = Helpers.getEntityData(context, 'df_estrutura_produto')
df_estrutura.columns = ["Codigo", "Componente", "Quantidade"] 
df_portalvendas = Helpers.getEntityData(context, 'portal_vendas')
df_produtos = Helpers.getEntityData(context, 'df_produtos')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Definir nomes de colunas
NR_DO_PEDIDO_MTE = "NR_DO_PEDIDO_MTE"
DATA_PEDIDO = "DATA_PEDIDO"
CODIGO_MTE = "CODIGO_MTE"
QTDE_PEDIDA = "QTDE_PEDIDA"
QTDE_ENTREGUE = "QTDE_ENTREGUE"
QTDE_SALDO = "QTDE_SALDO"
VALOR_UNITARIO = "VALOR_UNITARIO"
VALOR_FATURADO = "VALOR_FATURADO"
VALOR_SALDO = "VALOR_SALDO"
VALOR_PEDIDO = "VALOR_PEDIDO"
NR_PEDIDO_CLIENTE = "NR_PEDIDO_CLIENTE"
NOTA_FISCAL = "NOTA_FISCAL"
DATA_NOTA_FISCAL = "DATA_NOTA_FISCAL"
CODIGO_REPRESENTANTE = "CODIGO_REPRESENTANTE"
CODIGO_MERCADO = "CODIGO_MERCADO"
CODIGO_CLIENTE = "CODIGO_CLIENTE"
COD_MTE_COMP = "COD_MTE_COMP"
CODIGO_MTE_ORIG_EXP = "CODIGO_MTE_ORIG_EXP"

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Preparação dos dados
# 1. FILTRO RODAPÉ - Remove linha inválida no final do arquivo
df_portalvendas = df_portalvendas.drop(df_portalvendas.tail(1).index)

# Sanitização de colunas
df_portalvendas.columns = (
    df_portalvendas.columns.str.replace(" ", "_")
    .str.replace("._", "_", regex=False)
    .str.replace("_R\$", "", regex=True)
)

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

df_portalvendas = df_portalvendas.astype(dict_columns_types)

for col in ['QTDE_PEDIDA', 'QTDE_ENTREGUE', 'QTDE_SALDO']:
    df_portalvendas[col] = pd.to_numeric(df_portalvendas[col], errors='coerce').astype('Int64')

print("✅ Data types converted successfully!")

# 2. FILTRO JANELA TEMPORAL
filter_last_2years = False
if filter_last_2years:
    last_date = pd.to_datetime(pd.Timestamp.now().date())
    last_date_first_day_of_month = pd.Timestamp(f"{last_date.year}-{last_date.month:02d}-01")
    start_date = last_date_first_day_of_month - pd.DateOffset(years=2)

    # Filtrar: remover histórico antigo (> 2 anos) e mês corrente incompleto
    mask_time_out = (df_portalvendas[DATA_PEDIDO] < start_date) | (df_portalvendas[DATA_PEDIDO] >= last_date_first_day_of_month)
    df_portalvendas = df_portalvendas[~mask_time_out]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Identificar componentes importados
set_components_all = set(df_estrutura["Componente"].unique())
set_components_imported = set(df_estrutura[df_estrutura["Componente"].str.startswith("T")]["Componente"].unique())

set_products_use_imported = set()
for component in set_components_imported:
    set_products_use_imported = set_products_use_imported.union(
        set(df_estrutura[df_estrutura["Componente"] == component]["Codigo"].unique())
    )

print(f"Total de componentes: {len(set_components_all)}")
print(f"Componentes importados: {len(set_components_imported)}")
print(f"Produtos que usam componentes importados: {len(set_products_use_imported)}")

# ==================================================================================
# DEBUG: Diagnóstico do filtro de produtos importados
# ==================================================================================
print("\n" + "="*80)
print("🔍 DEBUG: DIAGNÓSTICO DO FILTRO DE PRODUTOS IMPORTADOS")
print("="*80)

# Amostras de set_products_use_imported
sample_products = list(set_products_use_imported)[:5]
print(f"\n📦 set_products_use_imported:")
print(f"   Tipo dos elementos: {type(sample_products[0]) if sample_products else 'N/A'}")
print(f"   Exemplos (primeiros 5): {sample_products}")
print(f"   Repr dos exemplos: {[repr(x) for x in sample_products]}")

# Amostras de COD_MTE_COMP no df_portalvendas
sample_cod_mte = df_portalvendas[COD_MTE_COMP].dropna().unique()[:5].tolist()
print(f"\n🛒 df_portalvendas[COD_MTE_COMP]:")
print(f"   Tipo da coluna: {df_portalvendas[COD_MTE_COMP].dtype}")
print(f"   Exemplos (primeiros 5): {sample_cod_mte}")
print(f"   Repr dos exemplos: {[repr(x) for x in sample_cod_mte]}")

# Verificar se há espaços em branco
has_spaces_estrutura = any(' ' in str(x) for x in sample_products)
has_spaces_portal = any(' ' in str(x) for x in sample_cod_mte)
print(f"\n🔤 Verificação de espaços:")
print(f"   set_products_use_imported contém espaços: {has_spaces_estrutura}")
print(f"   COD_MTE_COMP contém espaços: {has_spaces_portal}")

# Verificar intersecção direta
set_cod_mte = set(df_portalvendas[COD_MTE_COMP].dropna().unique())
intersecao_direta = set_products_use_imported.intersection(set_cod_mte)
print(f"\n🔗 Intersecção direta:")
print(f"   Tamanho set_products_use_imported: {len(set_products_use_imported)}")
print(f"   Tamanho set_cod_mte (portal): {len(set_cod_mte)}")
print(f"   Intersecção: {len(intersecao_direta)}")

# Verificar intersecção com strip (removendo espaços)
set_products_stripped = {str(x).strip() for x in set_products_use_imported}
set_cod_mte_stripped = {str(x).strip() for x in set_cod_mte}
intersecao_stripped = set_products_stripped.intersection(set_cod_mte_stripped)
print(f"\n🔗 Intersecção após strip():")
print(f"   Intersecção: {len(intersecao_stripped)}")
if len(intersecao_stripped) > len(intersecao_direta):
    print(f"   ⚠️ PROBLEMA DETECTADO: Espaços em branco estão causando mismatch!")
    print(f"   Exemplos de intersecção após strip: {list(intersecao_stripped)[:5]}")

print("="*80 + "\n")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# INÍCIO: BLOCO DE FUNÇÕES DE FEATURE ENGINEERING (sem modificações)
# ==================================================================================
def log_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result
    return wrapper

@log_step
def start_pipeline(df):
    return df.copy()

@log_step
def add_cols(
    df, cols=["QTD_VENDA_NACIONAL", "QTD_VENDA_EXPORTACAO"], new_col="QTD_VENDA_TOTAL"
):
    df[new_col] = df[cols].sum(axis=1)
    return df

@log_step
def filter_values(df, col, values):
    return df[df[col].isin(values)]


@log_step
def convert_col(df, col, dtype):
    df[col] = df[col].astype(dtype)
    return df


@log_step
def drop_cols(df, cols):
    return df.drop(cols, axis=1)

@log_step
def select_cols(df, cols):
    return df[cols]

@log_step
def create_features_from_dates(df, date_col="DATA"):
    """
    This function opens up a date column and creates new columns with the month,
    day of the week, week of the year, week of the month, christmas
    week flag, week before christmas flag, last workday of the month flag and
    day before the last workday of the month flag.
    """
    # create rows for the missing dates
    df = df.set_index(date_col)

    df = df.reindex(
        pd.date_range(start=df.index.min(), end=df.index.max(), freq="D"), fill_value=0
    )

    df = df.reset_index()
    df = df.rename(columns={"index": date_col})

    # create a column with the month
    df["MES"] = df[date_col].dt.month
    # create a column with the day of the week
    df["DIA_SEMANA"] = df[date_col].dt.dayofweek
    # create a column with the week of the year
    df["SEMANA_ANO"] = df[date_col].dt.isocalendar().week
    # create column with the week of the month
    df["SEMANA_MES"] = df[date_col].dt.day // 7 + 1
    # create column with the year
    df["ANO"] = df[date_col].dt.year
    
    # create a column with christmas day
    df["NATAL"] = pd.to_datetime(df["ANO"].apply(lambda x: dt.date(x, 12, 25)))
    # create column with christmas week of the year
    df["NATAL_SEMANA_ANO"] = df["NATAL"].dt.isocalendar().week
    # create column "IS_CHRISTMAS_WEEK"
    df["SEMANA_NATAL"] = df["NATAL_SEMANA_ANO"] == df["SEMANA_ANO"]
    # create column with is week before christmas week
    df["SEMANA_ANTES_NATAL"] = df["NATAL_SEMANA_ANO"] - df["SEMANA_ANO"] == 1

    df["SEMANA_NATAL"] = df["SEMANA_NATAL"].astype(bool)
    df["SEMANA_ANTES_NATAL"] = df["SEMANA_ANTES_NATAL"].astype(bool)

    # create column "ULTIMO_DIA_MES" with True if the date is the last workday of the month. If the value is not the last day of the month, but is the last workday of the month, the value is True.
    df["ULTIMO_DIA_MES"] = df[date_col].dt.is_month_end
    df

    # create column with if the date is the day before the last workday of the month
    # first, create a column with the next day
    df["DATA_PROXIMO_DIA"] = df[date_col] + dt.timedelta(days=1)
    df["PENULTIMO_DIA_MES"] = df["DATA_PROXIMO_DIA"].dt.is_month_end
    df
    # drop the columns that are not needed anymore: B1_COD_PP, DATA, CHRISTMAS, DATA_PROXIMO_DIA, CHRISTMAS_SEMANA_ANO
    df = df.drop(
        columns=["NATAL", "DATA_PROXIMO_DIA", "NATAL_SEMANA_ANO", "SEMANA_ANO", "ANO"]
    )
    df
    # one-hot encode the columns "MES", "DIA_SEMANA", "SEMANA_MES", "ANO"
    df = pd.get_dummies(df, columns=["MES", "DIA_SEMANA", "SEMANA_MES"])
    return df

@log_step
def create_chinese_holiday_count_column(df, date_column, new_column_name="chinese_holidays_in_month"):
    """
    Cria uma coluna no DataFrame com a contagem de feriados chineses no mês.
    """
    # Otimização: Calcular feriados únicos para (ano, mês) presentes no DF para evitar loops desnecessários
    unique_dates = df[date_column].dt.to_period('M').unique()
    
    # Dicionário de lookup: {(year, month): count}
    holiday_lookup = {}
    for period in unique_dates:
        holiday_lookup[(period.year, period.month)] = count_chinese_holidays_in_month(period.year, period.month)
    
    # Aplicação vetorizada usando map
    # Primeiro criamos tuplas temporárias (ano, mes)
    temp_periods = df[date_column].dt.to_period('M')
    
    # Mapeamos para a nova coluna
    df[new_column_name] = temp_periods.apply(lambda x: holiday_lookup.get((x.year, x.month), 0))
    
    return df

@log_step
def create_features_from_calendario(df, df_calendario, date_col="DATA"):
    # create rows for the missing dates
    df = df.set_index(date_col)

    df = df.reindex(
        pd.date_range(start=df.index.min(), end=df.index.max(), freq="D"), fill_value=0
    )

    df = df.reset_index()

    df = df.rename(columns={"index": date_col})
    df = df.merge(df_calendario, how="left", left_on=date_col, right_on="Date")

    df = choose_index(df, date_col)

    return df

@log_step
def create_features_from_calendario2(df, df_calendario, date_col="DATA"):
    # create rows for the missing dates
    # df = df.set_index(date_col)

    df = df.resample("D").sum()

    # df = df.rename(columns={"index": date_col})
    df = df.merge(df_calendario, how="left", left_on=date_col, right_on="Date")

    # df = choose_index(df, date_col)

    return df

@log_step
def merge(df, df_calendario):

    df = df.merge(df_calendario, how="left", left_index=True, right_index=True)

    return df

@log_step
def group_by_sum(df, group_by, sum_by = None, set_group_as_index=True):
    if set_group_as_index:
        df = pd.DataFrame(df.groupby(group_by)[sum_by].sum())
    else:
        df = df.groupby(group_by).sum().reset_index()
    return df

@log_step
def flag_first_and_last_month_of_quarter(df, date_column, flag_first_column="quarter_start", flag_last_column="quarter_end"):
    df[flag_first_column] = df[date_column].dt.is_quarter_start
    df[flag_last_column] = df[date_column].dt.is_quarter_end
    return df

@log_step
def fill_all_days(df, date_column, product_column):
    # create rows for the missing dates for each product

    # date_column may not have weekends and has duplicated dates
    
    df = df.set_index([date_column, product_column])
    df = df.reindex(
        pd.MultiIndex.from_product(
            [df.index.levels[0], df.index.levels[1]], names=df.index.names
        ),
        fill_value=0,
    )
    df = df.reset_index()
    return df

@log_step
def choose_index(df, col):
    df = df.set_index(col)
    return df

@log_step
def to_period(df, col, freq="M"):
    df[col] = df[col].dt.to_period(freq=freq)
    df[col] = df[col].dt.to_timestamp()
    return df

@log_step
def fill_all_missing_periods(df, date_column, product_column, selected_cols=None, freq="D", fill_dict=None):
    """
    Preenche datas ausentes para cada produto entre min e max date.
    CORRIGIDA: Evita erro de broadcast (shapes mismatch) ao concatenar colunas.
    """
    
    # CORREÇÃO: Converter df.columns para lista Python
    if selected_cols is None:
        selected_cols = df.columns.tolist() 
    else:
        selected_cols = list(selected_cols)
    
    # Gerar range de datas base
    min_date = df[date_column].min()
    max_date = df[date_column].max()
    date_range = pd.date_range(min_date, max_date, freq=freq)
    
    # Lista de produtos únicos
    product_codes = df[product_column].unique().tolist()
    
    # Criar todas as combinações (Produto x Data)
    all_combinations = pd.MultiIndex.from_product(
        [product_codes, date_range], 
        names=[product_column, date_column]
    )
    new_df = pd.DataFrame(index=all_combinations).reset_index()
    
    # Merge com os dados originais (Left Join mantém as datas vazias criadas)
    merged_df = pd.merge(new_df, df, on=[product_column, date_column], how='left')
    
    # Aplicação do dicionário de preenchimento (fill_dict)
    if fill_dict:
        for value, col in fill_dict.items():
            if col not in merged_df.columns:
                continue
                
            if value == "ffill":
                merged_df[col] = merged_df.groupby(product_column)[col].ffill()
            elif value == "bfill":
                merged_df[col] = merged_df.groupby(product_column)[col].bfill()
            else:
                merged_df[col] = merged_df[col].fillna(value)
    
    # Preencher colunas numéricas restantes com 0
    non_categorical_cols = merged_df.select_dtypes(exclude=['category', 'object', 'datetime']).columns
    merged_df[non_categorical_cols] = merged_df[non_categorical_cols].fillna(0)
    
    # DEFINIÇÃO SEGURA DAS COLUNAS FINAIS
    full_cols_list = [product_column, date_column] + selected_cols
    cols = list(dict.fromkeys(full_cols_list))
    cols = [c for c in cols if c in merged_df.columns]
    
    return merged_df[cols].copy()
 
    

@log_step
def create_features_from_price_raise_dates(df, df_raise_dates, date_col=None):
    df_out = df.copy()
    df_out["DIA_ANUNCIO"] = 0
    df_out["ULTIMO_DIA_PRECO_ANTIGO"] = 0
    df_out["AUMENTO_ANUNCIADO"] = 0
    df_out["TEMPO_ATE_AUMENTO"] = 1
    df_out["ULTIMO_DIA_MES_ANTERIOR_AUMENTO"] = 0
    anuncio_reajuste = "Anuncio Reajuste".replace(" ", "_")
    ultimo_dia_pra_receber_pedidos = "Ultimo dia pra Receber Pedidos".replace(" ", "_")
    if date_col is None:
        # cut out dates from df_raise_dates that are before the first date in df
        df_raise_dates = df_raise_dates[df_raise_dates[anuncio_reajuste] >= df.index.min()]
        df_raise_dates = df_raise_dates[df_raise_dates[ultimo_dia_pra_receber_pedidos] >= df.index.min()]

        # cut out dates from df_raise_dates that are after the last date in df
        df_raise_dates = df_raise_dates[df_raise_dates[anuncio_reajuste] <= df.index.max()]
        df_raise_dates = df_raise_dates[df_raise_dates[ultimo_dia_pra_receber_pedidos] <= df.index.max()]
    else:
        # cut out dates from df_raise_dates that are before the first date in df
        df_raise_dates = df_raise_dates[df_raise_dates[anuncio_reajuste] >= df[date_col].min()]
        df_raise_dates = df_raise_dates[df_raise_dates[ultimo_dia_pra_receber_pedidos] >= df[date_col].min()]

        # cut out dates from df_raise_dates that are after the last date in df
        df_raise_dates = df_raise_dates[df_raise_dates[anuncio_reajuste] <= df[date_col].max()]
        df_raise_dates = df_raise_dates[df_raise_dates[ultimo_dia_pra_receber_pedidos] <= df[date_col].max()]
    
    for index, row in df_raise_dates.iterrows():
        df_out.loc[row[anuncio_reajuste], "DIA_ANUNCIO"] = 1
        df_out.loc[row[ultimo_dia_pra_receber_pedidos], "ULTIMO_DIA_PRECO_ANTIGO"] = 1

        # get last working day before the announcement that is the last day of its month
        raise_month = row[ultimo_dia_pra_receber_pedidos].month
        month_before = raise_month - 1 if raise_month > 1 else 12

        # get the last day of the month before the raise
        last_day_of_month_before = row[ultimo_dia_pra_receber_pedidos].replace(month=month_before, day=1) + pd.offsets.MonthEnd(0)

        # get days between the announcement and the last day to receive orders
        days = (row[ultimo_dia_pra_receber_pedidos] - row[anuncio_reajuste]).days

        for i in range(0, days):
            date = row[anuncio_reajuste] + pd.Timedelta(days=i)
            df_out.loc[date, "AUMENTO_ANUNCIADO"] = 1
            df_out.loc[date, "TEMPO_ATE_AUMENTO"] = 1 - i/days

            if date == last_day_of_month_before:
                df["ULTIMO_DIA_MES_ANTERIOR_AUMENTO"] = 1

    return df_out

def count_price_raises_annoucements_in_month(year, month, price_raise_dates):
    anuncio_reajuste = "Anuncio Reajuste".replace(" ", "_")
    return len(price_raise_dates[(price_raise_dates[anuncio_reajuste].dt.year == year) & (price_raise_dates[anuncio_reajuste].dt.month == month)])

def count_price_raises_in_month(year, month, price_raise_dates):
    ultimo_dia_pra_receber_pedidos = "Ultimo dia pra Receber Pedidos".replace(" ", "_")
    return len(price_raise_dates[(price_raise_dates[ultimo_dia_pra_receber_pedidos].dt.year == year) & (price_raise_dates[ultimo_dia_pra_receber_pedidos].dt.month == month)])

@log_step
def create_price_raise_count_column(df, date_col, df_raise_dates, price_raise_name="price_raises_in_month", raise_announcement_name="price_raise_announcements_in_month"):
    df_out = df.copy()
    df_out[price_raise_name] = 0
    df_out[raise_announcement_name] = 0
    for date in df_out[date_col].unique():
        year = date.astype('datetime64[Y]').astype(int) + 1970
        month = date.astype('datetime64[M]').astype(int) % 12 + 1
        df_out.loc[df_out[date_col]==date, price_raise_name] = count_price_raises_in_month(year, month, df_raise_dates)

        # assume price raises are known beforehands
        df_out.loc[df_out[date_col]==date, raise_announcement_name] = count_price_raises_annoucements_in_month(year, month, df_raise_dates)

    return df_out


@log_step
def add_next_month_sales_by_part_number(df, part_number_col, sales_col, col_name="sales_next_month"):
    """
    Adds a new column to the DataFrame that represents the next month's sales
    by part number.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        part_number_col (str): The name of the column containing the part number.
        sales_col (str): The name of the column containing the sales.

    Returns:
        pandas.DataFrame: The input DataFrame with the new "sales_next_month" column.
    """
    # Create a new column with the next month's sales
    df[col_name] = df.groupby(part_number_col)[sales_col].shift(-1)
    return df

@log_step
def add_sales_lag_by_part_number(df, part_number_col, sales_col, lag=1, col_name="sales_lag"):
    """Add sales lag by part number

    Args:
        df (DataFrame): Input DataFrame
        part_number_col (str): Name of the column containing the part number
        sales_col (str): Name of the column containing sales
        lag (int or list): Number of lags to add. If list, add all lags in the list
        col_name (str): Name of the column containing the lagged sales. Default is "sales_lag"

    Returns:
        DataFrame: Output DataFrame with the new column
    """
    if hasattr(lag,"__iter__"):
        for l in lag:
            df[f"{col_name}_{l}"] = df.groupby(part_number_col)[sales_col].shift(l)
    else:
        df[f"{col_name}_{lag}"] = df.groupby(part_number_col)[sales_col].shift(lag)

    return df

@log_step
def add_price_percentage_change_by_part_number(df, part_number_col, price_col, col_name="price_change"):
    """Add price percentage change by part number

    Args:
        df (DataFrame): Input DataFrame
        part_number_col (str): Name of the column containing the part number
        price_col (str): Name of the column containing the price
        col_name (str): Name of the column containing the price change. Default is "price_change"

    Returns:
        DataFrame: Output DataFrame with the new column
    """
    # Calculate percentage change
    df[f"{col_name}_1"] = df.groupby(part_number_col)[price_col].pct_change()
    
    # Handle infinite values that occur when previous price is 0
    # Replace inf with a large but finite number, or set to NaN
    df[f"{col_name}_1"] = df[f"{col_name}_1"].replace([np.inf, -np.inf], np.nan)
    
    # Alternatively, you could cap the values:
    # df[f"{col_name}_1"] = df[f"{col_name}_1"].clip(-10, 10)  # Cap between -1000% and +1000%
    
    return df

@log_step
def add_sales_diff_by_part_number(df, part_number_col, sales_col, lag=1, col_name="sales_diff"):

    if hasattr(lag,"__iter__"):
        for l in lag:
            df[f"{col_name}_{l}"] = df.groupby(part_number_col)[sales_col].diff(l)
    else:
        df[f"{col_name}_{lag}"] = df.groupby(part_number_col)[sales_col].diff(lag)
    return df

@log_step
def add_rolling_mean_by_part_number(df, part_number_col, sales_col, window, col_name="mean_sales"):
    if hasattr(window, "__iter__"):
        for w in window:
            # SHIFT(1) para excluir o valor atual
            df[f"{col_name}_{w}"] = (df.groupby(part_number_col)[sales_col]
                                      .shift(1)  
                                      .rolling(w, min_periods=1)
                                      .mean()
                                      .reset_index(level=0, drop=True))
    else:
        df[f"{col_name}_{window}"] = (df.groupby(part_number_col)[sales_col]
                                       .shift(1)
                                       .rolling(window, min_periods=1)
                                       .mean()
                                       .reset_index(level=0, drop=True))
    return df

@log_step
def group_by_aggregate(df, group_by, col_func_dict):
    df = df.groupby(group_by).aggregate(col_func_dict)
    df = df.reset_index()
    return df

@log_step
def fill_values_with_neighbour_mean_by_product(df, product_col, fill_col):
    for product in df[product_col].unique():
        _temp_df_product = df.loc[df[product_col]==product, fill_col]
        df.loc[df[product_col]==product, fill_col] = (_temp_df_product.ffill()+_temp_df_product.bfill())/2
        df.loc[df[product_col]==product, fill_col] = df.loc[df[product_col]==product, fill_col].ffill().bfill()
    return df

@log_step
def add_rolling_max_by_part_number(df, part_number_col, sales_col, window, col_name="max_sales"):
    """
    Adds rolling maximum of sales by part number over a given window size.
    FIXED: Now excludes current value using shift(1).
    """
    if hasattr(window, "__iter__"):
        for w in window:
            df[f"{col_name}_{w}"] = (df.groupby(part_number_col)[sales_col]
                                      .shift(1)  # ← ADICIONAR
                                      .rolling(w, min_periods=1)
                                      .max()
                                      .reset_index(level=0, drop=True))
    else:
        df[f"{col_name}_{window}"] = (df.groupby(part_number_col)[sales_col]
                                       .shift(1)  # ← ADICIONAR
                                       .rolling(window, min_periods=1)
                                       .max()
                                       .reset_index(level=0, drop=True))
    return df

@log_step
def add_rolling_min_by_part_number(df, part_number_col, sales_col, window, col_name="min_sales"):
    """
    Adds rolling minimum of sales by part number over a given window size.
    FIXED: Now excludes current value using shift(1).
    """
    if hasattr(window, "__iter__"):
        for w in window:
            df[f"{col_name}_{w}"] = (df.groupby(part_number_col)[sales_col]
                                      .shift(1)  # ← ADICIONAR
                                      .rolling(w, min_periods=1)
                                      .min()
                                      .reset_index(level=0, drop=True))
    else:
        df[f"{col_name}_{window}"] = (df.groupby(part_number_col)[sales_col]
                                       .shift(1)  # ← ADICIONAR
                                       .rolling(window, min_periods=1)
                                       .min()
                                       .reset_index(level=0, drop=True))
    return df

@log_step
def add_rolling_std_by_part_number(df, part_number_col, sales_col, window, col_name="std_sales"):
    """
    Adds rolling standard deviation of sales by part number over a given window size.
    FIXED: Now excludes current value using shift(1).
    """
    if hasattr(window, "__iter__"):
        for w in window:
            df[f"{col_name}_{w}"] = (df.groupby(part_number_col)[sales_col]
                                      .shift(1)  # ← ADICIONAR
                                      .rolling(w, min_periods=1)
                                      .std()
                                      .reset_index(level=0, drop=True))
    else:
        df[f"{col_name}_{window}"] = (df.groupby(part_number_col)[sales_col]
                                       .shift(1)  # ← ADICIONAR
                                       .rolling(window, min_periods=1)
                                       .std()
                                       .reset_index(level=0, drop=True))
    return df

@log_step
def add_rolling_statistics_by_part_number(df, part_number_col, sales_col, window, col_name="rolling"):
    """
    Adds rolling statistics (mean, max, min, std) of sales by part number.
    FIXED: Now excludes current value using shift(1) for ALL statistics.
    """
    if hasattr(window, "__iter__"):
        for w in window:
            # Criar a série deslocada uma vez
            shifted_series = df.groupby(part_number_col)[sales_col].shift(1)
            
            df[f"rolling_mean_{col_name}_{w}"] = (shifted_series
                                                   .rolling(w, min_periods=1)
                                                   .mean()
                                                   .reset_index(level=0, drop=True))
            df[f"rolling_max_{col_name}_{w}"] = (shifted_series
                                                  .rolling(w, min_periods=1)
                                                  .max()
                                                  .reset_index(level=0, drop=True))
            df[f"rolling_min_{col_name}_{w}"] = (shifted_series
                                                  .rolling(w, min_periods=1)
                                                  .min()
                                                  .reset_index(level=0, drop=True))
            df[f"rolling_std_{col_name}_{w}"] = (shifted_series
                                                  .rolling(w, min_periods=1)
                                                  .std()
                                                  .reset_index(level=0, drop=True))
    else:
        # Criar a série deslocada uma vez
        shifted_series = df.groupby(part_number_col)[sales_col].shift(1)
        
        df[f"rolling_mean_{col_name}_{window}"] = (shifted_series
                                                    .rolling(window, min_periods=1)
                                                    .mean()
                                                    .reset_index(level=0, drop=True))
        df[f"rolling_max_{col_name}_{window}"] = (shifted_series
                                                   .rolling(window, min_periods=1)
                                                   .max()
                                                   .reset_index(level=0, drop=True))
        df[f"rolling_min_{col_name}_{window}"] = (shifted_series
                                                   .rolling(window, min_periods=1)
                                                   .min()
                                                   .reset_index(level=0, drop=True))
        df[f"rolling_std_{col_name}_{window}"] = (shifted_series
                                                   .rolling(window, min_periods=1)
                                                   .std()
                                                   .reset_index(level=0, drop=True))
    return df

@log_step
def impute_missing_values(df, imputer, columns, fit=True):
    """
    Imputes missing values in the DataFrame.
    """
    if fit:
        df[columns] = imputer.fit_transform(df[columns])
    else:
        df[columns] = imputer.transform(df[columns])
    return df


@log_step
def impute_missing_values_by_part_number(df, part_number_col, imputer, columns, fit=True):
    NotImplementedError
    """
    Imputes missing values in the DataFrame by part number.
    """
    if fit:
        df[columns] = df.groupby(part_number_col)[columns].transform(lambda x: imputer.fit_transform(x))
    else:
        df[columns] = df.groupby(part_number_col)[columns].transform(lambda x: imputer.transform(x))
    return df

@log_step
def create_is_last_day_of_price_column(df, date_col, df_raise_dates):
    df_out = df.copy()
    df_out["IS_LAST_DAY_OF_PRICE"] = 0
    ultimo_dia_pra_receber_pedidos = "Ultimo dia pra Receber Pedidos".replace(" ", "_")
    for index, row in df_raise_dates.iterrows():
        df_out.loc[df_out[date_col] == row[ultimo_dia_pra_receber_pedidos],"IS_LAST_DAY_OF_PRICE"] = 1
    return df_out

@log_step
def create_is_raise_announcement_day_column(df, date_col, df_raise_dates):
    df_out = df.copy()
    df_out["IS_RAISE_ANNOUNCEMENT_DAY"] = 0
    anuncio_reajuste = "Anuncio Reajuste".replace(" ", "_")
    for index, row in df_raise_dates.iterrows():
        df_out.loc[df_out[date_col] == row[anuncio_reajuste],"IS_RAISE_ANNOUNCEMENT_DAY"] = 1
    return df_out

@log_step
def create_price_raise_columns(df, date_col, df_raise_dates):
    df_out = df.copy()
    df_out = create_is_last_day_of_price_column(df_out, date_col, df_raise_dates)
    df_out = create_is_raise_announcement_day_column(df_out, date_col, df_raise_dates)
    return df_out

@log_step
def create_features_from_holidays(df, date_column=None, years=None):

    if date_column is None:
        if years is None:
            years = df.index.year.unique()
        df["is_holiday"] = df.index.isin(holidays.Brazil(years=years).keys())
    else:
        if years is None:
            years = df[date_column].dt.year.unique()
        df["is_holiday"] = df[date_column].isin(holidays.Brazil(years=years).keys())
    return df

@log_step
def create_month_column(df: pd.DataFrame, date_column, month_column_name="month", one_hot=True):
    df[month_column_name] = df[date_column].dt.month
    df[month_column_name] = df[month_column_name].astype("category")
    if one_hot:
        df = pd.get_dummies(df, columns=[month_column_name])

    return df

@log_step
def get_dummies(df, columns, cols_to_keep):
        
    original_column = df[cols_to_keep].copy()
    df = pd.get_dummies(df, columns=columns)
    df[cols_to_keep] = original_column
    return df

@log_step
def create_features_from_produtos_part_number(df, df_produtos, columns:list):
    columns.append("B1_COD")
    df_produtos = df_produtos[df_produtos["B1_COD"].str.startswith("T")]
    df_produtos.loc[:, "B1_COD"] = df_produtos["B1_COD"].apply(lambda x: remove_letters_from_string(x))
    df_produtos = df_produtos[columns]

    df["TempPartNumber"] = df[COD_MTE_COMP].apply(lambda x: remove_letters_from_string(x))
    df = df.merge(df_produtos, how="left", left_on="TempPartNumber", right_on="B1_COD")
    df = df.drop(columns=["B1_COD", "TempPartNumber"])
    return df

@log_step
def create_year_column(df ,date_column, year_column_name="year"):
    df[year_column_name] = df[date_column].dt.year
    return df

@log_step
def create_day_column(df ,date_column):
    df["day_of_month"] = df[date_column].dt.day
    return df

@log_step
def create_weekday_column(df ,date_column):
    df["weekday"] = df[date_column].dt.weekday
    return df

@log_step
def create_week_column(df ,date_column):
    df["week"] = df[date_column].dt.isocalendar().week
    df["week"] = df["week"].astype("category")
    return df

@log_step
def create_weekend_column(df ,date_column):
    df["weekend"] = df[date_column].apply(lambda x: 1 if x.weekday() in [5,6] else 0)
    return df

@log_step
def create_day_of_series_column(df ,date_column):
    min_date = df[date_column].min()
    df["day_of_series"] = df[date_column].apply(lambda x: (x - min_date))
    df["day_of_series"] = df["day_of_series"].apply(lambda x: x.days)
    return df

@log_step
def count_holidays_in_month(df, date_column, count_weekends=False):
    # get holiday dates in the month
    df["holidays_in_month"] = df[date_column].apply(lambda x: holidays.Brazil(years=x.year).get(x.month))

    return df
    
def count_holidays_in_month(year, month, count_weekends=False):
    # get holiday dates in the month
    holidays_ = holidays.Brazil(years=[year])
    holidays_ = [date for date in holidays_ if date.month == month]
    if not count_weekends:
        # remove weekends from the list
        holidays_ = [date for date in holidays_ if date.weekday() < 5]
    return len(holidays_)

def count_working_days_in_month(year, month):
    # get all dates in the month
    first_day = dt.date(year, month, 1)
    last_day = dt.date(year, month, calendar.monthrange(year, month)[1])
    dates = pd.bdate_range(first_day, last_day).to_pydatetime().tolist()
    
    holids = list(holidays.Brazil(years=[year]))

    # remove holidays from the list

    mte_recess_2019_2020 = pd.date_range("2019-12-24", "2020-01-05").to_pydatetime().tolist()
    mte_recess_2020_2021 = pd.date_range("2020-12-24", "2021-01-03").to_pydatetime().tolist()
    mte_recess_2021_2022 = pd.date_range("2021-12-24", "2022-01-02").to_pydatetime().tolist()
    mte_recess_2022_2023 = pd.date_range("2022-12-24", "2023-01-01").to_pydatetime().tolist()
    mte_recess_2023_2024 = pd.date_range("2023-12-24", "2024-01-01").to_pydatetime().tolist()
    mte_recess_2024_2025 = pd.date_range("2024-12-24", "2025-01-01").to_pydatetime().tolist()
    mte_recess_2025_2026 = pd.date_range("2025-12-24", "2026-01-01").to_pydatetime().tolist()

    remove_dates =  holids + mte_recess_2019_2020 + mte_recess_2020_2021 + mte_recess_2021_2022 + mte_recess_2022_2023 + mte_recess_2023_2024 + mte_recess_2024_2025 + mte_recess_2025_2026

    dates = [date for date in dates if date not in remove_dates]

    return len(dates)


@log_step
def create_working_days_count_column(df, date_column ,new_column_name="working_days_in_month"):
    for date in df[date_column].unique():
        year = date.astype('datetime64[Y]').astype(int) + 1970
        month = date.astype('datetime64[M]').astype(int) % 12 + 1
        df.loc[df[date_column] == date, new_column_name] = count_working_days_in_month(year, month)
    return df

@log_step
def create_holiday_count_column(df, date_column, count_weekends=False, new_column_name="holidays_in_month"):
    for date in df[date_column].unique():
        year = date.astype('datetime64[Y]').astype(int) + 1970
        month = date.astype('datetime64[M]').astype(int) % 12 + 1
        df.loc[df[date_column] == date, new_column_name] = count_holidays_in_month(year, month)
    return df


@log_step
def create_month_of_series_column(df ,date_column, new_column_name="month_of_series"):
    df[new_column_name] = df[date_column].rank(method="dense", )
    
    return df

@log_step
def drop_na(df, subset=None, active=True):
    if not active:  # if not active, return df
        return df
    df = df.dropna(subset=subset)
    return df

@log_step
def clean_infinite_values(df, columns=None):
    """Remove inf, -inf and very large values from the dataframe"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        # Replace inf and -inf with NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Replace very large values (optional - adjust threshold as needed)
        threshold = 1e10
        df[col] = df[col].where(df[col].abs() < threshold, np.nan)
    
    return df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# ETAPA 1: PIPELINE v2 (DO SEGUNDO CÓDIGO)
# ==================================================================================
def pipeline(df, df_datas_reajustes, df_produtos, _drop_na=True):
    df_processed = (
        df.pipe(start_pipeline)
        .pipe(fill_all_missing_periods, _next_month, COD_MTE_COMP)
        .pipe(to_period, _next_month, "M")
        .pipe(group_by_aggregate,[COD_MTE_COMP, _next_month],{QTDE_PEDIDA: "sum", VALOR_UNITARIO: "max"},)
        .pipe(create_month_of_series_column, _next_month)
        .pipe(create_month_column, _next_month, "next_month_is")
        .pipe(flag_first_and_last_month_of_quarter, _next_month)
        .pipe(create_year_column, _next_month, "next_months_year")
        .pipe(create_holiday_count_column,_next_month, count_weekends=False, new_column_name="holidays_next_month")
        .pipe(create_chinese_holiday_count_column, _next_month, new_column_name="chinese_holidays_next_month")
        .pipe(create_holiday_count_column,_next_month,count_weekends=False,new_column_name="holidays_next_month",)
        .pipe(create_working_days_count_column, _next_month, "working_days_next_month")
        .pipe(create_price_raise_count_column,_next_month,df_datas_reajustes,"price_raises_next_month","raise_annoucements_next_month",)
        .pipe(add_next_month_sales_by_part_number, COD_MTE_COMP, QTDE_PEDIDA,"sales_next_month",)
        
        # --- FEATURES DA RODADA 3 (v2) ---
        .pipe(add_sales_lag_by_part_number, COD_MTE_COMP, QTDE_PEDIDA, [1, 2, 3, 6, 12], "sales_lag")
        .pipe(add_sales_diff_by_part_number, COD_MTE_COMP, QTDE_PEDIDA, [1, 2, 3], "sales_diff",)
        .pipe(add_price_percentage_change_by_part_number, COD_MTE_COMP, VALOR_UNITARIO,"price_change",)
        .pipe(add_sales_lag_by_part_number, COD_MTE_COMP, "price_change_1", [1, 2], "price_change_lag")
        .pipe(add_sales_lag_by_part_number, COD_MTE_COMP,"sales_next_month", 12, "sales_lag_target_12m",)
        .pipe(add_rolling_statistics_by_part_number, COD_MTE_COMP, QTDE_PEDIDA, [2, 3, 6, 12], "sales",)
        # --- FIM DAS MODIFICAÇÕES ---
        
        .pipe(create_features_from_produtos_part_number,df_produtos,columns=["B1_DESC", "NOM_GRUP", "CURVA", "ORIGEM"],)
        .pipe(get_dummies,columns=[ COD_MTE_COMP, "B1_DESC", "NOM_GRUP", "CURVA", "ORIGEM"],cols_to_keep=[COD_MTE_COMP],)
        .pipe(clean_infinite_values)
        .pipe(drop_na, subset=["sales_next_month"], active=_drop_na)
    )
    return df_processed

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Filtrar e preparar dados
df_portalvendas_filtered = df_portalvendas[
    df_portalvendas[COD_MTE_COMP].isin(set_products_use_imported)
]

# ==================================================================================
# DEBUG: Log do resultado do filtro
# ==================================================================================
print("\n" + "="*80)
print("🔍 DEBUG: RESULTADO DO FILTRO DE PRODUTOS IMPORTADOS")
print("="*80)
print(f"   Linhas ANTES do filtro: {len(df_portalvendas):,}")
print(f"   Linhas APÓS o filtro: {len(df_portalvendas_filtered):,}")
print(f"   Percentual mantido: {len(df_portalvendas_filtered)/len(df_portalvendas)*100:.4f}%")

if len(df_portalvendas_filtered) < 100:
    print(f"\n⚠️ ALERTA: Muito poucas linhas após o filtro!")
    print(f"   COD_MTE_COMP únicos no filtrado: {df_portalvendas_filtered[COD_MTE_COMP].nunique()}")
    if len(df_portalvendas_filtered) > 0:
        print(f"   Exemplos: {df_portalvendas_filtered[COD_MTE_COMP].unique()[:10].tolist()}")

print("="*80 + "\n")

df_portalvendas_filtered = refresh_categories(
    df_portalvendas_filtered, COD_MTE_COMP
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Preparar para processamento
df_portalvendas_preprocessing = df_portalvendas_filtered[[COD_MTE_COMP, DATA_PEDIDO, QTDE_PEDIDA, VALOR_UNITARIO]]

_next_month = "_next_month"
df_portalvendas_preprocessing[_next_month] = df_portalvendas_preprocessing[DATA_PEDIDO] + pd.tseries.offsets.MonthBegin()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Funções auxiliares para criar datasets diários e mensais
def to_period(df, col, freq="M"):
    df[col] =df[col].dt.to_period(freq=freq)
    df[col] = df[col].dt.to_timestamp()
    return df

def group_by_aggregate(df, group_by, col_func_dict):
    df = df.groupby(group_by).aggregate(col_func_dict)
    df = df.reset_index()
    return df

def get_daily_portalvendas(df_portalvendas):
    df_daily_portalvendas = df_portalvendas.copy()
    df_daily_portalvendas = fill_all_missing_periods(
        df_daily_portalvendas,
        date_column=DATA_PEDIDO,
        product_column=COD_MTE_COMP,
        selected_cols = [ 'QTDE_PEDIDA', 'QTDE_SALDO', 'QTDE_ENTREGUE', VALOR_UNITARIO, VALOR_FATURADO, VALOR_SALDO, VALOR_PEDIDO],
        freq="D",
        fill_dict={"ffill": VALOR_UNITARIO}
    )
    return df_daily_portalvendas

def get_monthly_portalvendas(df_portalvendas):
    df_monthly_portalvendas = to_period(
        df_portalvendas, DATA_PEDIDO, freq="M"
    )
    
    agg_columns = { QTDE_PEDIDA: 'sum',
                   "QTDE_SALDO": 'sum',
                   "QTDE_ENTREGUE": 'sum',
                   VALOR_FATURADO: 'sum',
                   VALOR_SALDO: 'sum',
                   VALOR_PEDIDO: 'sum' }
    
    df_monthly_portalvendas = group_by_aggregate(
        df_monthly_portalvendas,
        [COD_MTE_COMP, DATA_PEDIDO],
        agg_columns,
    )
    
    df_monthly_portalvendas = df_monthly_portalvendas.rename(
        columns={DATA_PEDIDO: "MONTH"}
    )
    
    return df_monthly_portalvendas

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Criar datasets diários e mensais
print("Criando datasets diários e mensais...")

df_daily_portalvendas = get_daily_portalvendas(df_portalvendas_filtered)
df_monthly_portalvendas = get_monthly_portalvendas(df_daily_portalvendas)

print(f"✅ Daily: {df_daily_portalvendas.shape}")
print(f"✅ Monthly: {df_monthly_portalvendas.shape}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Traduzir para componentes
print("Traduzindo para componentes...")

df_daily_portalvendas_components = create_component_consumption_dataframe(
    df_daily_portalvendas,
    df_estrutura,
    sales_column=QTDE_PEDIDA,
    consumption_column="Consumption",
    code_column=COD_MTE_COMP,
    period_column=DATA_PEDIDO,
    other_columns_to_keep = ['QTDE_SALDO', 'QTDE_ENTREGUE',
    'VALOR_FATURADO', 'VALOR_SALDO', 'VALOR_PEDIDO']
)

df_monthly_portalvendas_components = create_component_consumption_dataframe(
    df_monthly_portalvendas,
    df_estrutura,
    sales_column=QTDE_PEDIDA,
    consumption_column="Consumption",
    code_column=COD_MTE_COMP,
    period_column="MONTH",
    other_columns_to_keep = ['QTDE_SALDO', 'QTDE_ENTREGUE',
    'VALOR_FATURADO', 'VALOR_SALDO', 'VALOR_PEDIDO']
)

print(f"✅ Daily Components: {df_daily_portalvendas_components.shape}")
print(f"✅ Monthly Components: {df_monthly_portalvendas_components.shape}")

Helpers.save_output_dataset(context=context, output_name="new_daily_portalvendas_components", data_frame=df_daily_portalvendas_components)
Helpers.save_output_dataset(context=context, output_name="new_monthly_portalvendas_components", data_frame=df_monthly_portalvendas_components)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Executar pipeline de processamento
df_processed = pipeline(df_portalvendas_preprocessing, df_datas_reajustes, df_produtos)

print(f"DataFrame processado: {df_processed.shape}")
print(f"Período: {df_processed[_next_month].min()} até {df_processed[_next_month].max()}")

# ==================================================================================
# DEBUG: Diagnóstico PRÉ-FILTRO CURVA_D
# ==================================================================================
print("\n" + "="*80)
print("🔍 DEBUG: DIAGNÓSTICO PRÉ-FILTRO CURVA_D")
print("="*80)

curva_cols = [c for c in df_processed.columns if 'CURVA' in c]
print(f"\n📊 Colunas CURVA encontradas: {curva_cols}")

if 'CURVA_D' in df_processed.columns:
    curva_d_1 = (df_processed['CURVA_D'] == 1).sum()
    curva_d_0 = (df_processed['CURVA_D'] == 0).sum()
    curva_d_nan = df_processed['CURVA_D'].isna().sum()
    curva_d_other = len(df_processed) - curva_d_1 - curva_d_0 - curva_d_nan

    print(f"\n📈 Distribuição de CURVA_D:")
    print(f"   CURVA_D == 1 (serão removidos): {curva_d_1:,}")
    print(f"   CURVA_D == 0 (serão mantidos): {curva_d_0:,}")
    print(f"   CURVA_D == NaN: {curva_d_nan:,}")
    print(f"   Outros valores: {curva_d_other:,}")
    print(f"   Total: {len(df_processed):,}")

    # Verificar outras colunas CURVA
    for col in curva_cols:
        if col != 'CURVA_D':
            print(f"   {col} == 1: {(df_processed[col] == 1).sum():,}")
else:
    print(f"\n⚠️ ALERTA: Coluna CURVA_D NÃO encontrada!")
    print(f"   Colunas disponíveis: {df_processed.columns.tolist()}")

print("="*80 + "\n")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Removendo itens sem venda (Curva D - componentes sem histórico de vendas)
linhas_antes_curva_d = len(df_processed)
if 'CURVA_D' in df_processed.columns:
    df_processed = df_processed[df_processed['CURVA_D'] != 1]
    linhas_apos_curva_d = len(df_processed)
    df_processed = df_processed.drop(columns=['CURVA_D'])
    print(f"🔄 Filtro CURVA_D aplicado:")
    print(f"   Linhas antes: {linhas_antes_curva_d:,}")
    print(f"   Linhas após: {linhas_apos_curva_d:,}")
    print(f"   Removidas: {linhas_antes_curva_d - linhas_apos_curva_d:,}")
else:
    print(f"ℹ️ Coluna CURVA_D não encontrada - nenhum item Curva D para remover")
    print(f"   Total de linhas: {linhas_antes_curva_d:,}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Split treino/teste
print("\n" + "="*80)
print("📊 PARTICIONAMENTO TREINO/TESTE (ALEATÓRIO TEMPORAL)")
print("="*80)

# Parâmetros
test_ratio = 0.2  # 20% dos meses para teste
random_seed = 42  # Para reprodutibilidade

# Obter lista única de meses
unique_months = sorted(df_processed[_next_month].unique())
n_months = len(unique_months)
n_test_months = int(n_months * test_ratio)

print(f"\n📅 Informações do Dataset:")
print(f"   Total de meses: {n_months}")
print(f"   Meses para teste: {n_test_months} ({test_ratio*100:.0f}%)")
print(f"   Meses para treino: {n_months - n_test_months} ({(1-test_ratio)*100:.0f}%)")
if n_months > 0:
    print(f"   Período: {unique_months[0]} até {unique_months[-1]}")

# Split temporal sequencial
n_months_test = int(n_months * test_ratio)
train_months = unique_months[:-n_months_test]
test_months = unique_months[-n_months_test:]

df_train_processed = df_processed[df_processed[_next_month].isin(train_months)].copy()
df_test_processed = df_processed[df_processed[_next_month].isin(test_months)].copy()

print(f"\n✅ Meses selecionados:")
print(f"\n   TREINO ({len(train_months)} meses):")
print(f"      Primeiro: {train_months[0]}")
print(f"      Último: {train_months[-1]}")
print(f"      Meses: {[str(m)[:7] for m in train_months[:5]]}... (primeiros 5)")

print(f"\n   TESTE ({len(test_months)} meses):")
print(f"      Primeiro: {test_months[0]}")
print(f"      Último: {test_months[-1]}")
print(f"      Meses: {[str(m)[:7] for m in test_months]}")

# Criar splits mantendo ordem temporal dentro de cada conjunto
df_train_processed = df_processed[df_processed[_next_month].isin(train_months)].copy()
df_test_processed = df_processed[df_processed[_next_month].isin(test_months)].copy()

# Ordenar por data para manter ordem temporal
df_train_processed = df_train_processed.sort_values([_next_month, COD_MTE_COMP]).reset_index(drop=True)
df_test_processed = df_test_processed.sort_values([_next_month, COD_MTE_COMP]).reset_index(drop=True)

print(f"\n📊 Tamanho dos conjuntos:")
print(f"   Treino: {df_train_processed.shape[0]:,} registros")
print(f"   Teste:  {df_test_processed.shape[0]:,} registros")
print(f"   Razão: {df_test_processed.shape[0]/df_train_processed.shape[0]:.2%}")

# Estatísticas de distribuição
print(f"\n📈 Distribuição de Vendas:")
total_train = df_train_processed['QTDE_PEDIDA'].sum()
total_test = df_test_processed['QTDE_PEDIDA'].sum()
print(f"   Treino: {total_train:,.0f} unidades ({total_train/(total_train+total_test)*100:.1f}%)")
print(f"   Teste:  {total_test:,.0f} unidades ({total_test/(total_train+total_test)*100:.1f}%)")

print("\n" + "="*80)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Preparar features
features = df_processed.columns.to_list()
features.remove("sales_next_month")
features.remove("_next_month")
features.remove(COD_MTE_COMP)

print(f"Total de features: {len(features)}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# ETAPA FINAL: GRID DE HIPERPARÂMETROS (DO SEGUNDO CÓDIGO)
# ==================================================================================

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

user_defined_param_grid = {
    'A': grid_A,
    'B': grid_B,
    'C': grid_C
}

print("✅ Grids de parâmetros (v5 - Polimento Final) definidos:")
print(f"Classe A: {len(grid_A)} chaves de parâmetros.")
print(f"Classe B: {len(grid_B)} chaves de parâmetros.")
print(f"Classe C: {len(grid_C)} chaves de parâmetros.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# TREINAR MODELO ABC (DO SEGUNDO CÓDIGO)
xgb_df_train_processed = df_train_processed.copy()
xgb_df_test_processed = df_test_processed.copy()

ensemble_model, y_train, y_pred_train, y_test, y_pred_test = train_abc_models(
    xgb_df_train_processed,
    xgb_df_test_processed,
    features,
    user_param_grid=user_defined_param_grid,
    n_iter_search=30  
)

print("\n" + "="*80)
print(f"TREINAMENTO ABC CONCLUÍDO")
print(f"Modelos disponíveis: {list(ensemble_model['models'].keys())}")
print("="*80)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Mapear classificação ABC (do ensemble_model)
product_class_map = ensemble_model['product_class_map']

# Criar mapeamento de VALOR_UNITARIO (último valor conhecido por produto)
price_map = xgb_df_train_processed.groupby(COD_MTE_COMP)['VALOR_UNITARIO'].last().to_dict()

def get_curva_xyz(row):
    """Extrai classificação XYZ baseado nas colunas CURVA"""
    if 'CURVA_A' in row and row.get('CURVA_A', 0) == 1:
        return 'X'
    elif 'CURVA_B' in row and row.get('CURVA_B', 0) == 1:
        return 'Y'
    elif 'CURVA_C' in row and row.get('CURVA_C', 0) == 1:
        return 'Z'
    else:
        return 'Z'  # Default

# Criar mapeamento XYZ por produto (último valor conhecido)
xyz_map = {}
for produto in xgb_df_train_processed[COD_MTE_COMP].unique():
    produto_data = xgb_df_train_processed[xgb_df_train_processed[COD_MTE_COMP] == produto].iloc[-1]
    xyz_map[produto] = get_curva_xyz(produto_data)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ============================================================================
# 2. CRIAR DATAFRAME DE PRODUTOS (TREINO)
# ============================================================================
df_pred_train = xgb_df_train_processed[[COD_MTE_COMP, '_next_month', 'sales_next_month', 'VALOR_UNITARIO']].copy()
df_pred_train['predito'] = y_pred_train
df_pred_train['observado'] = df_pred_train['sales_next_month']
df_pred_train['tipo'] = 'treino'

# Adicionar classificações
df_pred_train['classe_abc'] = df_pred_train[COD_MTE_COMP].map(product_class_map).fillna('C')
df_pred_train['classe_xyz'] = df_pred_train[COD_MTE_COMP].map(xyz_map).fillna('Z')

# Remover coluna redundante
df_pred_train = df_pred_train.drop(columns=['sales_next_month'])

print(f"   ✅ TREINO: {df_pred_train.shape}")

# ============================================================================
# 3. CRIAR DATAFRAME DE PRODUTOS (TESTE)
# ============================================================================
df_pred_test = xgb_df_test_processed[[COD_MTE_COMP, '_next_month', 'sales_next_month', 'VALOR_UNITARIO']].copy()
df_pred_test['predito'] = y_pred_test
df_pred_test['observado'] = df_pred_test['sales_next_month']
df_pred_test['tipo'] = 'teste'

# Adicionar classificações
df_pred_test['classe_abc'] = df_pred_test[COD_MTE_COMP].map(product_class_map).fillna('C')
df_pred_test['classe_xyz'] = df_pred_test[COD_MTE_COMP].map(xyz_map).fillna('Z')

# Remover coluna redundante
df_pred_test = df_pred_test.drop(columns=['sales_next_month'])

print(f"   ✅ TESTE:  {df_pred_test.shape}")

# ============================================================================
# 4. COMBINAR TREINO + TESTE
# ============================================================================
df_predictions_products = pd.concat([df_pred_train, df_pred_test], ignore_index=True)
df_predictions_products = df_predictions_products.rename(columns={
    '_next_month': 'periodo',
    COD_MTE_COMP: 'produto'
})

# Reordenar colunas para melhor visualização
cols_order = ['produto', 'periodo', 'tipo', 'classe_abc', 'classe_xyz', 
              'VALOR_UNITARIO', 'observado', 'predito']
df_predictions_products = df_predictions_products[cols_order]

print(f"\n   ✅ PRODUTOS COMBINADO: {df_predictions_products.shape}")
print(f"   Período: {df_predictions_products['periodo'].min()} a {df_predictions_products['periodo'].max()}")
print(f"   Produtos únicos: {df_predictions_products['produto'].nunique()}")

print(f"\n   📊 Distribuição de Classes:")
print(f"\n   ABC:")
for classe in ['A', 'B', 'C']:
    count = (df_predictions_products['classe_abc'] == classe).sum()
    pct = count / len(df_predictions_products) * 100
    print(f"      Classe {classe}: {count:,} ({pct:.1f}%)")

print(f"\n   XYZ:")
for classe in ['X', 'Y', 'Z']:
    count = (df_predictions_products['classe_xyz'] == classe).sum()
    pct = count / len(df_predictions_products) * 100
    print(f"      Classe {classe}: {count:,} ({pct:.1f}%)")

print(f"\n   Amostra:")
print(df_predictions_products.head(10))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Separar por tipo para conversão de componentes
df_products_treino = df_predictions_products[df_predictions_products['tipo'] == 'treino'].copy()
df_products_teste = df_predictions_products[df_predictions_products['tipo'] == 'teste'].copy()

print(f"   Separado para conversão:")
print(f"      Treino: {len(df_products_treino)} registros")
print(f"      Teste:  {len(df_products_teste)} registros")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ============================================================================
# 5. CRIAR DATAFRAME DE COMPONENTES COM INFORMAÇÕES ADICIONAIS
# ============================================================================
print("\n2️⃣ Criando DataFrame de componentes com informações completas...")

# Função auxiliar para criar componentes observados mantendo metadados
def create_components_observed_with_metadata(df_products, df_estrutura):
    """Converte vendas observadas de produtos para consumo de componentes com metadados"""
    df_temp = df_products[['produto', 'periodo', 'observado']].copy()
    df_temp = df_temp.rename(columns={
        'produto': COD_MTE_COMP, 
        'periodo': 'DATA_PEDIDO', 
        'observado': 'QTDE_PEDIDA'
    })
    
    df_components = create_component_consumption_dataframe(
        df_temp,
        df_estrutura,
        sales_column='QTDE_PEDIDA',
        consumption_column='observado',
        code_column=COD_MTE_COMP,
        period_column='DATA_PEDIDO'
    )
    
    return df_components.rename(columns={'Component': 'componente', 'DATA_PEDIDO': 'periodo'})

# Função auxiliar para criar componentes preditos mantendo metadados
def create_components_predicted_with_metadata(df_products, df_estrutura):
    """Converte vendas preditas de produtos para consumo de componentes com metadados"""
    df_temp = df_products[['produto', 'periodo', 'predito']].copy()
    df_temp = df_temp.rename(columns={
        'produto': COD_MTE_COMP, 
        'periodo': 'DATA_PEDIDO', 
        'predito': 'QTDE_PEDIDA'
    })
    
    df_components = create_component_consumption_dataframe(
        df_temp,
        df_estrutura,
        sales_column='QTDE_PEDIDA',
        consumption_column='predito',
        code_column=COD_MTE_COMP,
        period_column='DATA_PEDIDO'
    )
    
    return df_components.rename(columns={'Component': 'componente', 'DATA_PEDIDO': 'periodo'})

# ============================================================================
# Criar mapeamento de componente -> metadados (ABC, XYZ, VALOR_UNITARIO)
# ============================================================================
print("   Criando mapeamentos de metadados para componentes...")

# Mapeamento: componente -> produtos que o utilizam
component_to_products = {}
for _, row in df_estrutura.iterrows():
    produto = row['Codigo']
    componente = row['Componente']
    
    if componente not in component_to_products:
        component_to_products[componente] = []
    component_to_products[componente].append(produto)

# Função para determinar classe predominante do componente
def get_component_metadata(componente, df_products, product_class_map, xyz_map, price_map):
    """
    Determina metadados de um componente baseado nos produtos que o utilizam.
    Usa a classe predominante (moda) dos produtos.
    """
    if componente not in component_to_products:
        return 'C', 'Z', 0.0  # Defaults
    
    produtos_que_usam = component_to_products[componente]
    
    # Coletar classes ABC dos produtos
    classes_abc = [product_class_map.get(p, 'C') for p in produtos_que_usam]
    
    # Coletar classes XYZ dos produtos
    classes_xyz = [xyz_map.get(p, 'Z') for p in produtos_que_usam]
    
    # Coletar preços dos produtos (média ponderada seria ideal, mas usaremos média simples)
    prices = [price_map.get(p, 0) for p in produtos_que_usam if price_map.get(p, 0) > 0]
    
    # Determinar classe predominante ABC
    if classes_abc:
        classe_abc = max(set(classes_abc), key=classes_abc.count)
    else:
        classe_abc = 'C'
    
    # Determinar classe predominante XYZ
    if classes_xyz:
        classe_xyz = max(set(classes_xyz), key=classes_xyz.count)
    else:
        classe_xyz = 'Z'
    
    # Determinar preço médio
    if prices:
        valor_unitario = np.mean(prices)
    else:
        valor_unitario = 0.0
    
    return classe_abc, classe_xyz, valor_unitario

# Criar mapeamentos para componentes
component_abc_map = {}
component_xyz_map = {}
component_price_map = {}

# Obter lista única de componentes
unique_components = set()
for componente in df_estrutura['Componente'].unique():
    unique_components.add(componente)

print(f"   Processando {len(unique_components)} componentes únicos...")

for componente in unique_components:
    abc, xyz, price = get_component_metadata(
        componente, 
        df_predictions_products, 
        product_class_map, 
        xyz_map, 
        price_map
    )
    component_abc_map[componente] = abc
    component_xyz_map[componente] = xyz
    component_price_map[componente] = price

print(f"   ✅ Mapeamentos criados para componentes")

# ============================================================================
# Converter TREINO
# ============================================================================
print("   Convertendo TREINO...")
df_comp_obs_treino = create_components_observed_with_metadata(df_products_treino, df_estrutura)
df_comp_pred_treino = create_components_predicted_with_metadata(df_products_treino, df_estrutura)

# Merge observado e predito
df_comp_treino = pd.merge(
    df_comp_obs_treino[['periodo', 'componente', 'observado']],
    df_comp_pred_treino[['periodo', 'componente', 'predito']],
    on=['periodo', 'componente'],
    how='outer'
).fillna(0)

# Adicionar metadados
df_comp_treino['classe_abc'] = df_comp_treino['componente'].map(component_abc_map).fillna('C')
df_comp_treino['classe_xyz'] = df_comp_treino['componente'].map(component_xyz_map).fillna('Z')
df_comp_treino['VALOR_UNITARIO'] = df_comp_treino['componente'].map(component_price_map).fillna(0)
df_comp_treino['tipo'] = 'treino'

# ============================================================================
# Converter TESTE
# ============================================================================
print("   Convertendo TESTE...")
df_comp_obs_teste = create_components_observed_with_metadata(df_products_teste, df_estrutura)
df_comp_pred_teste = create_components_predicted_with_metadata(df_products_teste, df_estrutura)

# Merge observado e predito
df_comp_teste = pd.merge(
    df_comp_obs_teste[['periodo', 'componente', 'observado']],
    df_comp_pred_teste[['periodo', 'componente', 'predito']],
    on=['periodo', 'componente'],
    how='outer'
).fillna(0)

# Adicionar metadados
df_comp_teste['classe_abc'] = df_comp_teste['componente'].map(component_abc_map).fillna('C')
df_comp_teste['classe_xyz'] = df_comp_teste['componente'].map(component_xyz_map).fillna('Z')
df_comp_teste['VALOR_UNITARIO'] = df_comp_teste['componente'].map(component_price_map).fillna(0)
df_comp_teste['tipo'] = 'teste'

# ============================================================================
# Combinar TREINO + TESTE
# ============================================================================
df_predictions_components = pd.concat([df_comp_treino, df_comp_teste], ignore_index=True)

# Reordenar colunas
cols_order = ['componente', 'periodo', 'tipo', 'classe_abc', 'classe_xyz', 
              'VALOR_UNITARIO', 'observado', 'predito']
df_predictions_components = df_predictions_components[cols_order]

print(f"\n   ✅ COMPONENTES: {df_predictions_components.shape}")
print(f"   Período: {df_predictions_components['periodo'].min()} a {df_predictions_components['periodo'].max()}")
print(f"   Componentes únicos: {df_predictions_components['componente'].nunique()}")

# ============================================================================
# Análise de distribuição
# ============================================================================
print(f"\n   📊 Distribuição de Classes (agregada por componente):")
df_comp_unique = df_predictions_components.groupby('componente').agg({
    'classe_abc': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'C',
    'classe_xyz': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Z'
}).reset_index()

print(f"\n   ABC:")
for classe in ['A', 'B', 'C']:
    count = (df_comp_unique['classe_abc'] == classe).sum()
    pct = count / len(df_comp_unique) * 100 if len(df_comp_unique) > 0 else 0
    print(f"      Classe {classe}: {count:,} componentes ({pct:.1f}%)")

print(f"\n   XYZ:")
for classe in ['X', 'Y', 'Z']:
    count = (df_comp_unique['classe_xyz'] == classe).sum()
    pct = count / len(df_comp_unique) * 100 if len(df_comp_unique) > 0 else 0
    print(f"      Classe {classe}: {count:,} componentes ({pct:.1f}%)")

print(f"\n   💰 Estatísticas de Preço:")
print(f"      Média: R$ {df_predictions_components['VALOR_UNITARIO'].mean():.2f}")
print(f"      Mediana: R$ {df_predictions_components['VALOR_UNITARIO'].median():.2f}")
print(f"      Mín: R$ {df_predictions_components['VALOR_UNITARIO'].min():.2f}")
print(f"      Máx: R$ {df_predictions_components['VALOR_UNITARIO'].max():.2f}")

print(f"\n   Amostra:")
print(df_predictions_components.head(10))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ============================================================================
# 6. ADICIONAR BASELINE (MÉDIA MÓVEL) E CALCULAR MÉTRICAS
# ============================================================================
print("\n3️⃣ Adicionando baseline (Média Móvel 12m) e calculando métricas...")

# --- Adicionar baseline (MM12) para COMPONENTES ---
try:
    print("    Calculando baseline 'predito_baseline' (MM12) para componentes...")
    df_predictions_components = calcular_media_movel(
        df_predictions_components,
        group_col='componente',
        date_col='periodo',
        value_col='observado',
        window=12 
    )
    
    # <<< CORREÇÃO: Renomear 'media_movel' ANTES de tentar usá-la >>>
    df_predictions_components = df_predictions_components.rename(columns={'media_movel': 'predito_baseline'})
    
    # Agora podemos preencher os NaNs da coluna (que agora existe)
    df_predictions_components['predito_baseline'] = df_predictions_components['predito_baseline'].fillna(0)
    print("    ✅ Baseline (MM12) adicionado a df_predictions_components.")

except Exception as e:
    print(f"\n    ❌ ERRO ao calcular média móvel para componentes: {e}")


# --- Adicionar baseline (MM12) para PRODUTOS ---
try:
    print("    Calculando baseline 'predito_baseline' (MM12) para produtos...")
    df_predictions_products = calcular_media_movel(
        df_predictions_products,
        group_col='produto',
        date_col='periodo',
        value_col='observado',
        window=12
    )
    
    # <<< CORREÇÃO: Renomear 'media_movel' ANTES de tentar usá-la >>>
    df_predictions_products = df_predictions_products.rename(columns={'media_movel': 'predito_baseline'})

    # Agora podemos preencher os NaNs da coluna (que agora existe)
    df_predictions_products['predito_baseline'] = df_predictions_products['predito_baseline'].fillna(0)
    print("    ✅ Baseline (MM12) adicionado a df_predictions_products.")
except Exception as e:
    print(f"\n    ❌ ERRO ao calcular média móvel para produtos: {e}")

print("\n    Calculando métricas de erro (Modelo vs. Baseline)...")

# --- Processar PRODUTOS ---
print("    Processando Produtos...")
# Renomear predição do modelo
df_predictions_products = df_predictions_products.rename(columns={'predito': 'predito_model'})

# Calcular métricas do MODELO
df_predictions_products['erro_model'] = df_predictions_products['observado'] - df_predictions_products['predito_model']
df_predictions_products['erro_abs_model'] = np.abs(df_predictions_products['erro_model'])
df_predictions_products['erro_pct_model'] = np.where(
    df_predictions_products['observado'] != 0,
    (df_predictions_products['erro_model'] / df_predictions_products['observado']) * 100,
    0
)
df_predictions_products['erro_abs_pct_model'] = np.abs(df_predictions_products['erro_pct_model'])

# Calcular métricas do BASELINE
df_predictions_products['erro_baseline'] = df_predictions_products['observado'] - df_predictions_products['predito_baseline']
df_predictions_products['erro_abs_baseline'] = np.abs(df_predictions_products['erro_baseline'])
df_predictions_products['erro_pct_baseline'] = np.where(
    df_predictions_products['observado'] != 0,
    (df_predictions_products['erro_baseline'] / df_predictions_products['observado']) * 100,
    0
)
df_predictions_products['erro_abs_pct_baseline'] = np.abs(df_predictions_products['erro_pct_baseline'])

# --- Processar COMPONENTES ---
print("    Processando Componentes...")
# Renomear predição do modelo
df_predictions_components = df_predictions_components.rename(columns={'predito': 'predito_model'})

# Calcular métricas do MODELO
df_predictions_components['erro_model'] = df_predictions_components['observado'] - df_predictions_components['predito_model']
df_predictions_components['erro_abs_model'] = np.abs(df_predictions_components['erro_model'])
df_predictions_components['erro_pct_model'] = np.where(
    df_predictions_components['observado'] != 0,
    (df_predictions_components['erro_model'] / df_predictions_components['observado']) * 100,
    0
)
df_predictions_components['erro_abs_pct_model'] = np.abs(df_predictions_components['erro_pct_model'])

# Calcular métricas do BASELINE
df_predictions_components['erro_baseline'] = df_predictions_components['observado'] - df_predictions_components['predito_baseline']
df_predictions_components['erro_abs_baseline'] = np.abs(df_predictions_components['erro_baseline'])
df_predictions_components['erro_pct_baseline'] = np.where(
    df_predictions_components['observado'] != 0,
    (df_predictions_components['erro_baseline'] / df_predictions_components['observado']) * 100,
    0
)
df_predictions_components['erro_abs_pct_baseline'] = np.abs(df_predictions_components['erro_pct_baseline'])

print("    ✅ Métricas de erro duplicadas calculadas!")


# Reordenar colunas para melhor legibilidade
try:
    cols_order_prod = [
        'produto', 'periodo', 'tipo', 'classe_abc', 'classe_xyz', 'VALOR_UNITARIO',
        'observado', 'predito_model', 'predito_baseline',
        'erro_model', 'erro_abs_model', 'erro_pct_model', 'erro_abs_pct_model',
        'erro_baseline', 'erro_abs_baseline', 'erro_pct_baseline', 'erro_abs_pct_baseline'
    ]
    # Remover colunas que possam ter sido removidas em execuções anteriores (ex: 'erro')
    cols_order_prod = [col for col in cols_order_prod if col in df_predictions_products.columns]
    df_predictions_products = df_predictions_products[cols_order_prod]

    cols_order_comp = [
        'componente', 'periodo', 'tipo', 'classe_abc', 'classe_xyz', 'VALOR_UNITARIO',
        'observado', 'predito_model', 'predito_baseline',
        'erro_model', 'erro_abs_model', 'erro_pct_model', 'erro_abs_pct_model',
        'erro_baseline', 'erro_abs_baseline', 'erro_pct_baseline', 'erro_abs_pct_baseline'
    ]
    cols_order_comp = [col for col in cols_order_comp if col in df_predictions_components.columns]
    df_predictions_components = df_predictions_components[cols_order_comp]
    
    print("    ✅ Colunas reordenadas.")
except Exception as e:
    print(f"    ⚠️ Erro ao reordenar colunas (colunas podem estar faltando): {e}")


# ============================================================================
# 7. ANÁLISE POR CLASSE ABC (ATUALIZADA)
# ============================================================================
print("\n" + "="*80)
print("📊 ANÁLISE DE PERFORMANCE POR CLASSE ABC (Modelo vs. Baseline)")
print("="*80)

for tipo in ['treino', 'teste']:
    print(f"\n{'='*60}")
    print(f"{tipo.upper()}")
    print(f"{'='*60}")
    
    df_tipo = df_predictions_products[df_predictions_products['tipo'] == tipo]
    
    for classe in ['A', 'B', 'C']:
        df_classe = df_tipo[df_tipo['classe_abc'] == classe]
        
        if len(df_classe) == 0:
            continue
        
        # Métricas do Modelo
        mae_model = mean_absolute_error(df_classe['observado'], df_classe['predito_model'])
        wmape_model = wmape(df_classe['observado'].values, df_classe['predito_model'].values)
        r2_model = r2_score(df_classe['observado'], df_classe['predito_model'])

        # Métricas do Baseline
        mae_baseline = mean_absolute_error(df_classe['observado'], df_classe['predito_baseline'])
        wmape_baseline = wmape(df_classe['observado'].values, df_classe['predito_baseline'].values)
        
        print(f"\n    Classe {classe}:")
        print(f"       Registros: {len(df_classe):,}")
        print(f"       Volume Total: {df_classe['observado'].sum():,.0f}")
        print(f"       MAE (Modelo):    {mae_model:,.2f} | WMAPE (Modelo): {wmape_model:.2%}")
        print(f"       MAE (Baseline): {mae_baseline:,.2f} | WMAPE (Baseline): {wmape_baseline:.2%}")
        print(f"       R² (Modelo):     {r2_model:.4f}")

# ============================================================================
# 8. SALVAR DATASETS ATUALIZADOS
# ============================================================================
print("\n" + "="*80)
print("💾 SALVANDO DATASETS COM INFORMAÇÕES COMPLETAS")
print("="*80)

# Salvar produtos
Helpers.save_output_dataset(
    context=context,
    output_name="predictions_products_train_test",
    data_frame=df_predictions_products
)
print("    ✅ predictions_products_train_test")
print(f"       Colunas: {list(df_predictions_products.columns)}")

# Salvar componentes
Helpers.save_output_dataset(
    context=context,
    output_name="predictions_components_train_test",
    data_frame=df_predictions_components
)
print("    ✅ predictions_components_train_test")
print(f"       Colunas: {list(df_predictions_components.columns)}")

print("\n" + "="*80)
print("✅ DATASETS ATUALIZADOS COM SUCESSO!")
print("="*80)
print("\nColunas finais em PRODUTOS:")
print(f"    {list(df_predictions_products.columns)}")
print("\nColunas finais em COMPONENTES:")
print(f"    {list(df_predictions_components.columns)}")
print("="*80)

# ============================================================================
# 9. CRIAR df_predictions_source PARA COMPATIBILIDADE (ATUALIZADO)
# ============================================================================
print("\n" + "="*80)
print("🔄 CRIANDO df_predictions_source (COMPATIBILIDADE)")
print("="*80)

# Usar 'predito_model'
df_predictions_source = df_predictions_products[['produto', 'periodo', 'observado', 'predito_model', 'tipo']].copy()
df_predictions_source = df_predictions_source.rename(columns={
    'produto': COD_MTE_COMP,
    'periodo': 'DATA_PEDIDO',
    'predito_model': 'QTDE_PEDIDA'
})

# Adicionar VALOR_UNITARIO
price_map_for_source = df_predictions_products.set_index('produto')['VALOR_UNITARIO'].to_dict()
df_predictions_source[VALOR_UNITARIO] = df_predictions_source[COD_MTE_COMP].map(price_map_for_source)

print(f"    ✅ df_predictions_source criado: {df_predictions_source.shape}")
print(f"    Colunas: {list(df_predictions_source.columns)}")

# Salvar
Helpers.save_output_dataset(
    context=context,
    output_name="df_predictions_source",
    data_frame=df_predictions_source
)
print("    ✅ df_predictions_source salvo")

print("\n" + "="*80)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 6. VISUALIZAÇÃO RÁPIDA
# print("\n" + "="*80)
# print("📊 VISUALIZAÇÃO: TOP 5 COMPONENTES (Modelo vs. Baseline)")
# print("="*80)

# Top 5 componentes por volume
# top_5_components = df_predictions_components.groupby('componente')['observado'].sum().nlargest(5).index

# for comp in top_5_components:
#     df_comp = df_predictions_components[df_predictions_components['componente'] == comp]
    
#     # <<< MODIFICADO: Calcular métricas para Modelo e Baseline >>>
#     # Modelo
#     mae_model = mean_absolute_error(df_comp['observado'], df_comp['predito_model'])
#     wmape_model = wmape(df_comp['observado'].values, df_comp['predito_model'].values)
    
#     # Baseline
#     mae_baseline = mean_absolute_error(df_comp['observado'], df_comp['predito_baseline'])
#     wmape_baseline = wmape(df_comp['observado'].values, df_comp['predito_baseline'].values)
    
#     print(f"\n🔸 {comp}")
#     print(f"    Volume total: {df_comp['observado'].sum():,.0f}")
#     print(f"    MAE (Modelo):    {mae_model:.2f} | WMAPE (Modelo): {wmape_model:.2%}")
#     print(f"    MAE (Baseline): {mae_baseline:.2f} | WMAPE (Baseline): {wmape_baseline:.2%}")
#     print(f"    Meses: {df_comp['periodo'].nunique()}")

# print("\n" + "="*80)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# 7. ANÁLISE POR MÊS (COMPONENTES) 
# ==================================================================================
print("\n" + "="*80)
print("📅 ANÁLISE POR MÊS (COMPONENTES)")
print("="*80)

# --- MODIFICADO: Não precisamos recalcular a MM12, ela já existe como 'predito_baseline' ---
# A função 'calculate_monthly_metrics' agora usará 'predito_baseline' diretamente.
print("    Usando 'predito_baseline' (MM12) existente para análise...")

# --- NOVA FUNÇÃO DE AGREGAÇÃO ROBUSTA ---
def calculate_monthly_metrics(x):
    """
    Calcula métricas para um grupo (mês/tipo), 
    usando as colunas 'predito_model' e 'predito_baseline' existentes.
    """
    
    # Valores que podem ser calculados sempre
    n_componentes = x['componente'].nunique()
    volume_observado = x['observado'].sum()
    volume_predito_xgb = x['predito_model'].sum()
    volume_baseline = x['predito_baseline'].sum() # <<< MODIFICADO: Usa 'predito_baseline'
    bias_xgb_total = (x['observado'] - x['predito_model']).sum()
    bias_pct_xgb_total = (bias_xgb_total / volume_observado * 100) if volume_observado != 0 else 0
    bias_mm12_total = (x['observado'] - x['predito_baseline']).sum() # <<< MODIFICADO: Usa 'predito_baseline'
    bias_pct_mm12_total = (bias_mm12_total / volume_observado * 100) if volume_observado != 0 else 0

    # Calcular a média do valor unitário para o mês
    valor_unitario_medio = x['VALOR_UNITARIO'].mean()

    # Criar o dataframe "limpo" para métricas de erro
    # Na Célula 30, nós preenchemos NaNs com 0, mas podemos filtrar onde baseline é 0
    # para evitar dividir por zero ou métricas enviesadas em períodos sem histórico.
    # Usar 'predito_baseline > 0' é uma boa prática se os primeiros meses tiverem 0.
    x_clean = x[x['predito_baseline'] > 0].copy() 
    
    # --- VERIFICAÇÃO DE SEGURANÇA ---
    if x_clean.empty:
        # Se o df limpo está vazio (ex: primeiro mês), não podemos calcular métricas de erro
        mae_xgb = np.nan
        wmape_xgb = np.nan
        mae_mm12 = np.nan
        wmape_mm12 = np.nan
    else:
        # Se temos dados, calcular métricas normalmente
        y_true_clean = x_clean['observado']
        y_pred_xgb_clean = x_clean['predito_model']
        y_pred_mm12_clean = x_clean['predito_baseline'] # <<< MODIFICADO: Usa 'predito_baseline'
        
        mae_xgb = mean_absolute_error(y_true_clean, y_pred_xgb_clean)
        wmape_xgb = wmape(y_true_clean.values, y_pred_xgb_clean.values)
        mae_mm12 = mean_absolute_error(y_true_clean, y_pred_mm12_clean)
        wmape_mm12 = wmape(y_true_clean.values, y_pred_mm12_clean.values)

    return pd.Series({
        'n_componentes': n_componentes,
        'volume_observado': volume_observado,
        'volume_predito_xgb': volume_predito_xgb,
        'volume_baseline': volume_baseline,
        'valor_unitario_medio': valor_unitario_medio,
        'mae_xgb': mae_xgb,
        'wmape_xgb': wmape_xgb,
        'bias_xgb': bias_xgb_total,
        'bias_pct_xgb': bias_pct_xgb_total,
        'mae_mm12': mae_mm12,
        'wmape_mm12': wmape_mm12,
        'bias_mm12': bias_mm12_total,
        'bias_pct_mm12': bias_pct_mm12_total
    })
# --- FIM DA NOVA FUNÇÃO ---

# Usar a nova função de agregação
# <<< MODIFICADO: Passa 'df_predictions_components' diretamente >>>
df_monthly_analysis = df_predictions_components.groupby(['periodo', 'tipo']).apply(calculate_monthly_metrics).reset_index()

# Calcular melhorias (quanto menor o erro, melhor)
df_monthly_analysis['melhoria_mae_xgb'] = np.where(
    df_monthly_analysis['mae_mm12'] == 0, 0,
    ((df_monthly_analysis['mae_mm12'] - df_monthly_analysis['mae_xgb']) / df_monthly_analysis['mae_mm12']) * 100
)
df_monthly_analysis['melhoria_wmape_xgb'] = np.where(
    df_monthly_analysis['wmape_mm12'] == 0, 0,
    ((df_monthly_analysis['wmape_mm12'] - df_monthly_analysis['wmape_xgb']) / df_monthly_analysis['wmape_mm12']) * 100
)

# Formatar para exibição
df_display = df_monthly_analysis.copy()
df_display['periodo'] = df_display['periodo'].dt.strftime('%Y-%m')
df_display['volume_observado'] = df_display['volume_observado'].apply(lambda x: f"{x:,.0f}")
df_display['volume_predito_xgb'] = df_display['volume_predito_xgb'].apply(lambda x: f"{x:,.0f}")
df_display['volume_baseline'] = df_display['volume_baseline'].apply(lambda x: f"{x:,.0f}")

# Renomear colunas para exibição
df_display = df_display.rename(columns={
    'wmape_xgb': 'WMAPE (XGB)',
    'wmape_mm12': 'WMAPE (MM12)',
    'melhoria_wmape_xgb': 'Melhoria (WMAPE)',
    'valor_unitario_medio': 'Preço Médio'
})

# --- FORMATAÇÃO ROBUSTA PARA NANS ---
def format_percent(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:.2%}"

def format_percent_melhoria(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:+.1f}%"

def format_currency(x):
    if pd.isna(x):
        return "N/A"
    return f"R$ {x:,.2f}"

df_display['WMAPE (XGB)'] = df_display['WMAPE (XGB)'].apply(format_percent)
df_display['WMAPE (MM12)'] = df_display['WMAPE (MM12)'].apply(format_percent)
df_display['Melhoria (WMAPE)'] = df_display['Melhoria (WMAPE)'].apply(format_percent_melhoria)
df_display['Preço Médio'] = df_display['Preço Médio'].apply(format_currency)
# --- FIM DA FORMATAÇÃO ROBUSTA ---

# Selecionar colunas para exibir
colunas_display = [
    'periodo', 
    'tipo', 
    'n_componentes', 
    'volume_observado', 
    'volume_predito_xgb', 
    'volume_baseline',
    'Preço Médio',
    'WMAPE (XGB)',
    'WMAPE (MM12)',     
    'Melhoria (WMAPE)'
]

print("\n" + df_display[colunas_display].to_string(index=False))

# Salvar análise mensal (com todas as colunas)
Helpers.save_output_dataset(
    context=context,
    output_name="monthly_analysis_components_comparative", # Nome atualizado
    data_frame=df_monthly_analysis
)
print("\n    ✅ monthly_analysis_components_comparative_mm12 salvo")

print("\n" + "="*80)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Preparar datas base para previsão recursiva
base_dates = [df_portalvendas_preprocessing[DATA_PEDIDO].max() - pd.tseries.offsets.DateOffset(months=i) - pd.offsets.MonthBegin() for i in range(0, 1)][::-1]

print(f"Datas base para previsão: {base_dates}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# FUNÇÃO: recursive_prediction (Ajustada)
# Esta função chama 'predict_ensemble', que agora contém a lógica ABC.
# ==================================================================================
def recursive_prediction(df_in, model, n_months, base_month, df_datas_reajustes, df_produtos, processing_fn):
    '''
    Previsão recursiva multi-horizonte.
    
    COMPATÍVEL COM:
    - ensemble_model (dict com 'models', 'feature_names' e 'product_class_map')
    - xgb_model tradicional (objeto XGBRegressor)
    
    COMO FUNCIONA:
    1. Para cada mês futuro:
        a. Processa dados com pipeline de features
        b. Faz previsão com modelo (chama predict_ensemble)
        c. Adiciona previsão aos dados históricos
        d. Usa previsão como input para próximo mês
    
    Parameters:
    -----------
    df_in : DataFrame
        Dados históricos de vendas
    model : dict ou XGBRegressor
        Modelo treinado (o novo abc_models ou um modelo legado)
    n_months : int
        Número de meses para prever
    base_month : datetime
        Mês base (último mês com dados reais)
    df_datas_reajustes : DataFrame
        Datas de reajuste de preços
    df_produtos : DataFrame
        Informações dos produtos
    processing_fn : function
        Função pipeline() que cria features
    
    Returns:
    --------
    df_out : DataFrame
        Previsões para os próximos n_months
    '''
    
    print(f"🔮 Previsão recursiva iniciada")
    print(f"    Base: {base_month}")
    print(f"    Horizonte: {n_months} meses")
    
    # Meses que serão previstos
    future_months = [base_month + pd.tseries.offsets.MonthBegin(n) for n in range(1, n_months+1)]
    print(f"    Meses futuros: {future_months[0]} até {future_months[-1]}")
    print()
    
    # Preparar dados iniciais
    df_out = df_in.copy()
    df_out = fill_all_missing_periods(df_out, DATA_PEDIDO, COD_MTE_COMP)
    df_out[_next_month] = df_out[DATA_PEDIDO] + pd.tseries.offsets.MonthBegin()
    df_out = df_out[df_out[_next_month] <= base_month + pd.tseries.offsets.MonthBegin(1)]
    
    # Loop de previsão recursiva
    n_predicted_months = 0
    while n_predicted_months < n_months:
        current_month = base_month + pd.tseries.offsets.MonthBegin(n_predicted_months + 1)
        print(f"    📅 Prevendo mês {n_predicted_months + 1}/{n_months}: {current_month}")
        
        # Manter apenas últimos 12 meses de histórico
        df_out = df_out[df_out[_next_month] >= base_month - pd.tseries.offsets.MonthBegin(12)]
        
        # Processar dados (criar features)
        df_processed = processing_fn(df_out, df_datas_reajustes, df_produtos, _drop_na=False)
        
        # Filtrar apenas o mês que queremos prever
        df_modelinput = df_processed[df_processed[_next_month] == current_month]
        
        if len(df_modelinput) == 0:
            print(f"         ⚠️  Nenhum dado para prever no mês {current_month}")
            n_predicted_months += 1
            continue
        
        # ========================================================================
        # FAZER PREVISÃO - A função predict_ensemble agora lida com o roteamento ABC
        # ========================================================================
        try:
            model_output = predict_ensemble(model, df_modelinput)
            
            if isinstance(model, dict) and 'product_class_map' in model:
                print(f"         ✅ XGBoost (ABC) - {len(df_modelinput)} previsões")
            else:
                print(f"         ✅ XGBoost (Legado) - {len(df_modelinput)} previsões")
                
        except Exception as e:
            print(f"         ❌ Erro na predição: {e}")
            raise
        # ========================================================================
        
        # Adicionar previsões ao dataframe
        df_modelinput["sales_next_month"] = model_output
        
        # Criar dados sintéticos para próxima iteração
        df_synthetic = df_modelinput[[COD_MTE_COMP, "sales_next_month", VALOR_UNITARIO, _next_month]].copy()
        df_synthetic = df_synthetic.rename(columns={"sales_next_month": QTDE_PEDIDA})
        df_synthetic = df_synthetic.rename(columns={_next_month: DATA_PEDIDO})
        df_synthetic[_next_month] = df_synthetic[DATA_PEDIDO] + pd.tseries.offsets.MonthBegin(1)
        
        # Adicionar ao histórico para próxima iteração
        df_out = pd.concat([df_out, df_synthetic], axis="index", ignore_index=True)
        
        n_predicted_months += 1
    
    # Retornar apenas previsões (não histórico)
    df_out = df_out[df_out[_next_month] > base_month + pd.tseries.offsets.MonthBegin()]
    
    print(f"\n✅ Previsão recursiva concluída: {len(df_out)} registros gerados")
    
    return df_out

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ============================================================================
# PREVISÃO RECURSIVA MULTI-HORIZONTE USANDO MODELOS ABC-XGBOOST
# ============================================================================
n_horizons = 12
df_multi_horizon_pred = pd.DataFrame()
df_multi_horizon_pred_component = pd.DataFrame()

print("="*80)
print(f"INICIANDO PREVISÃO RECURSIVA PARA {n_horizons} MESES")
print("="*80 + "\n")

for base_date in base_dates:
    print(f"📅 Predicting for base date: {base_date}")
    
    # USAR ensemble_model (que agora contém os modelos ABC)
    multi_horizon_pred = recursive_prediction(
        df_portalvendas_preprocessing, 
        ensemble_model,  # ← Contém os modelos ABC
        n_horizons, 
        base_date, 
        df_datas_reajustes, 
        df_produtos, 
        processing_fn=pipeline
    )
    
    multi_horizon_pred_component = create_component_consumption_dataframe(
        multi_horizon_pred,
        df_estrutura,
        QTDE_PEDIDA,
        "consumption_predicted_month",
        code_column=COD_MTE_COMP,
        period_column=DATA_PEDIDO,
    )
    
    multi_horizon_pred["base_date"] = base_date
    multi_horizon_pred_component["base_date"] = base_date
    
    df_multi_horizon_pred = pd.concat([df_multi_horizon_pred, multi_horizon_pred], axis="index", ignore_index=True)
    df_multi_horizon_pred_component = pd.concat([df_multi_horizon_pred_component, multi_horizon_pred_component], axis="index", ignore_index=True)

print("\n✅ Previsão recursiva concluída!")
print(f"    Produtos: {df_multi_horizon_pred.shape}")
print(f"    Componentes: {df_multi_horizon_pred_component.shape}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("\n" + "="*80)
print("✅ PROCESSAMENTO COMPLETO!")
print("="*80)
print(f"\n📊 RESUMO DO MODELO ABC-XGBOOST:")
print(f"    - Modelos treinados: {list(ensemble_model['models'].keys())}")
print(f"    - Total de features: {len(features)}")
print(f"    - Previsões geradas: {n_horizons} meses à frente")
print(f"    - Produtos previstos: {df_multi_horizon_pred[COD_MTE_COMP].nunique()}")
print(f"    - Componentes previstos: {df_multi_horizon_pred_component['Component'].nunique()}")
print("="*80)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Salvar resultados usando Helpers
print("\n" + "="*80)
print("SALVANDO RESULTADOS")
print("="*80)

Helpers.save_output_dataset(context=context, output_name="new_daily_portalvendasmultihorizon_products", data_frame=df_multi_horizon_pred)
Helpers.save_output_dataset(context=context, output_name="new_multihorizon_components", data_frame=df_multi_horizon_pred_component)
Helpers.save_output_dataset(context=context, output_name="new_daily_portalvendas", data_frame=df_daily_portalvendas)
Helpers.save_output_dataset(context=context, output_name="new_monthly_portalvendas", data_frame=df_monthly_portalvendas)

print("✅ Todos os datasets foram salvos.")