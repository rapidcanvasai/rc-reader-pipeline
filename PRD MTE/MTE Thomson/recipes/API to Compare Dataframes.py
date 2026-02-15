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
from utils.rcclient.enums import DataSourceType
from utils.rc.dtos.global_variable import GlobalVariable
from utils.rc.dtos.project import Project
from utils.rc.client.requests import Requests

import pandas as pd
import numpy as np
import datetime
import os
import json
import logging
# Inicializa o contexto
context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# ------------------------------------------------------------------------------
# Função auxiliar: Normaliza dados para JSON serializable
def normalize_for_json(value):
    """
    Converte valores que podem quebrar o json.dumps, como np.int64, np.float64,
    datetime e NaN.
    """
    # Trata tipos NumPy
    if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
        return int(value)
    if isinstance(value, (np.float64, np.float32, np.float16)):
        return float(value)
    # Trata datetime
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    # Trata NaN/None
    if pd.isna(value):
        return ""
    return value

# ------------------------------------------------------------------------------
class Model(RCMLModel):
    def load(self, artifacts):
        pass

    def predict(self, model_input, context):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        def load_data(model_input):
            token = Helpers.get_user_token(context)
            Requests.setToken(token)

            df_main = pd.DataFrame(model_input['df_main'])
            df_user = pd.DataFrame(model_input['df_user'])

            return df_main, df_user

        def compare_dataframes(df1, df2):
            df1 = df1.copy()
            df2 = df2.copy()

            # Cenário A: Dimensões diferentes
            if df1.shape != df2.shape:
                return {
                    'status': 'error_dimensions',
                    'message': f"❌ Os arquivos têm dimensões diferentes (Original: {df1.shape}, Novo: {df2.shape}) e não podem ser comparados.",
                    'data': {
                        'original_shape': df1.shape,
                        'new_shape': df2.shape
                    }
                }

            # Normaliza colunas booleanas
            bool_cols = ['On_demand', 'NewProduct', 'IsException', 'Check_Suggestion']
            for col in bool_cols:
                if col in df1.columns:
                    df1[col] = df1[col].apply(lambda x: True if str(x).lower() == 'true' else (False if str(x).lower() == 'false' else x))
                    df2[col] = df2[col].apply(lambda x: True if str(x).lower() == 'true' else (False if str(x).lower() == 'false' else x))

            # Substitui emojis por números em Alert
            icon_to_number = {'🔴': 2, '🟡': 1, '🟢': 0}
            if 'Alert' in df1.columns:
                df1['Alert'] = df1['Alert'].replace(icon_to_number)
                df2['Alert'] = df2['Alert'].replace(icon_to_number)

            # Padroniza colunas e tipos
            df1 = df1.reset_index(drop=True)
            df2 = df2.reset_index(drop=True)
            df2.columns = df1.columns

            # Substitui "-" e "" por "0" em colunas numéricas como string
            for col in df1.columns:
                if df1[col].dtype == 'object' or df2[col].dtype == 'object':
                    try:
                        df1[col] = df1[col].replace(['-', ''], '0')
                        df2[col] = df2[col].replace(['-', ''], '0')
                    except Exception as e:
                        logger.warning(f"Não foi possível substituir '-' ou '' por '0' na coluna '{col}': {e}")

            # Calcular Total_Cost se necessário
            if 'Total_Cost' not in df1.columns and 'Cost' in df2.columns and 'Final_order' in df2.columns:
                df2['Total_Cost'] = df2['Cost'].astype(float) * df2['Final_order'].astype(float)
                if 'Cost' in df1.columns and 'Final_order' in df1.columns:
                    df1['Total_Cost'] = df1['Cost'].astype(float) * df1['Final_order'].astype(float)
            elif 'Total_Cost' in df2.columns and 'Cost' in df2.columns and 'Final_order' in df2.columns:
                df2['Total_Cost'] = df2['Cost'].astype(float) * df2['Final_order'].astype(float)
                if 'Cost' in df1.columns and 'Final_order' in df1.columns:
                    df1['Total_Cost'] = df1['Cost'].astype(float) * df1['Final_order'].astype(float)

            # Conversão de tipos
            for col in df1.columns:
                if col in df2.columns:
                    try:
                        df2[col] = df2[col].astype(df1[col].dtype)
                    except Exception as e:
                        logger.warning(f"Não foi possível converter coluna '{col}': {e}")
                        try:
                            if df1[col].dtype == 'object' or df2[col].dtype == 'object':
                                df1[col] = df1[col].astype(str)
                                df2[col] = df2[col].astype(str)
                            else:
                                df1[col] = pd.to_numeric(df1[col], errors='coerce')
                                df2[col] = pd.to_numeric(df2[col], errors='coerce')
                        except Exception as e2:
                            logger.warning(f"Falha total na conversão da coluna '{col}': {e2}")

            # Normalizar campos de string
            for col in ['Currency', 'Obs']:
                if col in df1.columns:
                    df1[col] = df1[col].fillna('').astype(str)
                    df2[col] = df2[col].fillna('').astype(str)

            logger.info(f"Shapes após processamento - df1: {df1.shape}, df2: {df2.shape}")
            logger.info(f"Colunas df1: {list(df1.columns)}")
            logger.info(f"Colunas df2: {list(df2.columns)}")

            # COMPARAÇÃO PERSONALIZADA: apenas Final_order
            col_to_compare = 'Final_order'
            if col_to_compare not in df1.columns or col_to_compare not in df2.columns:
                return {
                    'status': 'error_missing_column',
                    'message': f"❌ Coluna '{col_to_compare}' não encontrada em um dos DataFrames.",
                    'data': None
                }

            diff_indices = df1.index[df1[col_to_compare] != df2[col_to_compare]]

            if diff_indices.empty:
                return {
                    'status': 'identical',
                    'message': '✅ Os arquivos são idênticos na coluna Final_order.',
                    'data': None
                }

            # DataFrame com as colunas na ordem desejada e formatações
            df_diff = pd.DataFrame({
                'Supp_Cod': df2.loc[diff_indices, 'Supp_Cod'].astype(str).str.replace(r'\.0$', '', regex=True),
                'Component': df2.loc[diff_indices, 'Component'],
                'Final_order_anterior': df1.loc[diff_indices, 'Final_order'],
                'Final_order': df2.loc[diff_indices, 'Final_order'],
                'Cost': df2.loc[diff_indices, 'Cost'].astype(float).round(2),
                'Total_Cost': df2.loc[diff_indices, 'Total_Cost'].astype(float).round(2)
            }).reset_index(drop=True)

            # Normaliza para JSON
            def normalize_for_json(value):
                if pd.isna(value):
                    return None
                if isinstance(value, (pd.Timestamp, )):
                    return value.isoformat()
                return value

            diff_table = []
            for _, row in df_diff.iterrows():
                diff_table.append({
                    'Supp_Cod': normalize_for_json(row['Supp_Cod']),
                    'Component': normalize_for_json(row['Component']),
                    'Final_order_anterior': normalize_for_json(row['Final_order_anterior']),
                    'Final_order': normalize_for_json(row['Final_order']),
                    'Cost': normalize_for_json(row['Cost']),
                    'Total_Cost': normalize_for_json(row['Total_Cost'])
                })

            return {
                'status': 'differences_found',
                'message': f"⚠️ Encontradas {len(diff_table)} linhas com diferenças na coluna Final_order.",
                'data': {
                    'different_columns': ['Final_order'],
                    'differences_table': diff_table,
                    'summary': None
                }
            }

        # Função recursiva para garantir serialização total
        def normalize_structure(obj):
            if isinstance(obj, dict):
                return {k: normalize_structure(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [normalize_structure(v) for v in obj]
            else:
                return normalize_for_json(obj)

        try:
            df_main, df_user = load_data(model_input)
            result = compare_dataframes(df_main, df_user)
            return normalize_structure(result)
        except Exception as e:
            logger.exception("Erro inesperado na comparação de dataframes")
            return {
                'status': 'error',
                'message': f"❌ Erro inesperado: {str(e)}",
                'data': None
            }

# ------------------------------------------------------------------------------
# Salva modelo
Helpers.save_output_rc_ml_model(
    context=context,
    model_name='ps_compare_dataframe',
    model_obj=Model,
    artifacts={}
)