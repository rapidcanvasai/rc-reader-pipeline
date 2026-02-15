# ================================================================================
  # RECIPE: Protheus Recipe
  # ================================================================================
  # DEPENDÊNCIAS: Este recipe requer os seguintes datasets:
  #     1. df_main (sugestões de pedidos com Final_order, LT, Cost, Currency)
  #     2. faturamento_minimo (valores mínimos de faturamento por fornecedor)
  #     3. df_produto_fornecedor (relacionamento produto-fornecedor-fabricante)
  # ================================================================================
  #
  # PROPÓSITO: Gerar ordens de compra formatadas para importação no ERP Protheus,
  #            com cálculo automático de datas, validação de valores mínimos e
  #            enriquecimento com dados de fornecedor/fabricante.
  #
  # INPUTS:
  #   - df_main (sugestões de pedidos com componentes e quantidades)
  #   - faturamento_minimo (limites mínimos por fornecedor)
  #   - df_produto_fornecedor (mapeamento produto-fornecedor-fabricante)
  #
  # OUTPUT:
  #   - df_protheus (ordens de compra prontas para importação no Protheus)
  #
  # FILTROS APLICADOS:
  #   1. Por fornecedor (Supp_Cod == forn): Processa apenas componentes do fornecedor específico
  #      * Consequência: Componentes de outros fornecedores são excluídos da ordem atual
  #
  #   2. Quantidade zero (Final_order == 0 and skip_zero=True): Opcional via export_zeros
  #      * Consequência: Se skip_zero ativado, componentes com Final_order=0 são omitidos
  #
  #   3. Fornecedores válidos (supplier in supplier_selector): Apenas fornecedores da lista
  #      * Consequência: Fornecedores fora da lista não geram ordens de compra
  #
  #   4. Duplicatas (drop_duplicates): Remove linhas idênticas ao final
  #      * Consequência: Registros duplicados são eliminados mantendo apenas primeira ocorrência
  #
  # LÓGICA:
  #   Para cada fornecedor na lista:
  #     1. Valida se atingiu valor mínimo de faturamento
  #     2. Calcula DatPRF = Data Atual + LT (lead time do componente)
  #     3. Calcula DatEmbarque = DatPRF - 32 dias (transit time padrão)
  #     4. Enriquece com código de fabricante via df_produto_fornecedor
  #     5. Aplica padrões: Filial=1, CodComp=35, CC=20, Local=1
  #     6. Adiciona flag Min_Order_Value_Warning para fornecedores abaixo do mínimo
  #
  # EXEMPLO:
  #   Componente X: Final_order=1000, LT=107 dias, Fornecedor=123
  #   → DatPRF = Hoje + 107 dias
  #   → DatEmbarque = DatPRF - 32 dias
  #   → Linha de pedido formatada com todos os campos Protheus preenchidos

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Required imports (df_protheus Recipe)

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
import re
import datetime

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Your code goes here
df_main = Helpers.getEntityData(context, 'df_main')
df_faturamento_minimo = Helpers.getEntityData(context, 'faturamento_minimo')
df_produto_fornecedor = Helpers.getEntityData(context, 'df_produto_fornecedor')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def min_order_value_warning(df_main_user, df_faturamento_minimo):
    """
    Retorna False quando atingiu-se o mínimo para aquele fornecedor e True quando não atingiu.
    """
    supplier_list = df_main_user["Supp_Cod"].unique()
    suppliers_not_reached = {} 
    
    for supplier in supplier_list:
        df_order_supplier = df_main_user[df_main_user["Supp_Cod"] == supplier]
        if not df_order_supplier.empty:
            

            if 'Final_order' in df_order_supplier.columns and 'Cost' in df_order_supplier.columns:
                total = (df_order_supplier['Final_order'] * df_order_supplier['Cost']).sum()
            
            else:
                total = 0
                
            try:
                if 'Supp_Code' in df_faturamento_minimo.columns:
                    col_supplier = 'Supp_Code'
                elif 'Supp Code' in df_faturamento_minimo.columns:
                    col_supplier = 'Supp Code'
                else:
                    possible_cols = [col for col in df_faturamento_minimo.columns 
                                     if 'cod' in col.lower() or 'supp' in col.lower()]
                    if possible_cols:
                        col_supplier = possible_cols[0]
                    else:
                        continue
                
                df_faturamento_minimo[col_supplier] = df_faturamento_minimo[col_supplier].astype('str')
                aux_fat_min = df_faturamento_minimo[df_faturamento_minimo[col_supplier] == str(supplier)]
                
                if not aux_fat_min.empty:
                    aux_fat_min = aux_fat_min.reset_index(drop=True)
                    if 'Fatur.Min.' in aux_fat_min.columns:
                        fat_min = aux_fat_min.loc[0, 'Fatur.Min.']
                    elif 'Faturamento_M_nimo' in aux_fat_min.columns:
                        fat_min = aux_fat_min.loc[0, 'Faturamento_M_nimo']
                    else:
                        continue
                        
                    suppliers_not_reached[supplier] = total < fat_min
                        
            except Exception as e:
                import logging
                logging.warning(f"Erro ao processar fornecedor {supplier}: {e}")
                continue

    return suppliers_not_reached

def create_protheus_dataframe(df_main, df_produto_fornecedor, forn, filial="1", emissao=None, unidreq="1", codcomp="35", local="1", cc="20", fabr=None, transittime=32, skip_zero=True):
    def safe_str_int(value):
        try:
            return str(int(float(value)))
        except (ValueError, TypeError):
            return None
    
    df_main = df_main[df_main["Supp_Cod"] == forn]
    if 'Component' not in df_main.index.names:
        df_main = df_main.set_index('Component', drop=False)
    
    protheus_columns = ["Filial", "Emissao", "UnidReq", "CodComp", "Produto", "Local", "Quant", "CC", "DatPRF", "DatEmbarque", "Forn", "Fabr", "Moeda", "Preco_Unit"]
    df_protheus = pd.DataFrame(columns=protheus_columns)
    current_day = datetime.datetime.now() if emissao is None else datetime.datetime.strptime(emissao, "%d/%m/%Y")
    
    df_produto_fornecedor = df_produto_fornecedor.copy()
    
    df_produto_fornecedor["COD_FORNE"] = df_produto_fornecedor["COD_FORNE"].replace('<NA>', pd.NA)
    df_produto_fornecedor["COD_FORNE"] = pd.to_numeric(df_produto_fornecedor["COD_FORNE"], errors='coerce')
    
    if not pd.isna(forn):
        try:
            forn_float = float(forn)
            df_produto_fornecedor_filtered = df_produto_fornecedor[df_produto_fornecedor["COD_FORNE"] == forn_float]
        except (ValueError, TypeError):
            df_produto_fornecedor_filtered = pd.DataFrame(columns=df_produto_fornecedor.columns)
    else:
        df_produto_fornecedor_filtered = pd.DataFrame(columns=df_produto_fornecedor.columns)
    
    for index, row in df_main.iterrows():
        
        if row["Final_order"] == 0 and skip_zero:
            continue

        component = row["Component"]
        data_row = {}
        
        data_row["Filial"] = filial
        data_row["Emissao"] = current_day.strftime("%Y%m%d") if emissao is None else emissao
        data_row["UnidReq"] = unidreq
        data_row["CodComp"] = codcomp
        data_row["Produto"] = component
        data_row["Local"] = local
        
        data_row["Quant"] = row["Final_order"]

        data_row["CC"] = cc
        
        datprf = current_day + datetime.timedelta(days=row["LT"])
        data_row["DatPRF"] = datprf.strftime("%Y%m%d")
        datembarque = datprf - datetime.timedelta(days=transittime)
        data_row["DatEmbarque"] = datembarque.strftime("%Y%m%d")
        
        data_row["Forn"] = safe_str_int(forn)
        
        fabri_values = df_produto_fornecedor_filtered[df_produto_fornecedor_filtered["PRODUTO"] == component]["COD_FABRI"].values
        if len(fabri_values) > 0:
            fabri = fabri_values[0]
        else:
            fabri = None
        data_row["Fabr"] = safe_str_int(fabri)
        
        data_row["Moeda"] = row["Currency"]
        data_row["Preco_Unit"] = str(row["Cost"])
        
        new_row = pd.DataFrame(data=data_row, index=[component])
        df_protheus = pd.concat([df_protheus, new_row])
    
    if not df_protheus.empty:
        df_protheus['Quant'] = df_protheus['Quant'].astype('int64')
        df_protheus["Preco_Unit"] = df_protheus["Preco_Unit"].astype('float64')
                
    return df_protheus

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# faturamento minimo
default_list = min_order_value_warning(df_main, df_faturamento_minimo)
nao_atingiram_o_minimo = [k for k, v in default_list.items() if v] #Retorna os True
atingiram_o_minimo = [k for k, v in default_list.items() if not v] #Retorna os False

# supplier_selector = df_main.loc[df_main['Min_Order_Value_Warning'], 'Supp_Cod'].unique()
supplier_selector = list(pd.Series(list(default_list.keys()) + df_main['Supp_Cod'].dropna().tolist()).unique())
export_zeros = True

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# As colunas que o DataFrame final deve ter, mesmo que esteja vazio.
colunas_finais = ['Filial', 'Emissao', 'UnidReq', 'CodComp', 'Produto', 'Local', 'Quant',
                  'CC', 'DatPRF', 'DatEmbarque', 'Forn', 'Fabr', 'Moeda', 'Preco_Unit']

lista_de_dataframes = []

for supplier in supplier_selector:
    df_resultado_individual = create_protheus_dataframe(
        df_main, 
        df_produto_fornecedor, 
        supplier, 
        skip_zero=not export_zeros
    )
    lista_de_dataframes.append(df_resultado_individual)

if lista_de_dataframes:
    df_protheus = pd.concat(lista_de_dataframes, ignore_index=True)
else:
    df_protheus = pd.DataFrame(columns=colunas_finais)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
if len(df_protheus) > 0:
    df_protheus.drop_duplicates(inplace=True)
    df_protheus['Min Order Value Warning'] = df_protheus['Forn'].isin(atingiram_o_minimo)
    # df_protheus['Emissao'] = datetime_to_string_aaaammdd(df_protheus['Emissao'])
    # df_protheus['DatPRF'] = datetime_to_string_aaaammdd(df_protheus['DatPRF'])
    # df_protheus['DatEmbarque'] = datetime_to_string_aaaammdd(df_protheus['DatEmbarque'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
Helpers.save_output_dataset(context=context, output_name='df_protheus', data_frame=df_protheus)