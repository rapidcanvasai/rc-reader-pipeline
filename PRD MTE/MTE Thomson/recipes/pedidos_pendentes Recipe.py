# ================================================================================
  # RECIPE: pedidos_pendentes Recipe
  # ================================================================================
  # DEPENDÊNCIAS: Este recipe requer os seguintes datasets:
  #     1. pedidos_pendentes (dados brutos de pedidos em trânsito)
  #     2. produtos (cadastro mestre de produtos)
  # ================================================================================
  #
  # PROPÓSITO: Formatar e enriquecer dados de pedidos em trânsito, calculando datas
  #            previstas, determinando status de pedidos e gerando alertas para
  #            atrasos de produção e embarque.
  #
  # INPUTS:
  #   - pedidos_pendentes (pedidos em trânsito com datas e quantidades)
  #   - produtos (cadastro de produtos ativos com grupos)
  #
  # OUTPUT:
  #   - df_pending_orders (pedidos formatados com status, alertas e datas calculadas)
  #
  # FILTROS APLICADOS:
  #   1. B1_ATIVO == 'S' (linha 35): Apenas produtos ativos
  #      Exemplo: df_produtos = df_produtos[df_produtos["B1_ATIVO"]=='S']
  #      * Consequência: Produtos inativos não aparecem nos pedidos pendentes
  #
  #   2. replace("nan", None) (linha 40): Remove strings "nan" textuais
  #      Exemplo: df_pedidos_pendentes = df_pedidos_pendentes.replace("nan", None)
  #      * Consequência: Valores textuais "nan" são convertidos para None (nulo)
  #
  #   3. PROCESSO.isna() (linha 47): Determina se pedido está em negociação ou trânsito
  #      Exemplo: df_pedidos_pendentes["Status Pedido"] = np.where(
  #                   df_pedidos_pendentes["PROCESSO"].isna(), "Negociação", "Transito")
  #      * Consequência: Pedidos sem código de processo = "Negociação", com código = "Transito"
  #
  #   4. (today > DT_Prod_Prev) & DT_Prod_Real.isna() (linhas 50-53): Detecta atrasos de produção
  #      Exemplo: df_pedidos_pendentes["Alerta"] = np.where(
  #                   (today > df_pedidos_pendentes["DT. Prod. Prev."]) & 
  #                   (pd.isna(df_pedidos_pendentes['DT. Prod. Real'])), 
  #                   "Prod. Atrasada", "Em Progresso")
  #      * Consequência: Pedidos com data prevista ultrapassada e sem produção real = "Prod. Atrasada"
  #
  #   5. DT_Prod_Real.notna() (linha 94): Verifica se pedido foi produzido
  #      Exemplo: df_pedidos_pendentes['Alerta'] = np.where(
  #                   pd.notna(df_pedidos_pendentes['DT. Prod. Real']), 
  #                   np.where(DT_Prod_Real > DT_Prod_Prev, "Produzido após prazo", "Produzido"),
  #                   "Em Progresso")
  #      * Consequência: Pedidos com produção real confirmada recebem status "Produzido" ou "Produzido após prazo"
  #
  #   6. DT_Prod_Real > DT_Prod_Prev (linha 97): Verifica atraso na produção efetiva
  #      Exemplo: np.where(pd.to_datetime(df_pedidos_pendentes["DT. Prod. Real"], format="%d/%m/%Y") > 
  #                   df_pedidos_pendentes['DT. Prod. Prev.'], "Produzido após prazo", "Produzido")
  #      * Consequência: Produção após data prevista = "Produzido após prazo", dentro do prazo = "Produzido"
  #
  #   7. df_artifact.empty (linha 83): Valida se há dados de artifact para merge
  #      Exemplo: if not df_artifact.empty: # merge artifact data
  #      * Consequência: Se artifact vazio, usa apenas dados base sem datas reais de produção
  #
  #   8. columns in col_names.values() (linha 81): Seleciona apenas colunas mapeadas
  #      Exemplo: df_pedidos_pendentes = df_pedidos_pendentes[col_names.values()]
  #      * Consequência: Colunas não listadas no mapeamento são excluídas do output
  #
  # LÓGICA:
  #   FASE 1 - PREPARAÇÃO (linhas 25-28):
  #     1. Carrega pedidos_pendentes (dados brutos do ERP)
  #     2. Carrega produtos (cadastro mestre)
  #
  #   FASE 2 - PROCESSAMENTO (linhas 30-115 - função get_in_transit_orders):
  #     1. Filtra produtos ativos: df_produtos[B1_ATIVO=='S']
  #     2. Merge com produtos para obter grupo: left_on="PRODUTO", right_on="B1_COD"
  #     3. Limpa valores "nan" textuais: replace("nan", None)
  #     4. Inicializa campos editáveis: DT_Prod_Real="", OBS_1="", OBS_2=""
  #     5. Calcula DT_Prod_Prev = EMBARQUE_PREVISTO_PO - 15 dias
  #     6. Define Status_Pedido:
  #        - Se PROCESSO.isna() → "Negociação"
  #        - Senão → "Transito"
  #     7. Calcula Alerta inicial (sem artifact):
  #        - Se (today > DT_Prod_Prev) & DT_Prod_Real.isna() → "Prod. Atrasada"
  #        - Senão → "Em Progresso"
  #     8. Se artifact disponível:
  #        a. Cria chave: MTE + PO
  #        b. Merge com artifact para obter DT_Prod_Real
  #        c. Recalcula Alerta:
  #           - Se DT_Prod_Real existe:
  #             * Se DT_Prod_Real > DT_Prod_Prev → "Produzido após prazo"
  #             * Senão → "Produzido"
  #           - Senão:
  #             * Se today > DT_Prod_Prev → "Prod. Atrasada"
  #             * Senão → "Em Progresso"
  #     9. Renomeia 23 colunas para nomes de negócio em português
  #     10. Formata todas as datas para "%d-%b-%Y" (ex: "15-Jan-2025")
  #
  #   FASE 3 - FORMATAÇÃO FINAL (linhas 137-150):
  #     1. Chama get_in_transit_orders() sem artifact
  #     2. Define colunas não-editáveis (todas exceto DT_Prod_Real, OBS_1, OBS_2)
  #     3. Converte DT_Prod_Real para string
  #     4. Converte DT_Entrega_PO para datetime
  #     5. Cria coluna year_month = YYYY-MM para agrupamento
  #
  # EXEMPLO COMPLETO:
  #   INPUT:
  #     PRODUTO="ABC123", PROCESSO=null, EMBARQUE_PREVISTO_PO=2025-02-15
  #     QTDE_NAO_ENTREGUE=500, FORNEC_NOM="Supplier A"
  #     today=2025-02-01, df_artifact=empty
  #
  #   CÁLCULOS:
  #   → DT_Prod_Prev = 2025-02-15 - 15 dias = 2025-01-31
  #   → Status_Pedido = "Negociação" (PROCESSO é null)
  #   → Alerta: today (2025-02-01) > DT_Prod_Prev (2025-01-31) = True
  #              DT_Prod_Real.isna() = True
  #              → Alerta = "Prod. Atrasada"
  #   → Renomeia: PRODUTO → "MTE", QTDE_NAO_ENTREGUE → "Qtd", FORNEC_NOM → "Fornecedor"
  #   → Formata: DT_Prod_Prev = "31-Jan-2025"
  #
  #   OUTPUT:
  #   {
  #     "MTE": "ABC123",
  #     "Qtd": 500,
  #     "Fornecedor": "Supplier A",
  #     "Status Pedido": "Negociação",
  #     "Alerta": "Prod. Atrasada",
  #     "DT. Prod. Prev.": "31-Jan-2025",
  #     "DT. Prod. Real": "",
  #     "OBS 1": "",
  #     "OBS 2": ""
  #   }
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
import os
import pandas as pd
import numpy as np
import datetime

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Your code goes here
df_pedidos_pendentes = Helpers.getEntityData(context, 'df_pedidos_pendentes')
df_produtos = Helpers.getEntityData(context, 'df_produtos')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_in_transit_orders(df_pedidos_pendentes: pd.DataFrame, df_produtos: pd.DataFrame, df_artifact=pd.DataFrame()):
    
    # Get current date
    today = datetime.datetime.now()

    df_produtos = df_produtos[df_produtos["B1_ATIVO"]=='S']
    df_produtos = df_produtos[["B1_COD","B1_GRUPO"]]
    df_pedidos_pendentes = df_pedidos_pendentes.merge(df_produtos, left_on="PRODUTO", right_on="B1_COD", how="left")

    # There are some text "nan" values, we need to remove them from all the columns
    df_pedidos_pendentes = df_pedidos_pendentes.replace("nan", None)
    df_pedidos_pendentes["DT. Prod. Real"] = "" # preeenche manualmente
    df_pedidos_pendentes["OBS 1"] = "" # preeenche manualmente
    df_pedidos_pendentes["OBS 2"] = "" # preeenche manualmente

    df_pedidos_pendentes["DT. Prod. Prev."] = df_pedidos_pendentes["EMBARQUE_PREVISTO_PO"] - pd.Timedelta(15, unit="D") # embarque_PO - 15 = data prevista produção
    # When "PROCESSO" is filled, Status Pedido="Transito"
    df_pedidos_pendentes["Status Pedido"] = np.where(df_pedidos_pendentes["PROCESSO"].isna(), "Negociação", "Transito")
    # When today is past the "DT. Prod. Prev", Alerta="Prod. Atrasada"
    
    df_pedidos_pendentes["Alerta"] = np.where(
        (today > df_pedidos_pendentes["DT. Prod. Prev."]) & (pd.isna(df_pedidos_pendentes['DT. Prod. Real'])), 
        "Prod. Atrasada", 
        "Em Progresso")

    col_names = {
        "Alerta":"Alerta",
        "DATA_SI":"Data SI",
        "NUMERO_SI":"SI",
        "B1_GRUPO":"Grupo",
        "PRODUTO":"MTE",
        "CODIGO_X_MTE": "MTE X",
        "QTDE_NAO_ENTREGUE":"Qtd",
        "ENTREGA_PREVISTA_PO": "DT. Entrega PO",
        "CHEG_PORTO_ETA_15": "Entrega Prevista", # "Data Prevista Chegada Porto" + 15 dias
        "FORNEC_NOM": "Fornecedor",
        "COD_PROD_FOR": "Cod Prod. Forn.",
        "PROFORMA": "Proforma",
        "PEDIDO": "PO",
        "INVOICE": "Invoice",
        "PROCESSO": "Código Embarque",
        "CONFIRMACAO_PEDIDO": "Conf. PO",
        "DT. Prod. Prev.": "DT. Prod. Prev.",
        "DT. Prod. Real": "DT. Prod. Real",
        "EMBARQUE_EFET": "Data Embarque",
        "CHEG_PORTO_ETA": "Data Prevista Chegada Porto", # só é preenchida após incluir info de embarque no protheus (ETA), vem da tabela 'pedidos_pendentes'
        "Status Pedido": "Status Pedido",
        "OBS 1": "OBS 1",
        "OBS 2": "OBS 2"}

    df_pedidos_pendentes.rename(columns=col_names, inplace=True)
    df_pedidos_pendentes = df_pedidos_pendentes[col_names.values()]
    
    if not df_artifact.empty:
        df_artifact['key'] = df_artifact['MTE'].astype(str) + df_artifact['PO'].astype(str)
        df_pedidos_pendentes['key'] = df_pedidos_pendentes['MTE'].astype(str) + df_pedidos_pendentes['PO'].astype(str)
        df_artifact['Data Prod Artifact'] = df_artifact["DT. Prod. Real"]
        df_artifact = df_artifact[['key', 'Data Prod Artifact']]
        df_pedidos_pendentes = pd.merge(df_pedidos_pendentes, df_artifact, on='key', how='left' )
        df_pedidos_pendentes["DT. Prod. Real"] = df_pedidos_pendentes['Data Prod Artifact']
        df_pedidos_pendentes = df_pedidos_pendentes.drop(columns=['key', 'Data Prod Artifact'])
        
        # When there is a "DT. Prod. Real" order is produced, with delay or not depending on previous "DT. Prod. Prev."
        df_pedidos_pendentes['Alerta'] = np.where(
            pd.notna(df_pedidos_pendentes['DT. Prod. Real']),
            
            # Condition if there is a real date of production
            np.where(pd.to_datetime(df_pedidos_pendentes["DT. Prod. Real"], format="%d/%m/%Y") > df_pedidos_pendentes['DT. Prod. Prev.'],
            "Produzido após prazo",
            "Produzido"),
            
            # Condition if there is NOT a real date of production
            np.where(today > df_pedidos_pendentes["DT. Prod. Prev."], 
            "Prod. Atrasada",
            "Em Progresso")
        )
        
    # Format dates to (dd-mmm-yyyy)
    df_pedidos_pendentes["DT. Prod. Prev."] = df_pedidos_pendentes["DT. Prod. Prev."].dt.strftime("%d-%b-%Y")
    df_pedidos_pendentes["Data SI"] = df_pedidos_pendentes["Data SI"].dt.strftime("%d-%b-%Y")
    df_pedidos_pendentes["DT. Entrega PO"] = df_pedidos_pendentes["DT. Entrega PO"].dt.strftime("%d-%b-%Y")
    df_pedidos_pendentes["Entrega Prevista"] = df_pedidos_pendentes["Entrega Prevista"].dt.strftime("%d-%b-%Y")
    df_pedidos_pendentes["Data Embarque"] = df_pedidos_pendentes["Data Embarque"].dt.strftime("%d-%b-%Y")
    df_pedidos_pendentes["Data Prevista Chegada Porto"] = df_pedidos_pendentes["Data Prevista Chegada Porto"].dt.strftime("%d-%b-%Y")
    
    return df_pedidos_pendentes

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_pending_orders_from_artifact(df_pedidos_pendentes, df_produtos):
    # Merge with real_prod_date_table.csv in artifacts
    from utils.rc.dtos.artifact import Artifact
    # artifact = Artifact.get('user_real_prod_date')
    # artifact.download_file( 'user_real_prod_date' , 'real_prod_date_table.csv', '/tmp')
    file_path = '/tmp/real_prod_date_table.csv'
    if os.path.exists(file_path):
        df_intransit_artifact = pd.read_csv(file_path)
        # st.dataframe(df_intransit_artifact)
        # Merge table saved in artifact with table from protheus
        df_pending_orders = get_in_transit_orders(df_pedidos_pendentes, df_produtos, df_artifact=df_intransit_artifact)
    else:
        df_pending_orders = "Error"

    return df_pending_orders

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# st.subheader("In transit orders")
#df_pending_orders = get_pending_orders_from_artifact(df_pedidos_pendentes, df_produtos)
df_pending_orders = get_in_transit_orders(df_pedidos_pendentes, df_produtos, df_artifact=pd.DataFrame()) 

uniditable_columns_intransit = df_pending_orders.columns.tolist()
uniditable_columns_intransit.remove("DT. Prod. Real")
uniditable_columns_intransit.remove("OBS 1")
uniditable_columns_intransit.remove("OBS 2")

df_pending_orders["DT. Prod. Real"] = df_pending_orders["DT. Prod. Real"].where(
pd.notnull(df_pending_orders["DT. Prod. Real"]), None)

df_pending_orders["DT. Prod. Real"] = df_pending_orders["DT. Prod. Real"].astype(str)
df_pending_orders['DT. Entrega PO'] = pd.to_datetime(df_pending_orders['DT. Entrega PO'])
df_pending_orders['year_month'] = df_pending_orders['DT. Entrega PO'].dt.strftime('%Y-%m')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# df_pending_orders_edited = st.data_editor(
#                                             df_pending_orders,
#                                             column_config={
#                                                 "DT. Prod. Real": st.column_config.TextColumn(
#                                                     "DT. Prod. Real",
#                                                     help="Date format: DD/MM/YYYY",
#                                                     default="st.",
#                                                     max_chars=10,
#                                                     validate="^\d{2}/\d{2}/\d{4}$",
#                                                 )
#                                             },
                                            
#                                             use_container_width=True, 
#                                             key="df_prod_date_editor", 
#                                             disabled=uniditable_columns_intransit
#                                             )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
Helpers.save_output_dataset(context=context, output_name='df_pending_orders', data_frame=df_pending_orders)