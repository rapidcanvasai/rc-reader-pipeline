# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Required imports (Mantenha todos os imports necessários no topo)
from collections import defaultdict
import logging
import datetime
import numpy as np
import pandas as pd
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

# Configuração do Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())
logger.info("Contexto inicializado com sucesso")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE


def clean_whitespace(df, column):
    """Remove espaços em branco no início e fim de uma coluna de string."""
    # Garante que a coluna é do tipo string antes de aplicar .str.strip()
    return df[column].astype(str).str.strip()


def filter_t_components(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Filtra linhas onde a coluna especificada começa com 'T' seguida por um ou mais dígitos.

    Exemplo: 'T123', 'T9' são mantidos; 'TA', 't1' ou '1T' são removidos.

    Retorna:
    --------
    pd.DataFrame: DataFrame com registros que começam com 'T' + dígitos
    """
    mask = df[column].astype(str).str.match(r'^T\d+')
    return df[mask].copy()


def remove_double_dot_lines(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Remove linhas em que a coluna selecionada contém exatamente dois pontos ".".

    Exemplo removido: T9708.50.030

    Retorna:
    --------
    pd.DataFrame: DataFrame com registros que NÃO têm exatamente dois pontos
    """
    mask = df[column].astype(str).str.count(r'\.') != 2
    return df[mask].copy()


def fill_all_missing_periods(df, date_column, product_column, freq="D", fill_dict=None):
    """
    Preenche períodos ausentes em um DataFrame, aplicando métodos de preenchimento por grupo de produto.

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada.
        date_column (str): Nome da coluna de datas.
        product_column (str): Nome da coluna de produtos.
        freq (str): Frequência do período (ex: "D", "M", "W").
        fill_dict (dict): Dicionário com o formato {coluna: metodo_ou_valor}.
                          Exemplo: {"VALOR UNITARIO R$": "ffill", "ESTOQUE": 0}
    """
    # [A sua função original fill_all_missing_periods vai aqui, sem alteração]
    min_date = df[date_column].min()
    max_date = df[date_column].max()
    date_range = pd.date_range(min_date, max_date, freq=freq)
    product_codes = df[product_column].unique().tolist()

    all_combinations = pd.MultiIndex.from_product(
        [product_codes, date_range],
        names=[product_column, date_column]
    )
    new_df = pd.DataFrame(index=all_combinations).reset_index()

    merged_df = pd.merge(
        new_df, df, on=[product_column, date_column], how='left')

    # Aplica os preenchimentos conforme o dicionário
    if fill_dict:
        for col, method in fill_dict.items():
            if col not in merged_df.columns:
                continue  # ignora colunas inexistentes
            if method == "ffill":
                merged_df[col] = merged_df.groupby(product_column)[
                    col].fillna(method="ffill")
            elif method == "bfill":
                merged_df[col] = merged_df.groupby(product_column)[
                    col].fillna(method="bfill")
            else:
                merged_df[col] = merged_df[col].fillna(method)

    # Preenche o restante com 0
    merged_df = merged_df.fillna(0)

    return merged_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 🧱 Preparação de Dados - Produtos nao entregues pela exportacao

logger.info("=" * 60)
logger.info("Iniciando processamento: Produtos não entregues pela exportação")
df_products_not_delivered_exportation = Helpers.getEntityData(
    context, "products_not_delivered_exportation")
logger.info(f"Dados carregados: {len(df_products_not_delivered_exportation)} registros")
df_products_not_delivered_exportation = df_products_not_delivered_exportation.dropna(
    subset=['DESCRICAO', 'QUANT_BO'])
logger.info(f"Após dropna (DESCRICAO, QUANT_BO): {len(df_products_not_delivered_exportation)} registros")
df_products_not_delivered_exportation['PRODUTO'] = df_products_not_delivered_exportation['PRODUTO'].astype(
    str).str.strip()
df_products_not_delivered_exportation['DESCRICAO'] = df_products_not_delivered_exportation['DESCRICAO'].astype(
    str).str.strip()
df_products_not_delivered_exportation['QUANT_BO'] = pd.to_numeric(
    df_products_not_delivered_exportation['QUANT_BO'].astype(str).str.strip(), errors='coerce')
df_products_not_delivered_exportation['DT_ENTREGA'] = pd.to_datetime(
    df_products_not_delivered_exportation['DT_ENTREGA'].astype(str).str.strip(), format='%d/%m/%Y', errors='coerce')
Helpers.save_output_dataset(context=context, output_name="df_products_not_delivered_exportation",
                            data_frame=df_products_not_delivered_exportation)
logger.info(f"Dataset salvo: df_products_not_delivered_exportation ({len(df_products_not_delivered_exportation)} registros)")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 🧱 Preparação de Dados - Estrutura do Produto (BOM)
logger.info("=" * 60)
logger.info("Iniciando processamento: Estrutura do Produto (BOM)")
df_estrutura_produto = Helpers.getEntityData(context, 'estrutura_produto')
logger.info(f"Dados carregados: {len(df_estrutura_produto)} registros")

# Remover espaços em branco das colunas de código
df_estrutura_produto['COD_PRODUTO'] = clean_whitespace(df_estrutura_produto, 'COD_PRODUTO')
df_estrutura_produto['COD_COMPONENTE'] = clean_whitespace(df_estrutura_produto, 'COD_COMPONENTE')
logger.info("Whitespace removido das colunas COD_PRODUTO e COD_COMPONENTE")

# Aplica a função universal de filtragem
df_estrutura_produto = filter_t_components(
    df_estrutura_produto, 'COD_COMPONENTE')
logger.info(f"Após filtro T-components: {len(df_estrutura_produto)} registros")

# 2. Aplica a função de remoção de linhas com dois pontos
df_estrutura_produto = remove_double_dot_lines(
    df_estrutura_produto, 'COD_COMPONENTE')
logger.info(f"Após remoção de linhas com dois pontos: {len(df_estrutura_produto)} registros")

Helpers.save_output_dataset(
    context=context, output_name='df_estrutura_produto', data_frame=df_estrutura_produto)
logger.info(f"Dataset salvo: df_estrutura_produto ({len(df_estrutura_produto)} registros)")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 🧱 Preparação de Dados - Produto Fornecedor
logger.info("=" * 60)
logger.info("Iniciando processamento: Produto Fornecedor")
df_produto_fornecedor = Helpers.getEntityData(context, "produto_fornecedor")
logger.info(f"Dados carregados: {len(df_produto_fornecedor)} registros")
df_produto_fornecedor["PRODUTO"] = clean_whitespace(
    df_produto_fornecedor, "PRODUTO")

# Aplica a função universal de filtragem
df_produto_fornecedor = filter_t_components(df_produto_fornecedor, 'PRODUTO')
logger.info(f"Após filtro T-components: {len(df_produto_fornecedor)} registros")

# 2. Aplica a função de remoção de linhas com dois pontos
df_produto_fornecedor = remove_double_dot_lines(
    df_produto_fornecedor, 'PRODUTO')
logger.info(f"Após remoção de linhas com dois pontos: {len(df_produto_fornecedor)} registros")

# 3. Apenas produtos com preco maior que 0
df_produto_fornecedor = df_produto_fornecedor[df_produto_fornecedor['CUSTO_PRODUTO'] > 0]
logger.info(f"Após filtro CUSTO_PRODUTO > 0: {len(df_produto_fornecedor)} registros")

# --- 1. Limpeza e Padronização de COD_FORNE e COD_FABRI ---
df_produto_fornecedor["COD_FORNE"] = (
    df_produto_fornecedor["COD_FORNE"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .str.replace("nan", "", regex=False)
)

df_produto_fornecedor["COD_FABRI"] = (
    df_produto_fornecedor["COD_FABRI"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .str.replace("nan", "", regex=False)
)
logger.info("Colunas COD_FORNE e COD_FABRI padronizadas")

# 3. Definição das colunas de String
string_cols = ["PRODUTO", "COD_FORNE", "COD_FABRI",
               "FORNEC_NOM", "MOEDA", "COD_PROD_FOR"]

# 4. Aplica o tipo string (pd.StringDtype) nas colunas definidas
df_produto_fornecedor[string_cols] = df_produto_fornecedor[string_cols].astype(
    "string")

# Remover espaços em branco da coluna COD_PROD_FOR
df_produto_fornecedor['COD_PROD_FOR'] = clean_whitespace(df_produto_fornecedor, 'COD_PROD_FOR')

# 5. Conversão e arredondamento
float_cols = ["CUSTO_PRODUTO", "PTAX"]
df_produto_fornecedor[float_cols] = df_produto_fornecedor[float_cols].apply(
    pd.to_numeric, errors="coerce").round(2)
logger.info("Conversão de tipos concluída")

Helpers.save_output_dataset(
    context=context, output_name='df_produto_fornecedor', data_frame=df_produto_fornecedor)
logger.info(f"Dataset salvo: df_produto_fornecedor ({len(df_produto_fornecedor)} registros)")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 📜 Histórico de Inventário e Backcasting (Formato Longo)
logger.info("=" * 60)
logger.info("Iniciando processamento: Histórico de Inventário e Backcasting")

# --- Importa
df_estoque_atual = Helpers.getEntityData(context, 'estoque_atual')
logger.info(f"Estoque atual carregado: {len(df_estoque_atual)} registros")
df_estoque_atual["CODIGO_PI"] = clean_whitespace(df_estoque_atual, "CODIGO_PI")

# Aplica a função universal de filtragem
df_estoque_atual = filter_t_components(df_estoque_atual, 'CODIGO_PI')
logger.info(f"Estoque atual após filtro T-components: {len(df_estoque_atual)} registros")

# 2. Aplica a função de remoção de linhas com dois pontos
df_estoque_atual = remove_double_dot_lines(df_estoque_atual, 'CODIGO_PI')
logger.info(f"Estoque atual após remoção de dois pontos: {len(df_estoque_atual)} registros")

# --- Importa
df_movimentacao_estoque = Helpers.getEntityData(
    context, 'movimentacao_estoque')
logger.info(f"Movimentação estoque carregada: {len(df_movimentacao_estoque)} registros")
df_movimentacao_estoque["B1_COD"] = clean_whitespace(
    df_movimentacao_estoque, "B1_COD")

# Aplica a função universal de filtragem
df_movimentacao_estoque = filter_t_components(
    df_movimentacao_estoque, 'B1_COD')
logger.info(f"Movimentação após filtro T-components: {len(df_movimentacao_estoque)} registros")

# 2. Aplica a função de remoção de linhas com dois pontos
df_movimentacao_estoque = remove_double_dot_lines(
    df_movimentacao_estoque, 'B1_COD')
logger.info(f"Movimentação após remoção de dois pontos: {len(df_movimentacao_estoque)} registros")

# Manipulacao
set_estoque_atual = df_estoque_atual["CODIGO_PI"].unique()
logger.info(f"Produtos únicos no estoque atual: {len(set_estoque_atual)}")

# --- Pré-processamento e Preenchimento de Dias Ausentes ---
df_movimentacao_estoque["DTA_LANCAMENTO"] = pd.to_datetime(
    df_movimentacao_estoque["DTA_LANCAMENTO"])

logger.info("Iniciando preenchimento de períodos ausentes...")
df_movimentacao_estoque_alldays = fill_all_missing_periods(
    df_movimentacao_estoque,
    "DTA_LANCAMENTO",
    "B1_COD",
    freq="D",
    fill_dict={"QTD_LANCADA": 0}
)
logger.info(f"Movimentação com dias preenchidos: {len(df_movimentacao_estoque_alldays)} registros")

min_date = df_movimentacao_estoque_alldays["DTA_LANCAMENTO"].min()
max_date = df_movimentacao_estoque_alldays["DTA_LANCAMENTO"].max()
logger.info(f"Período do histórico: {min_date} até {max_date}")
df_inventory_histories_base = pd.DataFrame(  # Renomeado para evitar conflito
    index=pd.date_range(start=min_date, end=max_date, freq="D")
)

# --- Backcasting (Cálculo Retroativo) ---
logger.info("Iniciando backcasting (cálculo retroativo)...")
estoque_por_part = dict(tuple(df_estoque_atual.groupby("CODIGO_PI")))
movimentacoes_por_part = dict(
    tuple(df_movimentacao_estoque_alldays.groupby("B1_COD")))
new_inventory_histories = defaultdict(dict)

processed_count = 0
total_parts = len(set_estoque_atual)
for part_number in set_estoque_atual:
    df_part_number = estoque_por_part.get(part_number)
    df_deltas = movimentacoes_por_part.get(part_number)

    if df_part_number is None or df_part_number.empty:
        continue

    if df_deltas is None or df_deltas.empty:
        last_stock = df_part_number["QTD_TOT_EST"].iloc[-1]
        for day in df_inventory_histories_base.index:
            new_inventory_histories[day][part_number] = last_stock
        continue

    df_deltas_reversed = df_deltas.sort_values(
        by="DTA_LANCAMENTO", ascending=False)
    stock = df_part_number["QTD_TOT_EST"].iloc[-1]
    unique_dates = df_deltas_reversed["DTA_LANCAMENTO"].unique()

    for day in df_inventory_histories_base.index[::-1]:
        if day < min_date:
            continue

        if day in unique_dates:
            delta = df_deltas_reversed[df_deltas_reversed["DTA_LANCAMENTO"]
                                       == day]["QTD_LANCADA"].sum()
            stock -= delta

        new_inventory_histories[day][part_number] = stock

    processed_count += 1
    if processed_count % 500 == 0:
        logger.info(f"Backcasting: {processed_count}/{total_parts} produtos processados")

logger.info(f"Backcasting concluído: {processed_count} produtos processados")

# --- Finalização do DataFrame de Histórico (Cria Pivotado, DERRETE para Longo) ---
logger.info("Criando DataFrame de histórico de inventário...")

# 1. Cria o DataFrame pivotado (Este passo é necessário para usar o resultado do backcasting)
df_inventory_histories_pivot = pd.DataFrame.from_dict(
    new_inventory_histories, orient="index").sort_index()
logger.info(f"DataFrame pivotado criado: {df_inventory_histories_pivot.shape}")

# 2. Renomeia colunas (mantida a lógica de substituir "_" por "/")
df_inventory_histories_pivot.columns = df_inventory_histories_pivot.columns.astype(
    str).str.replace("_", "/")
df_inventory_histories_pivot = df_inventory_histories_pivot.reset_index(
).rename(columns={'index': 'date'})

# 3. DESPIVOTAMENTO
logger.info("Realizando despivotamento (melt)...")
df_inventory_histories = df_inventory_histories_pivot.melt(
    id_vars=['date'],                      # Coluna a ser mantida (Datas)
    # Nova coluna para os nomes dos componentes (T123, T456)
    var_name='Componente',
    value_name='QTD_ESTOQUE'               # Nova coluna para os valores de estoque
)
logger.info(f"Após melt: {len(df_inventory_histories)} registros")

# 4. Limpeza final e tipo
df_inventory_histories['Componente'] = df_inventory_histories['Componente'].astype(
    str)

Helpers.save_output_dataset(
    context=context, output_name='new_inventory_histories2', data_frame=df_inventory_histories)
logger.info(f"Dataset salvo: new_inventory_histories2 ({len(df_inventory_histories)} registros)")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 🧱 Preparação de Dados - Produtos
logger.info("=" * 60)
logger.info("Iniciando processamento: Produtos")
df_produtos = Helpers.getEntityData(context, 'produtos')
logger.info(f"Dados carregados: {len(df_produtos)} registros")

# Aplicar clean_whitespace em todas as colunas de texto
for col in df_produtos.columns:
    if df_produtos[col].dtype == 'object' or str(df_produtos[col].dtype) == 'string':
        df_produtos[col] = clean_whitespace(df_produtos, col)
        logger.info(f"Whitespace removido da coluna: {col}")

logger.info(f"Sanitização de produtos concluída: {len(df_produtos)} registros")

Helpers.save_output_dataset(
    context=context, output_name='df_produtos', data_frame=df_produtos)
logger.info(f"Dataset salvo: df_produtos ({len(df_produtos)} registros)")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 🧩 Compilação e Classificação de Componentes (Metadados)
logger.info("=" * 60)
logger.info("Iniciando processamento: Compilação e Classificação de Componentes")
df_sobdemanda = Helpers.getEntityData(context, 'produtos_sob_demanda')
logger.info(f"Produtos sob demanda carregados: {len(df_sobdemanda)} registros")
df_ativos = Helpers.getEntityData(context, 'produto_fornecedor_ativo')
logger.info(f"Produtos ativos carregados: {len(df_ativos)} registros")
df_excecoes = Helpers.getEntityData(
    context, 'excecoes_produtos_sem_fornecedores')
logger.info(f"Exceções carregadas: {len(df_excecoes)} registros")
df_parametro_fornecedor = Helpers.getEntityData(
    context, 'parametro_fornecedor')
logger.info(f"Parâmetros de fornecedor carregados: {len(df_parametro_fornecedor)} registros")

# --- Limpeza e Marcação ---
# (Manter a lógica de remoção de duplicatas e criação de colunas de flag)
df_ativos = df_ativos.drop_duplicates(
    subset=['Cod_Produto']).assign(Ativo=True)
df_excecoes = df_excecoes.drop_duplicates(
    subset=['Cod_Produto']).assign(Excecao=True)
df_sobdemanda = df_sobdemanda.drop_duplicates(
    subset=['Cod_Produto']).assign(Sob_Demanda=True)
logger.info("Duplicatas removidas e flags criadas")

# --- Junção dos Metadados ---
logger.info("Realizando merge dos metadados...")
df_compiled = df_ativos.merge(df_excecoes, how="outer", on="Cod_Produto")
df_compiled = df_compiled.merge(df_sobdemanda, how="outer", on="Cod_Produto")
df_compiled = df_compiled.fillna(
    {"Ativo": False, "Excecao": False, "Sob_Demanda": False})
logger.info(f"Após merge inicial: {len(df_compiled)} registros")

# --- Adicionar Histórico de Estoque e Regras de Negócio ---
components_inventory_histories = df_inventory_histories['Componente'].unique()
df_compiled["In_Inventory_Histories"] = df_compiled["Cod_Produto"].isin(
    components_inventory_histories)

df_compiled = df_compiled.merge(df_produtos, how="left", left_on=[
                                "Cod_Produto"], right_on=["B1_COD"])
logger.info(f"Após merge com produtos: {len(df_compiled)} registros")

# Aplicação de regras de negócio
df_compiled.loc[df_compiled["B1_ATIVO"] == "N", "Excecao"] = True
df_compiled.loc[df_compiled["Sob_Demanda"] == True, "Ativo"] = False
df_compiled.loc[df_compiled["Excecao"] == True, "Ativo"] = False
logger.info("Regras de negócio aplicadas")

df_compiled["Cod_Produto"] = clean_whitespace(df_compiled, "Cod_Produto")

# Aplica a função universal de filtragem
df_compiled = filter_t_components(df_compiled, 'Cod_Produto')
logger.info(f"Após filtro T-components: {len(df_compiled)} registros")

# 2. Aplica a função de remoção de linhas com dois pontos
df_compiled = remove_double_dot_lines(df_compiled, 'Cod_Produto')
logger.info(f"Após remoção de linhas com dois pontos: {len(df_compiled)} registros")

df_compiled["Cod_Fornecedor"] = (
    df_compiled["Cod_Fornecedor"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .str.replace("nan", "", regex=False)
)

Helpers.save_output_dataset(
    context=context, output_name='new_compiled_components2', data_frame=df_compiled)
logger.info(f"Dataset salvo: new_compiled_components2 ({len(df_compiled)} registros)")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 🧱 Preparação de Dados - Histórico de Pedidos
logger.info("=" * 60)
logger.info("Iniciando processamento: Histórico de Pedidos")
df_historico_pedidos = Helpers.getEntityData(context, "historico_pedidos")
logger.info(f"Dados carregados: {len(df_historico_pedidos)} registros")

# Remover espaços em branco da coluna PRODUTO
df_historico_pedidos['PRODUTO'] = clean_whitespace(df_historico_pedidos, 'PRODUTO')

# Padronizar COD_FORNE como string (para compatibilidade em merges)
df_historico_pedidos['COD_FORNE'] = df_historico_pedidos['COD_FORNE'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
logger.info("Coluna COD_FORNE padronizada como string")

Helpers.save_output_dataset(
    context=context, output_name='df_historico_pedidos', data_frame=df_historico_pedidos)
logger.info(f"Dataset salvo: df_historico_pedidos ({len(df_historico_pedidos)} registros)")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 🧱 Preparação de Dados - Pedidos Pendentes
logger.info("=" * 60)
logger.info("Iniciando processamento: Pedidos Pendentes")
df_pedidos_pendentes = Helpers.getEntityData(context, "pedidos_pendentes")
logger.info(f"Dados carregados: {len(df_pedidos_pendentes)} registros")

# Remover espaços em branco da coluna PRODUTO
df_pedidos_pendentes['PRODUTO'] = clean_whitespace(df_pedidos_pendentes, 'PRODUTO')

Helpers.save_output_dataset(
    context=context, output_name='df_pedidos_pendentes', data_frame=df_pedidos_pendentes)
logger.info(f"Dataset salvo: df_pedidos_pendentes ({len(df_pedidos_pendentes)} registros)")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 🧱 Preparação de Dados - Vendas
logger.info("=" * 60)
logger.info("Iniciando processamento: Vendas")
df_vendas = Helpers.getEntityData(context, "vendas")
logger.info(f"Dados carregados: {len(df_vendas)} registros")

# Remover espaços em branco da coluna B1_COD
df_vendas['B1_COD'] = clean_whitespace(df_vendas, 'B1_COD')

# Manter apenas componentes que comecam com a letra T
df_vendas = filter_t_components(df_vendas, 'B1_COD')
logger.info(f"Após filtro T-components: {len(df_vendas)} registros")

# 2. Aplica a função de remoção de linhas com dois pontos
df_vendas = remove_double_dot_lines(df_vendas, 'B1_COD')
logger.info(f"Após remoção de linhas com dois pontos: {len(df_vendas)} registros")

Helpers.save_output_dataset(
    context=context, output_name='df_vendas', data_frame=df_vendas)
logger.info(f"Dataset salvo: df_vendas ({len(df_vendas)} registros)")

logger.info("=" * 60)
logger.info("Sanitizer Recipe concluído com sucesso!")