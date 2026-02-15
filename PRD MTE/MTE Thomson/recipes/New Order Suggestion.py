# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Required imports

import math
from sklearn.metrics import make_scorer
from scipy.stats import norm
from dateutil import parser
import numpy as np
import logging
from datetime import datetime
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

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Carregar dados
df_inventory_histories = Helpers.getEntityData(
    context, "new_inventory_histories2")
df_estoque_atual = Helpers.getEntityData(context, "estoque_atual")
df_produtos = Helpers.getEntityData(context, "df_produtos")
df_estru_produtos = Helpers.getEntityData(context, "df_estrutura_produto")
df_products_not_delivered_exportation = Helpers.getEntityData(
    context, "df_products_not_delivered_exportation")
df_parametro_fornecedor = Helpers.getEntityData(
    context, "parametro_fornecedor")
df_parametro_fornecedor['Cod_Fornecedor'] = df_parametro_fornecedor['Cod_Fornecedor'].astype(str).str.strip()

# === DEBUG LOG: parametro_fornecedor ===
print(f"\n{'='*60}")
print(f"DEBUG: parametro_fornecedor")
print(f"{'='*60}")
print(f"  Linhas: {len(df_parametro_fornecedor)}")
print(f"  Colunas: {list(df_parametro_fornecedor.columns)}")
print(f"  Cod_Fornecedor dtype: {df_parametro_fornecedor['Cod_Fornecedor'].dtype}")
print(f"  Cod_Fornecedor unicos: {df_parametro_fornecedor['Cod_Fornecedor'].nunique()}")
_dup_pf = df_parametro_fornecedor[df_parametro_fornecedor.duplicated(subset=['Cod_Fornecedor'], keep=False)]
if len(_dup_pf) > 0:
    print(f"  *** DUPLICATAS em Cod_Fornecedor: {len(_dup_pf)} linhas ***")
    print(_dup_pf.to_string())
else:
    print(f"  Sem duplicatas em Cod_Fornecedor")
print(f"{'='*60}\n")

df_daily_portalvendas_components = Helpers.getEntityData(
    context, "new_daily_portalvendas_components")
df_multi_horizon_pred_component = Helpers.getEntityData(
    context, "new_multihorizon_components")
df_vendas = Helpers.getEntityData(context, "df_vendas")
df_produtos_sob_demanda = Helpers.getEntityData(
    context, "produtos_sob_demanda")
df_lista_excecoes_produtos_sem_fornecedores = Helpers.getEntityData(
    context, "excecoes_produtos_sem_fornecedores")
df_compiled_components = Helpers.getEntityData(
    context, "new_compiled_components2")
df_historico_pedidos = Helpers.getEntityData(context, "df_historico_pedidos")
df_pedidos_pendentes = Helpers.getEntityData(context, "df_pedidos_pendentes")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Inicializa lista de auditoria (único df_sem_match do pipeline)
lista_dfs_sem_match = []

df_produto_fornecedor_ativo = Helpers.getEntityData(
    context, "produto_fornecedor_ativo")
df_produto_fornecedor_original = Helpers.getEntityData(context, "df_produto_fornecedor")

# Padronizar tipos para merge (garantir compatibilidade string)
df_produto_fornecedor_ativo['Cod_Fornecedor'] = df_produto_fornecedor_ativo['Cod_Fornecedor'].astype(str).str.strip()
df_produto_fornecedor_ativo['Cod_Produto'] = df_produto_fornecedor_ativo['Cod_Produto'].astype(str).str.strip()

# === DEBUG LOG: produto_fornecedor_ativo ===
print(f"\n{'='*60}")
print(f"DEBUG: produto_fornecedor_ativo")
print(f"{'='*60}")
print(f"  Linhas: {len(df_produto_fornecedor_ativo)}")
print(f"  Colunas: {list(df_produto_fornecedor_ativo.columns)}")
print(f"  Cod_Produto dtype: {df_produto_fornecedor_ativo['Cod_Produto'].dtype}")
print(f"  Cod_Produto unicos: {df_produto_fornecedor_ativo['Cod_Produto'].nunique()}")
print(f"  Cod_Fornecedor dtype: {df_produto_fornecedor_ativo['Cod_Fornecedor'].dtype}")
_dup_pfa = df_produto_fornecedor_ativo[df_produto_fornecedor_ativo.duplicated(subset=['Cod_Produto'], keep=False)]
if len(_dup_pfa) > 0:
    print(f"  *** DUPLICATAS em Cod_Produto: {len(_dup_pfa)} linhas ***")
    print(_dup_pfa.to_string())
else:
    print(f"  Sem duplicatas em Cod_Produto")
print(f"{'='*60}\n")

df_produto_fornecedor_original['COD_FORNE'] = df_produto_fornecedor_original['COD_FORNE'].astype(str).str.strip()
df_produto_fornecedor_original['PRODUTO'] = df_produto_fornecedor_original['PRODUTO'].astype(str).str.strip()

# Verificar e tratar df_produto_fornecedor vazio
print(f"df_produto_fornecedor: {df_produto_fornecedor_original.shape[0]} linhas")

if df_produto_fornecedor_original.shape[0] > 0:
    # AUDITORIA DETALHADA: Identificar motivos específicos de não-match
    # Analisa ANTES de filtrar custo zero para identificar o motivo correto
    print("Analisando motivos de não-match...")

    for _, row in df_produto_fornecedor_ativo.iterrows():
        cod_produto = row['Cod_Produto']
        cod_fornecedor = row['Cod_Fornecedor']

        # Buscar o componente em produto_fornecedor (SEM filtro de custo)
        df_match_produto = df_produto_fornecedor_original[
            df_produto_fornecedor_original['PRODUTO'] == cod_produto
        ]

        if len(df_match_produto) == 0:
            # Caso 1: Componente não existe em produto_fornecedor
            lista_dfs_sem_match.append({
                'Cod_component': cod_produto,
                'Cod_Fornecedor_Ativo': cod_fornecedor,
                'origem': 'produto_fornecedor_ativo.csv e produto_fornecedor.csv',
                'motivo': 'Componente nao existe em produto_fornecedor.csv'
            })
        else:
            # Componente existe, verificar se o fornecedor ativo está cadastrado
            df_match_fornecedor = df_match_produto[
                df_match_produto['COD_FORNE'] == cod_fornecedor
            ]

            if len(df_match_fornecedor) == 0:
                # Caso 2: Fornecedor ativo não está cadastrado para este componente
                fornecedores_disponiveis = ', '.join(df_match_produto['COD_FORNE'].unique())
                lista_dfs_sem_match.append({
                    'Cod_component': cod_produto,
                    'Cod_Fornecedor_Ativo': cod_fornecedor,
                    'origem': 'produto_fornecedor_ativo.csv e produto_fornecedor.csv',
                    'motivo': f'Fornecedor ativo ({cod_fornecedor}) nao cadastrado em produto_fornecedor. Disponiveis: {fornecedores_disponiveis}'
                })
            else:
                # Fornecedor existe, verificar custo
                custo = df_match_fornecedor['CUSTO_PRODUTO'].values[0]
                if custo == 0 or pd.isna(custo):
                    # Caso 3: Custo é zero ou nulo
                    lista_dfs_sem_match.append({
                        'Cod_component': cod_produto,
                        'Cod_Fornecedor_Ativo': cod_fornecedor,
                        'origem': 'produto_fornecedor_ativo.csv e produto_fornecedor.csv',
                        'motivo': f'CUSTO_PRODUTO = 0 para o fornecedor ativo ({cod_fornecedor})'
                    })

    # Converter lista de dicts para DataFrame
    if lista_dfs_sem_match:
        df_audit_temp = pd.DataFrame(lista_dfs_sem_match)
        lista_dfs_sem_match = [df_audit_temp]
        print(f"⚠️ {len(df_audit_temp)} componentes com problemas de cadastro identificados")

    # Agora aplica os filtros para o processamento normal
    # 1. FILTRO CUSTO ZERO
    mask_custo_zero = df_produto_fornecedor_original['CUSTO_PRODUTO'] == 0.0
    df_produto_fornecedor = df_produto_fornecedor_original[~mask_custo_zero].copy()

    # 2. Remover componentes sem match de df_produto_fornecedor_ativo
    if lista_dfs_sem_match and len(lista_dfs_sem_match[0]) > 0:
        produtos_sem_match = set(lista_dfs_sem_match[0]['Cod_component'].values)
        df_produto_fornecedor_ativo = df_produto_fornecedor_ativo[
            ~df_produto_fornecedor_ativo['Cod_Produto'].isin(produtos_sem_match)
        ]
        print(f"⚠️ {len(produtos_sem_match)} componentes removidos de produto_fornecedor_ativo")

    # Aplica o merge real (Inner)
    df_produto_fornecedor = (
        df_produto_fornecedor
        .merge(
            df_produto_fornecedor_ativo,
            left_on=['PRODUTO', 'COD_FORNE'],
            right_on=['Cod_Produto', 'Cod_Fornecedor'],
            how='inner'
        )[df_produto_fornecedor.columns]
    )
    print(f"df_produto_fornecedor após filtros: {df_produto_fornecedor.shape[0]} linhas")
else:
    df_produto_fornecedor = df_produto_fornecedor_original.copy()
    print("⚠️ df_produto_fornecedor vazio - custos não estarão disponíveis")

# Criar coluna In_Multihorizon_Predictions se não existir
if 'In_Multihorizon_Predictions' not in df_compiled_components.columns:
    print("Criando coluna In_Multihorizon_Predictions...")
    components_with_predictions = set(
        df_multi_horizon_pred_component["Component"].unique())
    df_compiled_components['In_Multihorizon_Predictions'] = df_compiled_components['Cod_Produto'].isin(
        components_with_predictions)
    print(
        f"Componentes com previsões: {df_compiled_components['In_Multihorizon_Predictions'].sum()}/{len(df_compiled_components)}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE


def calculate_mean_lead_time(df_historico_pedidos, component_code):
    df_hist = df_historico_pedidos[df_historico_pedidos["PRODUTO"]
                                   == component_code].copy()
    if df_hist.shape[0] == 0:
        return np.nan, np.nan

    df_hist["Lead_Time"] = (df_hist["ENTREGA_EFET"] -
                            df_hist["DATA_SI"]).dt.days
    avg_lt = df_hist["Lead_Time"].mean()
    sd_lt = df_hist["Lead_Time"].std()
    return avg_lt, sd_lt


def calculate_safety_stock(
    date,
    component_code,
    df_consumption_true,
    df_historico_pedidos,
    lt,
    service_level=0.95,
    method=4,
    safety_factor=1,
    max_lookback=999,
    max_lookback_orders=999,
    max_lookback_method="months"
):
    Z = norm.ppf(service_level)
    df_filtered = df_consumption_true[df_consumption_true["Component"]
                                      == component_code].copy()

    if df_filtered.shape[0] == 0:
        return 0

    df_filtered = df_filtered[df_filtered['DATA_PEDIDO'] < date]

    if max_lookback_method in ["orders", "both"]:
        df_filtered = df_filtered.sort_values(
            by='DATA_PEDIDO', ascending=False)
        df_filtered = df_filtered.head(max_lookback_orders)

    if max_lookback_method in ["months", "both"]:
        df_filtered = df_filtered[df_filtered['DATA_PEDIDO']
                                  >= date - pd.DateOffset(months=max_lookback)]

    avg_lt, std_lt = calculate_mean_lead_time(
        df_historico_pedidos, component_code)
    avg_consumption = df_filtered["Consumption"].mean()
    std_consumption = df_filtered["Consumption"].std()

    if method == 3:
        safety_stock = Z * std_consumption * np.sqrt(avg_lt)
    elif method == 4:
        # Conversão: avg_consumption está em unidades/MÊS, lead times em DIAS
        # Dividir lead times por 30 para converter para MESES
        if pd.isna(avg_lt):
            safety_stock = avg_consumption * (lt / 30)
        elif pd.isna(std_lt):
            safety_stock = Z * avg_consumption * (avg_lt / 30)
        else:
            safety_stock = Z * avg_consumption * (std_lt / 30)
    elif method == 5:
        safety_stock = Z * \
            math.sqrt((avg_lt * (std_consumption) ** 2 +
                      (avg_consumption * std_lt) ** 2))
    else:
        safety_stock = 0

    safety_stock = safety_stock * safety_factor
    return safety_stock if not pd.isna(safety_stock) else 0


def get_in_transit_orders(date, component, df_historico_pedidos):
    df_hist = df_historico_pedidos[df_historico_pedidos["PRODUTO"] == component].copy(
    )
    df_hist = df_hist[df_hist["DATA_SI"] <= date]
    df_hist = df_hist[(df_hist["ENTREGA_EFET"].isna()) |
                      (df_hist["ENTREGA_EFET"] > date)]
    return df_hist


def get_in_transit_orders_from_pendentes(date, component, df_pedidos_pendentes):
    """
    Busca pedidos em trânsito a partir de pedidos_pendentes.
    Essa fonte é mais completa que historico_pedidos para pedidos não entregues.

    Args:
        date: Data de referência
        component: Código do componente
        df_pedidos_pendentes: DataFrame com pedidos pendentes

    Returns:
        DataFrame com pedidos em trânsito, com coluna QUANTIDADE para compatibilidade
    """
    df_pend = df_pedidos_pendentes[df_pedidos_pendentes["PRODUTO"] == component].copy(
    )
    if len(df_pend) > 0:
        df_pend["DATA_SI"] = pd.to_datetime(df_pend["DATA_SI"])
        df_pend = df_pend[df_pend["DATA_SI"] <= date]
        # Renomear coluna para compatibilidade com código existente
        df_pend = df_pend.rename(columns={"QTDE_NAO_ENTREGUE": "QUANTIDADE"})
    return df_pend


def get_demand_ltrp(base_date, df_forecast, lt_rp, component):
    df_forecast = df_forecast[
        (df_forecast["Component"] == component) &
        (df_forecast["base_date"] == base_date)
    ].copy()

    if df_forecast.shape[0] == 0:
        return 0.0

    df_forecast_low = df_forecast[
        df_forecast['DATA_PEDIDO'] <= base_date +
        pd.Timedelta(lt_rp, unit="D") + pd.offsets.MonthEnd(0)
    ]

    if df_forecast_low.shape[0] == 0:
        return 0.0

    forecast_span_low = (
        base_date + pd.Timedelta(lt_rp, unit="D") +
        pd.offsets.MonthEnd(0) - df_forecast_low['DATA_PEDIDO'].min()
    )
    forecast_span_low_days = forecast_span_low.days

    df_forecast_high = df_forecast[
        df_forecast['DATA_PEDIDO'] <= base_date +
        pd.Timedelta(lt_rp, unit="D") + pd.offsets.MonthEnd(2)
    ]

    if df_forecast_high.shape[0] == 0:
        return df_forecast_low["consumption_predicted_month"].sum()

    forecast_span_high = (
        base_date + pd.Timedelta(lt_rp, unit="D") + pd.offsets.MonthEnd(2) -
        df_forecast_high['DATA_PEDIDO'].min()
    )
    forecast_span_high_days = forecast_span_high.days

    demand_low = df_forecast_low["consumption_predicted_month"].sum()
    demand_high = df_forecast_high["consumption_predicted_month"].sum()

    if forecast_span_high_days == forecast_span_low_days:
        demand = demand_low
    else:
        demand = demand_low + ((demand_high - demand_low) / (forecast_span_high_days -
                               forecast_span_low_days)) * (lt_rp - forecast_span_low_days)

    return 0.0 if pd.isna(demand) else demand


def calculate_lead_time_by_supplier(df_historico_pedidos):
    df_hist = df_historico_pedidos.copy()
    df_hist["Calculated_Lead_Time"] = (
        df_hist["ENTREGA_EFET"] - df_hist["DATA_SI"]).dt.days
    df_lt_supplier = df_hist.groupby(
        "COD_FORNE")["Calculated_Lead_Time"].mean().reset_index()
    return df_lt_supplier


def create_df_lt_rp(
    available_components,
    df_produto_fornecedor,
    df_produto_fornecedor_ativo_default,
    df_parametro_fornecedor_default,
    df_historico_pedidos
):
    print(f"\n{'='*60}")
    print(f"DEBUG: DENTRO de create_df_lt_rp")
    print(f"{'='*60}")
    print(f"  Input produto_fornecedor_ativo: {len(df_produto_fornecedor_ativo_default)} linhas")
    print(f"    Cod_Produto unicos: {df_produto_fornecedor_ativo_default['Cod_Produto'].nunique()}")
    print(f"    Cod_Produto dtype: {df_produto_fornecedor_ativo_default['Cod_Produto'].dtype}")
    print(f"    Cod_Fornecedor dtype: {df_produto_fornecedor_ativo_default['Cod_Fornecedor'].dtype}")
    _dup_input = df_produto_fornecedor_ativo_default[df_produto_fornecedor_ativo_default.duplicated(subset=['Cod_Produto'], keep=False)]
    if len(_dup_input) > 0:
        print(f"    *** DUPLICATAS Cod_Produto na entrada: {len(_dup_input)} ***")
        print(_dup_input[['Cod_Produto', 'Cod_Fornecedor']].head(20).to_string())

    print(f"  Input parametro_fornecedor: {len(df_parametro_fornecedor_default)} linhas")
    print(f"    Cod_Fornecedor unicos: {df_parametro_fornecedor_default['Cod_Fornecedor'].nunique()}")
    print(f"    Cod_Fornecedor dtype: {df_parametro_fornecedor_default['Cod_Fornecedor'].dtype}")
    print(f"    Colunas: {list(df_parametro_fornecedor_default.columns)}")
    _dup_param = df_parametro_fornecedor_default[df_parametro_fornecedor_default.duplicated(subset=['Cod_Fornecedor'], keep=False)]
    if len(_dup_param) > 0:
        print(f"    *** DUPLICATAS Cod_Fornecedor no parametro: {len(_dup_param)} ***")
        print(_dup_param.to_string())

    df_lt_supplier = calculate_lead_time_by_supplier(df_historico_pedidos)
    print(f"  df_lt_supplier: {len(df_lt_supplier)} linhas, COD_FORNE dtype: {df_lt_supplier['COD_FORNE'].dtype}")

    df_lt_rp = df_produto_fornecedor_ativo_default.merge(
        df_parametro_fornecedor_default,
        left_on="Cod_Fornecedor",
        right_on="Cod_Fornecedor",
        how="left",
    )
    print(f"  Apos merge 1 (ativo x param): {len(df_lt_rp)} linhas")
    _dup_m1 = df_lt_rp[df_lt_rp.duplicated(subset=['Cod_Produto'], keep=False)]
    if len(_dup_m1) > 0:
        print(f"    *** DUPLICATAS Cod_Produto apos merge 1: {len(_dup_m1)} ***")
        print(_dup_m1[['Cod_Produto', 'Cod_Fornecedor', 'Lead_Time']].head(20).to_string())

    df_lt_rp = df_lt_rp.merge(
        df_lt_supplier, how='left', left_on='Cod_Fornecedor', right_on='COD_FORNE')
    print(f"  Apos merge 2 (+ lt_supplier): {len(df_lt_rp)} linhas")
    _dup_m2 = df_lt_rp[df_lt_rp.duplicated(subset=['Cod_Produto'], keep=False)]
    if len(_dup_m2) > 0:
        print(f"    *** DUPLICATAS Cod_Produto apos merge 2: {len(_dup_m2)} ***")
        print(_dup_m2[['Cod_Produto', 'Cod_Fornecedor', 'Lead_Time']].head(20).to_string())

    df_lt_rp = df_lt_rp[["Cod_Produto", "Cod_Fornecedor",
                         "Review_Period", "Lead_Time", "Calculated_Lead_Time"]]
    df_lt_rp = df_lt_rp.rename(
        columns={
            "Cod_Produto": "Component",
            "Cod_Fornecedor": "Supplier",
            "Review_Period": "Review_Period",
            "Lead_Time": "Lead_Time",
        }
    )

    df_lt_rp = df_lt_rp.sort_values(by=["Lead_Time", "Supplier"])
    print(f"  Antes drop_duplicates: {len(df_lt_rp)} linhas")
    df_lt_rp = df_lt_rp.drop_duplicates(subset=["Component"], keep="first")
    print(f"  Apos drop_duplicates: {len(df_lt_rp)} linhas")
    df_lt_rp = df_lt_rp.set_index("Component")
    df_lt_rp = df_lt_rp.fillna(35)

    print(f"  Index unico: {df_lt_rp.index.is_unique}")
    if not df_lt_rp.index.is_unique:
        _dup_idx = df_lt_rp.index[df_lt_rp.index.duplicated(keep=False)]
        print(f"  *** INDEX DUPLICADO: {len(_dup_idx)} entradas ***")
        print(f"  Valores duplicados: {_dup_idx.unique().tolist()[:20]}")
        print(df_lt_rp.loc[_dup_idx].head(20).to_string())
    print(f"  Lead_Time dtype: {df_lt_rp['Lead_Time'].dtype}")
    print(f"  Review_Period dtype: {df_lt_rp['Review_Period'].dtype}")
    print(f"{'='*60}\n")

    return df_lt_rp


def adjust_data_row(data_row):
    if data_row["Total Stock"] > data_row["Sales 12M"]:
        data_row["Order sug"] = 0
        data_row["Final order"] = 0
    return data_row


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def create_main_dataframe(
    df_daily_sales,
    df_vendas,
    df_produtos,
    df_produto_fornecedor,
    df_produto_fornecedor_ativo,
    df_excecoes,
    df_sob_demanda,
    df_order_history,
    df_estoque_atual,
    df_inventory_histories,
    df_lt_rp,
    df_forecasts,
    base_date,
    base_month,
    df_compiled_components,
    usage_date: pd.Timestamp,
    days_since_base,
    df_pedidos_pendentes=None  # Novo: fonte para pedidos em trânsito
):
    df_vendas["DATA"] = pd.to_datetime(df_vendas["DATA"])
    df_vendas_12meses = df_vendas[df_vendas["DATA"]
                                  >= (base_date - pd.DateOffset(months=12))]

    df_main = pd.DataFrame()

    # 3. FILTRO: COMPONENTES SEM ESTOQUE ATUAL
    mask_in_stock = df_compiled_components["Cod_Produto"].isin(
        df_estoque_atual["CODIGO_PI"])
    df_compiled_components = df_compiled_components[mask_in_stock]

    current_date = base_date + pd.DateOffset(days_since_base, "D")
    max_available_date = df_inventory_histories["date"].max()
    actual_inventory_date = min(current_date, max_available_date)

    if current_date > max_available_date:
        print(
            f"Warning: Using latest available date: {actual_inventory_date.date()}")

    # Pivotar para performance
    print("Criando pivot do histórico de estoque...")
    df_inventory_pivot = df_inventory_histories.pivot(
        index="date", columns="Componente", values="QTD_ESTOQUE"
    ).fillna(0)

    # Lista para coletar componentes sem histórico (Filtro 4)
    missing_history_list = []

    for i, row in df_compiled_components.iterrows():
        component = row["Cod_Produto"]
        data_row = {}

        # 4. FILTRO: SEM HISTÓRICO DE ESTOQUE
        if component not in df_inventory_pivot.columns:
            missing_history_list.append(component)
            continue

        df_vendas_comp = df_vendas_12meses[df_vendas_12meses["B1_COD"] == component]
        total_sales_repo = df_vendas_comp["QTD_VENDA_NACIONAL"].sum()
        total_sales_expo = df_vendas_comp["QTD_VENDA_EXPORTACAO"].sum()
        total_sales = total_sales_repo + total_sales_expo
        data_row["Sales 12M"] = total_sales

        pctg_exportacao = total_sales_expo / total_sales if total_sales > 0 else 0
        data_row[r"% Export 12M"] = pctg_exportacao * 100

        # Vendas perdidas
        df_inv_comp = df_inventory_histories[df_inventory_histories["Componente"] == component]
        df_inv_comp = df_inv_comp[df_inv_comp["date"]
                                  >= (base_date - pd.DateOffset(months=12))]
        lost_sales_days = df_inv_comp[df_inv_comp["QTD_ESTOQUE"] == 0].shape[0]
        avg_daily_sales = total_sales / 365
        data_row["Lost 12M"] = lost_sales_days * avg_daily_sales

        unfullfilled_orders_repo = df_vendas_comp["NAO_ATENDIDOS_REPO"].sum()
        unfullfilled_orders_expo = df_vendas_comp["NAO_ATENDIDOS_EXPO"].sum()
        data_row["Unfulfilled 12M"] = unfullfilled_orders_repo + \
            unfullfilled_orders_expo

        # Estoque do histórico
        current_stock = df_inventory_pivot.loc[actual_inventory_date, component]
        data_row["Stock"] = current_stock
        inventory_level = df_inventory_pivot.loc[actual_inventory_date, component]

        # Buscar dados de estoque atual
        df_estoque_comp = df_estoque_atual[df_estoque_atual["CODIGO_PI"] == component]

        if len(df_estoque_comp) > 0:
            # alterado para usar QTD_TOT_EST_INSP ao invés QTD_TOT_EST
            inspection = df_estoque_comp["QTD_TOT_EST_INSP"].values[0]
            reserved = df_estoque_comp["QTD_RESE"].values[0]
            if inspection < 0:
                print(f"⚠️ Inspection negativo para {component}: {inspection}")
        else:
            inspection = 0
            reserved = 0

        # Pedidos em trânsito (usando pedidos_pendentes como fonte principal)
        if df_pedidos_pendentes is not None:
            df_in_transit = get_in_transit_orders_from_pendentes(
                current_date, component, df_pedidos_pendentes)
        else:
            df_in_transit = get_in_transit_orders(
                current_date, component, df_order_history)
        transit = df_in_transit["QUANTIDADE"].sum(
        ) if "QUANTIDADE" in df_in_transit.columns else 0

        # Cálculo do total
        total_stock = inventory_level + inspection - reserved + transit

        data_row["Inventory level"] = inventory_level
        data_row["Inspection"] = inspection
        data_row["Reserved"] = reserved
        data_row["Total Stock"] = total_stock
        data_row["Transit"] = transit

        component_has_active_supplier = component in df_lt_rp.index

        if component_has_active_supplier:
            df_forn_ativo_comp = df_produto_fornecedor_ativo[
                df_produto_fornecedor_ativo["Cod_Produto"] == component
            ]

            if len(df_forn_ativo_comp) > 0:
                active_supplier_code = df_forn_ativo_comp["Cod_Fornecedor"].values[0]
                active_supplier_name = df_forn_ativo_comp["Nome_Fornecedor"].values[0] if not pd.isna(
                    df_forn_ativo_comp["Nome_Fornecedor"].values[0]) else f"Fornecedor_{active_supplier_code}"
                active_cod_x = df_forn_ativo_comp["Cod_X"].values[0] if "Cod_X" in df_forn_ativo_comp.columns else None

                # Buscar custo
                if df_produto_fornecedor.shape[0] > 0:
                    df_custo = df_produto_fornecedor[
                        (df_produto_fornecedor["PRODUTO"] == component) &
                        (df_produto_fornecedor["COD_FORNE"]
                         == active_supplier_code)
                    ]
                    if len(df_custo) > 0:
                        cost = df_custo["CUSTO_PRODUTO"].values[0]
                        currency = df_custo["MOEDA"].values[0]
                    else:
                        cost = None
                        currency = None
                else:
                    cost = None
                    currency = None

                # === DEBUG: verificar tipo antes de int() ===
                _lt_raw = df_lt_rp.loc[component, "Lead_Time"]
                if isinstance(_lt_raw, pd.Series):
                    print(f"*** BUG DETECTADO: df_lt_rp.loc[{component}, 'Lead_Time'] retornou Series (len={len(_lt_raw)}) ***")
                    print(f"    Valores: {_lt_raw.values.tolist()}")
                    print(f"    Index entries para {component}: {df_lt_rp.index[df_lt_rp.index == component].tolist()}")
                    print(f"    Index dtype: {df_lt_rp.index.dtype}, component type: {type(component)}")
                    _lt_raw = _lt_raw.iloc[0]
                lt = int(_lt_raw)

                usage_date_month = usage_date.month
                extra_rp = 0
                if str(active_supplier_code) != '2102':
                    if usage_date_month == 10:
                        extra_rp = 55
                    elif usage_date_month == 11:
                        extra_rp = 30

                _rp_raw = df_lt_rp.loc[component, "Review_Period"]
                if isinstance(_rp_raw, pd.Series):
                    print(f"*** BUG DETECTADO: df_lt_rp.loc[{component}, 'Review_Period'] retornou Series (len={len(_rp_raw)}) ***")
                    print(f"    Valores: {_rp_raw.values.tolist()}")
                    _rp_raw = _rp_raw.iloc[0]
                rp = int(_rp_raw) + extra_rp
                ltrp = lt + rp

                has_prediction = row.get("In_Multihorizon_Predictions", False)

                if has_prediction:
                    if component not in df_sob_demanda["Cod_Produto"].values:
                        df_prod_comp = df_produtos[df_produtos["B1_COD"]
                                                   == component]
                        if len(df_prod_comp) > 0:
                            component_group = df_prod_comp["B1_GRUPO"].values[0]
                            mlb = 18 if component_group in ["TC28"] else 999
                            safety_stock = int(calculate_safety_stock(
                                base_date, component, df_daily_sales,
                                df_order_history, lt, method=4, max_lookback=mlb
                            ))
                        else:
                            safety_stock = 0
                    else:
                        safety_stock = 0

                    demand_ltrp = int(get_demand_ltrp(
                        base_month, df_forecasts, ltrp, component))
                else:
                    safety_stock = 0
                    demand_ltrp = 0

                order_sug = int(safety_stock - total_stock + demand_ltrp)
                order_sug = max(order_sug, 0)

                df_prod_comp = df_produtos[df_produtos["B1_COD"] == component]
                if len(df_prod_comp) > 0:
                    multiplier = int(df_prod_comp["MULTIPLO_COMPRA"].values[0])
                    component_group = df_prod_comp["B1_GRUPO"].values[0]
                    if component_group in ["TC38"]:
                        multiplier = 12
                    order_sug = int(
                        np.ceil(order_sug / multiplier) * multiplier)
                else:
                    multiplier = 1
            else:
                active_supplier_code = None
                active_supplier_name = None
                active_cod_x = None
                lt = 100
                rp = 35
                ltrp = 135
                safety_stock = None
                demand_ltrp = 0
                order_sug = None
                cost = None
                currency = None
                multiplier = None
        else:
            # Componente não está em df_lt_rp - não deveria chegar aqui
            # pois df_compiled_components já foi filtrado, mas por segurança:
            continue

        data_row["Supp Cod"] = active_supplier_code
        data_row["Supplier"] = active_supplier_name
        data_row["Cod X"] = active_cod_x
        data_row["LT"] = lt
        data_row["RP"] = rp
        data_row["LT+RP"] = lt + rp
        data_row["Safety stock"] = safety_stock
        data_row["Demand (LT+RP)"] = demand_ltrp
        data_row["On demand"] = component in df_sob_demanda["Cod_Produto"].values

        if component not in df_excecoes["Cod_Produto"].values:
            data_row["Order sug"] = order_sug
        else:
            data_row["Order sug"] = 0

        data_row["Final order"] = data_row["Order sug"]
        data_row["Cost"] = cost
        data_row["Currency"] = currency
        data_row["Component"] = component
        data_row["Multiplier"] = multiplier

        data_row = adjust_data_row(data_row)

        new_row = pd.DataFrame(data=data_row, index=[component])
        df_main = pd.concat([df_main, new_row])

    dict_produtos_cols = {
        "B1_COD": "B1_COD",
        "B1_DESC": "Description",
        "B1_GRUPO": "Group code",
        "NOM_GRUP": "Group name",
        "ORIGEM": "Origin",
        "KANBAN_MIN": "KanBan Min",
        "KANBAN_MAX": "KanBan Max",
        "CURVA": "ABC",
    }

    df_produtos_select = df_produtos[list(dict_produtos_cols.keys())].copy()
    df_produtos_select = df_produtos_select.set_index("B1_COD")
    df_produtos_select = df_produtos_select.rename(columns=dict_produtos_cols)

    df_main = df_produtos_select.merge(
        df_main, left_index=True, right_index=True, how="right")
    df_main["Total Cost"] = df_main["Final order"] * df_main["Cost"]

    df_main = df_main.reindex(
        columns=[
            "Component",
            "Description", "Group code", "Group name", "Supp Cod", "Supplier", "Cod X", "ABC",
            "Stock", "Transit", "Inspection", "Reserved", "Total Stock",
            "Sales 12M", "Lost 12M", "Unfulfilled 12M",
            "KanBan Min", "KanBan Max", "Order sug", "Final order",
            "Total Stock/Sales 12M", "Order sug/Sales 12M", r"% Export 12M",
            "Safety stock", "RP", "LT", "LT+RP", "Inventory level", "Demand (LT+RP)",
            "On demand", "Origin", "Cost", "Currency", "Multiplier", "Total Cost",
        ]
    )

    df_main["Total Stock/Sales 12M"] = df_main.apply(
        lambda row: (row["Total Stock"] / row["Sales 12M"]
                     * 100) if row["Sales 12M"] > 0 else 0,
        axis=1
    )

    df_main["Order sug/Sales 12M"] = df_main.apply(
        lambda row: (row["Order sug"] / row["Sales 12M"]
                     * 100) if row["Sales 12M"] > 0 else 0,
        axis=1
    )

    return df_main, missing_history_list


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Preparar dados
df_inventory_histories["date"] = pd.to_datetime(df_inventory_histories["date"])
df_inventory_histories = df_inventory_histories.sort_values(
    ["date", "Componente"])
latest_date = df_inventory_histories["date"].max()

# Processar df_code_supplier (com tratamento para df_produto_fornecedor vazio)
if df_produto_fornecedor.shape[0] > 0:
    df_code_supplier = df_produto_fornecedor[[
        "COD_FORNE", "FORNEC_NOM"]].drop_duplicates()
    df_produto_fornecedor_ativo["Nome_Fornecedor"] = df_produto_fornecedor_ativo["Cod_Fornecedor"].map(
        df_code_supplier.set_index("COD_FORNE")["FORNEC_NOM"]
    )
else:
    print("⚠️ Criando Nome_Fornecedor genérico")
    df_produto_fornecedor_ativo["Nome_Fornecedor"] = "Fornecedor_" + \
        df_produto_fornecedor_ativo["Cod_Fornecedor"].astype(str)

df_products_not_delivered_exportation['PRODUTO'] = df_products_not_delivered_exportation['PRODUTO'].str.strip(
)

df_components_not_delivered_exportation = df_products_not_delivered_exportation.merge(
    df_estru_produtos, how='left', right_on='COD_PRODUTO', left_on='PRODUTO'
)
df_components_not_delivered_exportation["NAO_ATENDIDOS_EXPO"] = (
    df_components_not_delivered_exportation["G1_QUANT"] *
    df_components_not_delivered_exportation["QUANT_BO"]
)

df_components_not_delivered_exportation_grouped = (
    df_components_not_delivered_exportation
    .groupby(['COD_COMPONENTE', 'DT_ENTREGA'])['NAO_ATENDIDOS_EXPO']
    .sum()
    .reset_index()
)

df_components_not_delivered_exportation_grouped["QTD_VENDA_NACIONAL"] = 0
df_components_not_delivered_exportation_grouped["RECEITA_VENDA_NACIONAL"] = 0
df_components_not_delivered_exportation_grouped["QTD_VENDA_EXPORTACAO"] = 0
df_components_not_delivered_exportation_grouped["RECEITA_VENDA_EXPORTACAO"] = 0
df_components_not_delivered_exportation_grouped["NAO_ATENDIDOS_REPO"] = 0

columns = df_components_not_delivered_exportation_grouped.columns.tolist()
columns.remove("NAO_ATENDIDOS_EXPO")
columns.append("NAO_ATENDIDOS_EXPO")
df_components_not_delivered_exportation_grouped = df_components_not_delivered_exportation_grouped.reindex(
    columns, axis=1)

df_components_not_delivered_exportation_grouped.rename(
    columns={"COD_COMPONENTE": "B1_COD", "DT_ENTREGA": "DATA"},
    inplace=True
)

df_vendas_with_exportacao = pd.concat(
    [df_vendas, df_components_not_delivered_exportation_grouped], ignore_index=True)
df_vendas_with_exportacao = (
    df_vendas_with_exportacao
    .groupby(['B1_COD', 'DATA'])[
        ["QTD_VENDA_NACIONAL", "RECEITA_VENDA_NACIONAL", "QTD_VENDA_EXPORTACAO",
         "RECEITA_VENDA_EXPORTACAO", "NAO_ATENDIDOS_REPO", "NAO_ATENDIDOS_EXPO"]
    ]
    .sum()
    .reset_index()
)

df_historico_pedidos["DATA_SI"] = pd.to_datetime(
    df_historico_pedidos["DATA_SI"])
df_historico_pedidos["ENTREGA_EFET"] = pd.to_datetime(
    df_historico_pedidos["ENTREGA_EFET"])

df_lt_rp = create_df_lt_rp(
    df_multi_horizon_pred_component["Component"].unique(),
    df_produto_fornecedor,
    df_produto_fornecedor_ativo,
    df_parametro_fornecedor,
    df_historico_pedidos
)

# === DEBUG LOG: df_lt_rp pos-criacao ===
print(f"\n{'='*60}")
print(f"DEBUG: df_lt_rp APOS create_df_lt_rp")
print(f"{'='*60}")
print(f"  Shape: {df_lt_rp.shape}")
print(f"  Index unico: {df_lt_rp.index.is_unique}")
print(f"  Index dtype: {df_lt_rp.index.dtype}")
print(f"  Lead_Time dtype: {df_lt_rp['Lead_Time'].dtype}")
print(f"  Review_Period dtype: {df_lt_rp['Review_Period'].dtype}")
if not df_lt_rp.index.is_unique:
    _dup_final = df_lt_rp.index[df_lt_rp.index.duplicated(keep=False)]
    print(f"  *** PROBLEMA: {len(_dup_final)} entradas duplicadas no index ***")
    print(f"  Primeiros duplicados: {_dup_final.unique().tolist()[:10]}")
    for _comp in _dup_final.unique()[:5]:
        print(f"    {_comp}: Lead_Time = {df_lt_rp.loc[_comp, 'Lead_Time']}")
        print(f"      tipo = {type(df_lt_rp.loc[_comp, 'Lead_Time'])}")
print(f"  Amostra (5 primeiros):")
print(df_lt_rp.head().to_string())
print(f"{'='*60}\n")

base_months = list(df_multi_horizon_pred_component["base_date"].unique())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# CORREÇÃO: Filtrar df_compiled_components para manter APENAS componentes
# que estão em df_lt_rp (têm fornecedor ativo E custo cadastrado)
# Isso remove tanto os auditados (sem custo) quanto os que nunca tiveram fornecedor
componentes_validos = set(df_lt_rp.index)
antes = len(df_compiled_components)
df_compiled_components = df_compiled_components[
    df_compiled_components['Cod_Produto'].isin(componentes_validos)
]
depois = len(df_compiled_components)
print(f"📋 df_compiled_components filtrado por df_lt_rp: {antes} → {depois} ({antes - depois} removidos)")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_main_all_bases = pd.DataFrame()
usage_date = latest_date

for base_month in base_months:
    days_since_base = 0
    base_date = base_month + pd.offsets.MonthEnd(0)

    print(f"\nProcessando base_month: {pd.Timestamp(base_month).date()}")

    df_main, components_without_inventory_history = create_main_dataframe(
        df_daily_portalvendas_components,
        df_vendas,
        df_produtos,
        df_produto_fornecedor,
        df_produto_fornecedor_ativo,
        df_lista_excecoes_produtos_sem_fornecedores,
        df_produtos_sob_demanda,
        df_historico_pedidos,
        df_estoque_atual,
        df_inventory_histories,
        df_lt_rp,
        df_multi_horizon_pred_component,
        base_date,
        base_month,
        df_compiled_components,
        usage_date,
        days_since_base,
        df_pedidos_pendentes=df_pedidos_pendentes  # Fonte correta para Transit
    )

    df_main["base_month"] = base_month
    df_main["base_date"] = base_date
    df_main_all_bases = pd.concat(
        [df_main_all_bases, df_main], ignore_index=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
last_base_month = latest_date
last_available_date = df_inventory_histories["date"].max()

print(f"\nProcessando última data: {pd.Timestamp(last_available_date).date()}")

df_main_last, _ = create_main_dataframe(
    df_daily_portalvendas_components,
    df_vendas,
    df_produtos,
    df_produto_fornecedor,
    df_produto_fornecedor_ativo,
    df_lista_excecoes_produtos_sem_fornecedores,
    df_produtos_sob_demanda,
    df_historico_pedidos,
    df_estoque_atual,
    df_inventory_histories,
    df_lt_rp,
    df_multi_horizon_pred_component,
    last_available_date,
    last_base_month,
    df_compiled_components,
    usage_date,
    0,
    df_pedidos_pendentes=df_pedidos_pendentes  # Fonte correta para Transit
)

df_main_last["base_month"] = last_base_month
df_main_last["base_date"] = last_available_date
df_main_all_bases = pd.concat(
    [df_main_all_bases, df_main_last], ignore_index=True)

df_main_all_bases = df_main_all_bases.reset_index(drop=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# CONSOLIDAÇÃO DO DATAFRAME DE AUDITORIA (df_sem_match)

if len(lista_dfs_sem_match) > 0:
    df_sem_match = pd.concat(lista_dfs_sem_match, ignore_index=True)
else:
    df_sem_match = pd.DataFrame(columns=['Cod_component', 'Cod_Fornecedor_Ativo', 'origem', 'motivo'])

# Garantir a ordem das colunas e remover duplicatas
colunas_sem_match = ['Cod_component', 'Cod_Fornecedor_Ativo', 'origem', 'motivo']
for col in colunas_sem_match:
    if col not in df_sem_match.columns:
        df_sem_match[col] = None
df_sem_match = df_sem_match[colunas_sem_match].drop_duplicates()

# Resumo por tipo de problema
print(f"\n📋 RESUMO DE COMPONENTES SEM MATCH:")
if len(df_sem_match) > 0:
    resumo = df_sem_match['motivo'].apply(
        lambda x: 'CUSTO_PRODUTO = 0' if 'CUSTO_PRODUTO' in str(x)
        else ('Fornecedor nao cadastrado' if 'nao cadastrado' in str(x)
        else ('Componente nao existe' if 'nao existe' in str(x) else 'Outro'))
    ).value_counts()
    for motivo, qtd in resumo.items():
        print(f"   {qtd:3d} | {motivo}")
else:
    print("   Nenhum componente sem match")

Helpers.save_output_dataset(
    context=context, output_name='df_sem_match', data_frame=df_sem_match)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(f"\n{'='*60}")
print(f"RESULTADOS")
print(f"{'='*60}")
print(f"Total de linhas: {df_main_all_bases.shape[0]}")
print(
    f"Componentes únicos: {df_main_all_bases['Component'].nunique() if 'Component' in df_main_all_bases.columns else 'N/A'}")
print(f"{'='*60}\n")
Helpers.save_output_dataset(
    context=context, output_name="lt_rp", data_frame=df_lt_rp)
Helpers.save_output_dataset(
    context=context, output_name="df_vendas_with_exportacao", data_frame=df_vendas_with_exportacao)
Helpers.save_output_dataset(
    context=context, output_name="main", data_frame=df_main_all_bases)