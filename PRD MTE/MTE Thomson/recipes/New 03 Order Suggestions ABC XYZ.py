# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Required imports

import math
from sklearn.metrics import make_scorer
from scipy.stats import norm
from dateutil import parser
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

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_estoque_atual = Helpers.getEntityData(context, 'estoque_atual')
df_lista_excecoes_produtos_sem_fornecedores = Helpers.getEntityData(
    context, "excecoes_produtos_sem_fornecedores")
df_daily_portalvendas_components = Helpers.getEntityData(
    context, "new_daily_portalvendas_components")
df_multi_horizon_pred_component = Helpers.getEntityData(
    context, "new_multihorizon_components_abcxyz")
df_products_not_delivered_exportation = Helpers.getEntityData(
    context, "df_products_not_delivered_exportation")
df_produtos = Helpers.getEntityData(context, "df_produtos")
df_produtos_sob_demanda = Helpers.getEntityData(
    context, "produtos_sob_demanda")
df_vendas = Helpers.getEntityData(context, "df_vendas")
df_inventory_histories = Helpers.getEntityData(
    context, 'new_inventory_histories2')
df_estru_produtos = Helpers.getEntityData(context, "df_estrutura_produto")

df_compiled_components = Helpers.getEntityData(
    context, "new_compiled_components2")  # new_compiled_components_abcxyz
df_compiled_components['In_Multihorizon_Predictions'] = df_compiled_components['Cod_Produto'].isin(
    df_multi_horizon_pred_component['Component'].unique()
)

df_historico_pedidos = Helpers.getEntityData(context, "df_historico_pedidos")
df_historico_pedidos['COD_FORNE'] = df_historico_pedidos['COD_FORNE'].astype(
    str)

df_parametro_fornecedor = Helpers.getEntityData(
    context, "parametro_fornecedor")
df_parametro_fornecedor['Cod_Fornecedor'] = df_parametro_fornecedor['Cod_Fornecedor'].astype(
    str)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Inicializa lista de auditoria global
lista_dfs_sem_match = []

df_produto_fornecedor_ativo = Helpers.getEntityData(
    context, "produto_fornecedor_ativo")
df_produto_fornecedor_ativo['Cod_Fornecedor'] = df_produto_fornecedor_ativo['Cod_Fornecedor'].astype(
    str)

df_produto_fornecedor = Helpers.getEntityData(context, "df_produto_fornecedor")

# Verificar e tratar df_produto_fornecedor vazio
print(f"df_produto_fornecedor: {df_produto_fornecedor.shape[0]} linhas")
if df_produto_fornecedor.shape[0] > 0:
    # --- FILTRO 1: Custo Zero ---
    mask_custo_zero = df_produto_fornecedor['CUSTO_PRODUTO'] == 0.0
    if mask_custo_zero.sum() > 0:
        df_removed_cost = df_produto_fornecedor[mask_custo_zero].copy()
        df_audit_cost = df_removed_cost[['PRODUTO']].rename(
            columns={'PRODUTO': 'Cod_component'})
        df_audit_cost['origem'] = 'df_produto_fornecedor'
        df_audit_cost['motivo'] = 'Custo do produto igual a 0.0'
        lista_dfs_sem_match.append(df_audit_cost)

    df_produto_fornecedor = df_produto_fornecedor[~mask_custo_zero]

    # --- FILTRO 2: Merge Inner Join ---
    # Verifica quais produtos ATIVOS não têm dados em produto_fornecedor
    df_merge_check = df_produto_fornecedor_ativo.merge(
        df_produto_fornecedor,
        left_on=['Cod_Produto', 'Cod_Fornecedor'],
        right_on=['PRODUTO', 'COD_FORNE'],
        how='left',
        indicator=True
    )

    # Captura produtos ativos SEM dados de custo em produto_fornecedor
    mask_no_match = df_merge_check['_merge'] == 'left_only'
    if mask_no_match.sum() > 0:
        df_removed_merge = df_merge_check[mask_no_match].copy()
        df_audit_merge = df_removed_merge[['Cod_Produto']].rename(
            columns={'Cod_Produto': 'Cod_component'})
        df_audit_merge['origem'] = 'df_produto_fornecedor_ativo'
        df_audit_merge['motivo'] = 'Produto ativo sem dados em produto_fornecedor'
        lista_dfs_sem_match.append(df_audit_merge)

    # Aplica o merge real (Inner) conforme original
    df_produto_fornecedor = (
        df_produto_fornecedor
        .merge(
            df_produto_fornecedor_ativo,
            left_on=['PRODUTO', 'COD_FORNE'],
            right_on=['Cod_Produto', 'Cod_Fornecedor'],
            how='inner'
        )[df_produto_fornecedor.columns]
    )
    print(
        f"df_produto_fornecedor após filtros: {df_produto_fornecedor.shape[0]} linhas")
else:
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


def adjust_data_row(data_row):
    '''
    This function adjusts some of the order suggestions in the main dataframa according to clients requests and other heuristics.
    These are simple business rules that can be applied to the main dataframe.
    Ideally this solution would be replaced by having a more robust ML model or better safety stock calculations. 
    '''

    # System should not suggest orders when the current stock is greater than the last 12 months of sales
    if data_row["Total Stock"] > data_row["Sales 12M"]:
        data_row["Order sug"] = 0
        data_row["Final order"] = 0
    return data_row


def calculate_mean_lead_time(df_historico_pedidos, component_code):
    df_historico_pedidos = df_historico_pedidos[
        df_historico_pedidos["PRODUTO"] == component_code
    ]
    df_historico_pedidos["Lead_Time"] = (
        df_historico_pedidos["ENTREGA_EFET"] - df_historico_pedidos["DATA_SI"]
    ).dt.days
    avg_lt = (
        df_historico_pedidos["ENTREGA_EFET"] - df_historico_pedidos["DATA_SI"]
    ).dt.days.mean()
    sd_lt = (
        df_historico_pedidos["ENTREGA_EFET"] - df_historico_pedidos["DATA_SI"]
    ).dt.days.std()

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
    '''
    https://abcsupplychain.com/safety-stock-formula-calculation/
    '''
    Z = norm.ppf(service_level)

    # filter the consumption dataset to only the component we want
    df_filtered = df_consumption_true[
        df_consumption_true["Component"] == component_code
    ]

    # if there are no orders for that component, return 0
    if df_filtered.shape[0] == 0:
        return 0

    # filter for only before the current (usage) date
    df_filtered = df_filtered[
        df_filtered['DATA_PEDIDO'] < date
    ]

    if max_lookback_method in ["orders", "both"]:
        # filter for only a maximum of X orders, where X is the max_lookback_orders parameter
        df_filtered = df_filtered.sort_values(
            by='DATA_PEDIDO', ascending=False)
        df_filtered = df_filtered.head(max_lookback_orders)
    elif max_lookback_method in ["months", "both"]:
        # filter for only the last X months, where X is the max_lookback parameter
        df_filtered = df_filtered[
            df_filtered['DATA_PEDIDO'] >= date -
            pd.DateOffset(months=max_lookback)
        ]
    avg_lt, std_lt = calculate_mean_lead_time(
        df_historico_pedidos, component_code)

    avg_consumption = df_filtered["Consumption"].mean()
    std_consumption = df_filtered["Consumption"].std()

    if method == 3:
        safety_stock = Z * std_consumption * np.sqrt(avg_lt)
    elif method == 4:
        if avg_lt != avg_lt:  # obs x!=x is true for NaN
            safety_stock = avg_consumption * lt
        elif std_lt != std_lt:
            safety_stock = Z * avg_consumption * avg_lt
        else:
            safety_stock = Z * avg_consumption * std_lt
    elif method == 5:
        safety_stock = Z * math.sqrt(
            (avg_lt * (std_consumption) ** 2 + (avg_consumption * std_lt) ** 2)
        )
    else:
        safety_stock = 0
    safety_stock = safety_stock * safety_factor

    if pd.isna(safety_stock):
        return 0

    return safety_stock


def get_in_transit_orders(date, component, df_historico_pedidos):
    df_historico_pedidos = df_historico_pedidos[
        df_historico_pedidos["PRODUTO"] == component
    ]
    df_historico_pedidos = df_historico_pedidos[df_historico_pedidos["DATA_SI"] <= date]
    df_historico_pedidos = df_historico_pedidos[
        (df_historico_pedidos["ENTREGA_EFET"].isna())
        | (df_historico_pedidos["ENTREGA_EFET"] > date)
    ]

    return df_historico_pedidos


def get_demand_ltrp(base_date, df_forecast, lt_rp, component):
    # filter the forecast to only include the component and base date
    df_forecast = df_forecast[df_forecast["Component"] == component]
    df_forecast = df_forecast[df_forecast["base_date"] == base_date]

    df_forecast_low = df_forecast[
        df_forecast['DATA_PEDIDO']
        <= base_date + pd.Timedelta(lt_rp, unit="D") + pd.offsets.MonthEnd(0)
    ]
    forecast_span_low = (
        base_date
        + pd.Timedelta(lt_rp, unit="D")
        + pd.offsets.MonthEnd(0)
        - df_forecast_low['DATA_PEDIDO'].min()
    )
    forecast_span_low_days = forecast_span_low.days

    df_forecast_high = df_forecast[
        df_forecast['DATA_PEDIDO']
        <= base_date + pd.Timedelta(lt_rp, unit="D") + pd.offsets.MonthEnd(2)
    ]
    forecast_span_high = (
        base_date
        + pd.Timedelta(lt_rp, unit="D")
        + pd.offsets.MonthEnd(2)
        - df_forecast_high['DATA_PEDIDO'].min()
    )
    forecast_span_high_days = forecast_span_high.days

    demand_low = df_forecast_low["consumption_predicted_month"].sum()
    demand_high = df_forecast_high["consumption_predicted_month"].sum()

    demand = demand_low + (
        (demand_high - demand_low) /
        (forecast_span_high_days - forecast_span_low_days)
    ) * (lt_rp - forecast_span_low_days)

    if math.isnan(demand):
        return 0.0
    else:
        return demand


def calculate_lead_time_by_supplier(df_historico_pedidos):
    df_historico_pedidos["Calculated_Lead_Time"] = (
        df_historico_pedidos["ENTREGA_EFET"] - df_historico_pedidos["DATA_SI"]
    ).dt.days
    df_lt_supplier = df_historico_pedidos.groupby(
        "COD_FORNE")["Calculated_Lead_Time"].mean().reset_index()

    return df_lt_supplier


def create_df_lt_rp(
    available_components,
    df_produto_fornecedor,
    df_produto_fornecedor_ativo_default,
    df_parametro_fornecedor_default,
    df_historico_pedidos
):
    # get the predictions
    df_produto_fornecedor = df_produto_fornecedor[
        df_produto_fornecedor["PRODUTO"].isin(available_components)
    ]

    # get calculated (real) lead time
    df_lt_supplier = calculate_lead_time_by_supplier(df_historico_pedidos)

    df_lt_rp = df_produto_fornecedor_ativo_default.merge(
        df_parametro_fornecedor_default,
        on=["Cod_Fornecedor", "Cod_Fornecedor"],
        how="left",
    )

    df_lt_rp = df_lt_rp.merge(
        df_lt_supplier, how='left', left_on='Cod_Fornecedor', right_on='COD_FORNE')

    df_lt_rp = df_lt_rp[["Cod_Produto", "Cod_Fornecedor",
                         "Review_Period", "Lead_Time", "Calculated_Lead_Time"]]
    # rename columns
    df_lt_rp = df_lt_rp.rename(
        columns={
            "Cod_Produto": "Component",
            "Cod_Fornecedor": "Supplier",
            "Review_Period": "Review_Period",
            "Lead_Time": "Lead_Time",
        }
    )

    df_lt_rp = df_lt_rp.sort_values(by=["Lead_Time", "Supplier"])
    df_lt_rp = df_lt_rp.drop_duplicates(subset=["Component"], keep="first")
    df_lt_rp = df_lt_rp.set_index("Component")
    df_lt_rp = df_lt_rp.fillna(35)
    return df_lt_rp


def get_inventory_value(df_inventory, date, component):
    """
    Busca valor de estoque para uma data e componente específicos.
    Retorna 0 se não encontrado.
    """
    result = df_inventory[
        (df_inventory["date"] == date) &
        (df_inventory["Componente"] == component)
    ]
    return result["QTD_ESTOQUE"].values[0] if len(result) > 0 else 0

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
    audit_list=None  # <--- Novo Parâmetro
):
    # TODO: refactor this function. It is too long and has too many parameters.
    # creates the order suggestions dataframe
    df_vendas["DATA"] = pd.to_datetime(df_vendas["DATA"])

    df_vendas_12meses = df_vendas[df_vendas["DATA"]
                                  >= (base_date - pd.DateOffset(months=12))]

    df_main = pd.DataFrame()
    components_without_inventory_history = []

    # Calculate the actual date to use for inventory lookup
    current_date = base_date + pd.DateOffset(days_since_base, "D")
    max_available_date = df_inventory_histories["date"].max()

    # Use the latest available date if requested date is beyond data range
    actual_inventory_date = min(current_date, max_available_date)

    if current_date > max_available_date:
        print(
            f"Warning: Requested date {current_date.date()} is beyond available inventory data.")
        print(f"Using latest available date: {actual_inventory_date.date()}")

    print("Criando versão pivotada do histórico de estoque para melhor performance...")
    df_inventory_pivot = df_inventory_histories.pivot(
        index="date",
        columns="Componente",
        values="QTD_ESTOQUE"
    ).fillna(0)
    print(f"✓ Pivot criado: {df_inventory_pivot.shape[1]} componentes")

    total_components = len(df_compiled_components)
    print(f"\n{'='*80}")
    print(f"PROCESSANDO {total_components} COMPONENTES")
    print(
        f"Base date: {pd.to_datetime(base_date).date()} | Base month: {pd.to_datetime(base_month).date()}")
    print(f"{'='*80}\n")

    # Filter df_vendas_12_months by component
    for idx, (i, row) in enumerate(df_compiled_components.iterrows(), 1):
        component = row["Cod_Produto"]
        data_row = {}

        # --- FILTRO 3: Sem Histórico de Estoque ---
        if component not in df_inventory_pivot.columns:
            components_without_inventory_history.append(component)

            # Auditoria
            if audit_list is not None:
                audit_df = pd.DataFrame([{
                    'Cod_component': component,
                    'origem': 'create_main_dataframe',
                    'motivo': 'Sem historico de estoque (df_inventory_pivot)'
                }])
                audit_list.append(audit_df)

            if idx % 50 == 0 or idx <= 10:
                print(f"  ⚠️  {component}: SEM histórico de estoque - PULADO")
            continue

        df_vendas_12meses_component = df_vendas_12meses[df_vendas_12meses["B1_COD_PP"] == component]

        # Calculate total sales
        total_sales_repo = df_vendas_12meses_component["QTD_VENDA_NACIONAL"].sum(
        )
        total_sales_expo = df_vendas_12meses_component["QTD_VENDA_EXPORTACAO"].sum(
        )
        total_sales = total_sales_repo + total_sales_expo
        data_row["Sales 12M"] = total_sales

        # Calculate the percentage of exportation
        pctg_exportacao = total_sales_expo / total_sales if total_sales > 0 else 0
        data_row[r"% Export 12M"] = pctg_exportacao * 100

        # Estimate lost sales
        df_inventory_histories_component = df_inventory_histories[
            df_inventory_histories["Componente"] == component
        ]
        df_inventory_histories_component = df_inventory_histories_component[
            df_inventory_histories_component["date"] >= (
                base_date - pd.DateOffset(months=12))
        ]

        lost_sales_days = df_inventory_histories_component[
            df_inventory_histories_component["QTD_ESTOQUE"] == 0
        ].shape[0]

        avg_daily_sales = total_sales / 365
        lost_sales_days = lost_sales_days * avg_daily_sales
        data_row["Lost 12M"] = lost_sales_days

        # Calculate unfulfilled orders
        unfullfilled_orders_repo = df_vendas_12meses_component["NAO_ATENDIDOS_REPO"].sum(
        )
        unfullfilled_orders_expo = df_vendas_12meses_component["NAO_ATENDIDOS_EXPO"].sum(
        )
        unfullfilled_orders = unfullfilled_orders_repo + unfullfilled_orders_expo
        data_row["Unfulfilled 12M"] = unfullfilled_orders

        # Usar o df pivotado para buscar estoque
        if actual_inventory_date in df_inventory_pivot.index:
            current_stock = df_inventory_pivot.loc[actual_inventory_date, component]
        else:
            # Buscar a data mais próxima disponível
            available_dates = df_inventory_pivot.index
            if len(available_dates) > 0:
                # Pegar a data disponível mais próxima (anterior ou posterior)
                closest_date = min(available_dates, key=lambda x: abs(
                    (x - actual_inventory_date).days))
                current_stock = df_inventory_pivot.loc[closest_date, component]
                if idx % 50 == 0 or idx <= 10:
                    print(
                        f"  ⚠️  Data {actual_inventory_date.date()} não encontrada. Usando {closest_date.date()} para {component}")
            else:
                current_stock = 0
                if idx % 50 == 0 or idx <= 10:
                    print(
                        f"  ⚠️  Nenhuma data disponível no histórico para {component}")
        data_row["Stock"] = current_stock

        # Usar o df pivotado para buscar inventory level
        if actual_inventory_date in df_inventory_pivot.index:
            inventory_level = df_inventory_pivot.loc[actual_inventory_date, component]
        else:
            available_dates = df_inventory_pivot.index
            if len(available_dates) > 0:
                closest_date = min(available_dates, key=lambda x: abs(
                    (x - actual_inventory_date).days))
                inventory_level = df_inventory_pivot.loc[closest_date, component]
            else:
                inventory_level = 0

        inspection = df_estoque_atual[df_estoque_atual["CODIGO_PI"]
                                      == component]["QTD_TOT_EST_INSP"].values[0]
        reserved = df_estoque_atual[df_estoque_atual["CODIGO_PI"]
                                    == component]["QTD_RESE"].values[0]

        df_in_transit_orders = get_in_transit_orders(
            current_date, component, df_order_history)
        transit = df_in_transit_orders["QUANTIDADE"].sum()
        total_stock = inventory_level + inspection - reserved + transit

        data_row["Inventory level"] = inventory_level
        data_row["Inspection"] = inspection
        data_row["Reserved"] = reserved
        data_row["Total Stock"] = total_stock
        data_row["Transit"] = transit

        component_has_active_supplier = True if component in df_lt_rp.index else False

        if component_has_active_supplier:
            df_produto_fornecedor_ativo_component = df_produto_fornecedor_ativo[
                df_produto_fornecedor_ativo["Cod_Produto"] == component]
            active_supplier_code = df_produto_fornecedor_ativo_component["Cod_Fornecedor"].values[0]
            active_supplier_name = df_produto_fornecedor_ativo_component[
                "Nome_Fornecedor"].values[0]
            active_cod_x = df_produto_fornecedor_ativo_component["Cod_X"].values[0]

            df_produto_fornecedor_select = df_produto_fornecedor[df_produto_fornecedor["PRODUTO"]
                                                                 == component][df_produto_fornecedor["COD_FORNE"] == active_supplier_code]

            # --- FILTRO 4: Fornecedor Ativo mas sem dados no df_produto_fornecedor ---
            if df_produto_fornecedor_select.shape[0] == 0:
                if audit_list is not None:
                    audit_df = pd.DataFrame([{
                        'Cod_component': component,
                        'origem': 'create_main_dataframe',
                        'motivo': 'Fornecedor ativo encontrado, mas ausente em df_produto_fornecedor'
                    }])
                    audit_list.append(audit_df)

                if idx % 50 == 0 or idx <= 10:
                    print(
                        f"  ⚠️  {component}: Fornecedor ativo não encontrado em produto_fornecedor - PULADO")
                continue

            cost = df_produto_fornecedor_select["CUSTO_PRODUTO"].values[0]
            currency = df_produto_fornecedor_select["MOEDA"].values[0]
            lt = int(df_lt_rp.loc[component, "Lead_Time"])

            # extra lead time for chinese new year
            usage_date_month = usage_date.month
            extra_rp = 0
            if str(active_supplier_code) != '2102':
                if usage_date_month == 10:
                    extra_rp = 55
                elif usage_date_month == 11:
                    extra_rp = 30

            rp = int(df_lt_rp.loc[component, "Review_Period"]) + extra_rp
            ltrp = lt + rp

            if row["In_Multihorizon_Predictions"]:
                if component not in df_sob_demanda["Cod_Produto"].values:
                    df_produtos_component = df_produtos[df_produtos["B1_COD"] == component]
                    if df_produtos_component.shape[0] == 0:
                        # Nota: Aqui o original fazia 'continue'. Vamos auditar também.
                        if audit_list is not None:
                            audit_df = pd.DataFrame([{
                                'Cod_component': component,
                                'origem': 'create_main_dataframe',
                                'motivo': 'Produto nao encontrado em df_produtos'
                            }])
                            audit_list.append(audit_df)

                        if idx % 50 == 0 or idx <= 10:
                            print(
                                f"  ⚠️  {component}: Produto não encontrado em df_produtos - PULADO")
                        continue

                    component_group = df_produtos_component["B1_GRUPO"].values[0]
                    if component_group in ["TC28"]:
                        mlb = 18
                    else:
                        mlb = 999
                    safety_stock_value = calculate_safety_stock(
                        base_date,
                        component,
                        df_daily_sales,
                        df_order_history,
                        lt,
                        method=4,
                        max_lookback=mlb,
                    )
                    # Garantir que é um número válido antes de converter para int
                    safety_stock = int(safety_stock_value) if pd.notna(
                        safety_stock_value) else 0
                else:
                    safety_stock = 0

                demand_ltrp_value = get_demand_ltrp(
                    base_month,
                    df_forecasts,
                    ltrp,
                    component,
                )
                demand_ltrp = int(demand_ltrp_value) if pd.notna(
                    demand_ltrp_value) else 0
            else:
                safety_stock = 0
                demand_ltrp = 0

            order_sug = int(safety_stock - total_stock + demand_ltrp)
            order_sug = max(order_sug, 0)

            df_produtos_component = df_produtos[df_produtos["B1_COD"] == component]
            if df_produtos_component.shape[0] == 0:
                if audit_list is not None:
                    audit_df = pd.DataFrame([{
                        'Cod_component': component,
                        'origem': 'create_main_dataframe',
                        'motivo': 'Produto nao encontrado em df_produtos (para multiplicador)'
                    }])
                    audit_list.append(audit_df)

                if idx % 50 == 0 or idx <= 10:
                    print(
                        f"  ⚠️  {component}: Produto não encontrado para pegar multiplicador - PULADO")
                continue

            multiplier = int(df_produtos_component["MULTIPLO_COMPRA"])
            component_group = df_produtos_component["B1_GRUPO"].values[0]
            if component_group in ["TC38"]:
                multiplier = 12

            order_sug = int(np.ceil(order_sug / multiplier) * multiplier)
        else:
            # Componente sem fornecedor ativo
            active_supplier_code = None
            active_supplier_name = None
            active_cod_x = None
            lt = 100
            rp = 35
            ltrp = lt+rp
            safety_stock = None
            if row["In_Multihorizon_Predictions"]:
                demand_ltrp = int(
                    get_demand_ltrp(
                        base_month,
                        df_forecasts,
                        ltrp,
                        component,
                    )
                )
            else:
                demand_ltrp = 0
            order_sug = None
            cost = None
            currency = None
            multiplier = None

        data_row["Supp Cod"] = active_supplier_code
        data_row["Supplier"] = active_supplier_name
        data_row["Cod X"] = active_cod_x
        data_row["LT"] = lt
        data_row["RP"] = rp
        data_row["LT+RP"] = ltrp
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

    print(f"\n{'='*80}")
    print(f"✓ PROCESSAMENTO CONCLUÍDO")
    print(f"{'='*80}")
    print(f"Total processados: {len(df_main)}")
    print(
        f"Sem histórico de estoque: {len(components_without_inventory_history)}")
    print(
        f"Taxa de sucesso: {100 * len(df_main) / total_components:.1f}% if total_components > 0 else 0%")
    print(f"{'='*80}\n")

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

    df_produtos = df_produtos[list(dict_produtos_cols.keys())]
    df_produtos = df_produtos.set_index("B1_COD")
    df_produtos = df_produtos.rename(columns=dict_produtos_cols)

    df_main = df_produtos.merge(
        df_main, left_index=True, right_index=True, how="right")
    df_main["Total Cost"] = df_main["Final order"] * df_main["Cost"]

    df_main = df_main.reindex(
        columns=[
            "Description",
            "Group code",
            "Group name",
            "Supp Cod",
            "Supplier",
            "Cod X",
            "ABC",
            "Stock",
            "Transit",
            "Inspection",
            "Reserved",
            "Total Stock",
            "Sales 12M",
            "Lost 12M",
            "Unfulfilled 12M",
            "KanBan Min",
            "KanBan Max",
            "Order sug",
            "Final order",
            "Total Stock/Sales 12M",
            "Order sug/Sales 12M",
            r"% Export 12M",
            "Safety stock",
            "RP",
            "LT",
            "LT+RP",
            "Inventory level",
            "Demand (LT+RP)",
            "On demand",
            "Origin",
            "Cost",
            "Currency",
            "Multiplier",
            "Total Cost",
        ]
    )

    return df_main, components_without_inventory_history


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Preparar df_inventory_histories - garantir que date é datetime
df_inventory_histories["date"] = pd.to_datetime(df_inventory_histories["date"])
df_inventory_histories = df_inventory_histories.sort_values(
    ["date", "Componente"])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_code_supplier = df_produto_fornecedor[[
    "COD_FORNE", "FORNEC_NOM"]].drop_duplicates()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_produto_fornecedor_ativo["Nome_Fornecedor"] = df_produto_fornecedor_ativo["Cod_Fornecedor"].map(
    df_code_supplier.set_index("COD_FORNE")["FORNEC_NOM"]
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_components_not_delivered_exportation = df_products_not_delivered_exportation.merge(
    df_estru_produtos, how='left', right_on='COD_PRODUTO', left_on='PRODUTO')
df_components_not_delivered_exportation["NAO_ATENDIDOS_EXPO"] = df_components_not_delivered_exportation["G1_QUANT"] * \
    df_components_not_delivered_exportation["QUANT_BO"]

df_components_not_delivered_exportation_grouped = df_components_not_delivered_exportation.groupby(
    ['COD_COMPONENTE', 'DT_ENTREGA'])['NAO_ATENDIDOS_EXPO'].sum().reset_index()

df_components_not_delivered_exportation_grouped["QTD_VENDA_NACIONAL"] = 0
df_components_not_delivered_exportation_grouped["RECEITA_VENDA_NACIONAL"] = 0
df_components_not_delivered_exportation_grouped["QTD_VENDA_EXPORTACAO"] = 0
df_components_not_delivered_exportation_grouped["RECEITA_VENDA_EXPORTACAO"] = 0
df_components_not_delivered_exportation_grouped["NAO_ATENDIDOS_REPO"] = 0

# Change to last position so we can concatenate to 'df_vendas' later
columns = df_components_not_delivered_exportation_grouped.columns.tolist()
columns.remove("NAO_ATENDIDOS_EXPO")
columns.append("NAO_ATENDIDOS_EXPO")
df_components_not_delivered_exportation_grouped = df_components_not_delivered_exportation_grouped.reindex(
    columns, axis=1)

# Rename to be equal to 'df_vendas'
df_components_not_delivered_exportation_grouped.rename(
    columns={"COD_COMPONENTE": "B1_COD_PP", "DT_ENTREGA": "DATA"}, inplace=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_vendas_with_exportacao = pd.concat(
    [df_vendas, df_components_not_delivered_exportation_grouped], ignore_index=True)
df_vendas_with_exportacao = df_vendas_with_exportacao.groupby(['B1_COD_PP', 'DATA'])[["QTD_VENDA_NACIONAL",
                                                                                      "RECEITA_VENDA_NACIONAL",
                                                                                      "QTD_VENDA_EXPORTACAO",
                                                                                      "RECEITA_VENDA_EXPORTACAO",
                                                                                      "NAO_ATENDIDOS_REPO",
                                                                                      "NAO_ATENDIDOS_EXPO"]].sum()
df_vendas_with_exportacao = df_vendas_with_exportacao.reset_index()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_vendas = df_vendas_with_exportacao.copy()

# convert the date column to datetime
df_historico_pedidos["DATA_SI"] = pd.to_datetime(
    df_historico_pedidos["DATA_SI"])
df_historico_pedidos["ENTREGA_EFET"] = pd.to_datetime(
    df_historico_pedidos["ENTREGA_EFET"])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_lt_rp = create_df_lt_rp(df_multi_horizon_pred_component["Component"].unique(),
                           df_produto_fornecedor,
                           df_produto_fornecedor_ativo,
                           df_parametro_fornecedor,
                           df_historico_pedidos)

base_months = list(df_multi_horizon_pred_component["base_date"].unique())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_main_all_bases = pd.DataFrame()
# get current date
usage_date = pd.Timestamp.now(tz="America/Sao_Paulo").normalize()

for base_month in base_months:
    days_since_base = 0
    base_date = base_month + pd.offsets.MonthEnd(0)

    # Passando lista_dfs_sem_match
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
        audit_list=lista_dfs_sem_match  # <--- Injeção da lista
    )
    df_main["base_month"] = base_month
    df_main["base_date"] = base_date

    df_main_all_bases = df_main_all_bases.append(df_main)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
last_base_month = df_main_all_bases["base_month"].max()
last_available_date = df_inventory_histories["date"].max()

# Passando lista_dfs_sem_match na última execução também
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
    days_since_base,
    audit_list=lista_dfs_sem_match
)

df_main_last["base_month"] = last_base_month
df_main_last["base_date"] = last_available_date

df_main_all_bases = df_main_all_bases.append(df_main_last)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Consolidação Final do DataFrame de Auditoria
if len(lista_dfs_sem_match) > 0:
    df_sem_match = pd.concat(lista_dfs_sem_match, ignore_index=True)
else:
    df_sem_match = pd.DataFrame(columns=['Cod_component', 'origem', 'motivo'])

# Garantir a ordem das colunas e remover duplicatas
df_sem_match = df_sem_match[['Cod_component',
                             'origem', 'motivo']].drop_duplicates()
Helpers.save_output_dataset(
    context=context, output_name='df_sem_match_atual_5', data_frame=df_sem_match)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_main_last.sort_values(by=["Total Stock", "base_date"], ascending=False)

df_main_all_bases = df_main_all_bases.reset_index()
df_main_all_bases = df_main_all_bases.rename(columns={"index": "Component"})
Helpers.save_output_dataset(
    context=context, output_name='new_main_abcxyz', data_frame=df_main_all_bases)