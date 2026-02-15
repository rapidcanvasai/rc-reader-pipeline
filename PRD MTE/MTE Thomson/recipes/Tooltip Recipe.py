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
import pandas as pd

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 1. Definição dos dados com as descrições
df_main_tooltip = [
    # Identificadores do Componente
    {'Tabela': 'df_main', 'Variavel': 'Cod_X', 'Tooltip': 'O código "X" do MTE (MTE X).'},
    {'Tabela': 'df_main', 'Variavel': 'Component', 'Tooltip': 'O código principal do componente (MTE).'},
    {'Tabela': 'df_main', 'Variavel': 'Description', 'Tooltip': 'A descrição textual do componente.'},
    {'Tabela': 'df_main', 'Variavel': 'Group_code', 'Tooltip': 'O código do grupo ao qual o produto pertence.'},
    {'Tabela': 'df_main', 'Variavel': 'Group_name', 'Tooltip': 'O nome do grupo do produto.'},
    {'Tabela': 'df_main', 'Variavel': 'Supplier', 'Tooltip': 'O nome do fornecedor do componente.'},
    {'Tabela': 'df_main', 'Variavel': 'Supp_Cod', 'Tooltip': 'O código do fornecedor.'},
    {'Tabela': 'df_main', 'Variavel': 'Currency', 'Tooltip': 'A moeda do custo do componente (ex: USD, BRL).'},
    
    # Métricas de Demanda e Vendas
    {'Tabela': 'df_main', 'Variavel': 'Sales_12M', 'Tooltip': 'O total de vendas (quantidade) do item nos últimos 12 meses.'},
    {'Tabela': 'df_main', 'Variavel': 'Unfulfilled_12M', 'Tooltip': 'O total de vendas não atendidas (demanda perdida) nos últimos 12 meses.'},
    {'Tabela': 'df_main', 'Variavel': '%_Export_12M', 'Tooltip': 'O percentual das vendas dos últimos 12 meses que foi destinado à exportação.'},
    {'Tabela': 'df_main', 'Variavel': 'Venda_mensal', 'Tooltip': 'A média de vendas mensais, calculada como `Sales_12M / 12`.'},
    {'Tabela': 'df_main', 'Variavel': 'Demand_(LT+RP)', 'Tooltip': 'A previsão de demanda (consumo) para o período combinado de Lead Time e Período de Revisão.'},
    
    # Colunas dinâmicas de Vendas (Sales-M)
    {'Tabela': 'df_main', 'Variavel': 'Sales-M6', 'Tooltip': 'Vendas totais do Mês 6 (M-6) nos últimos 6 meses.'},
    {'Tabela': 'df_main', 'Variavel': 'Sales-M7', 'Tooltip': 'Vendas totais do Mês 7 (M-7) nos últimos 6 meses.'},
    {'Tabela': 'df_main', 'Variavel': 'Sales-M8', 'Tooltip': 'Vendas totais do Mês 8 (M-8) nos últimos 6 meses.'},
    {'Tabela': 'df_main', 'Variavel': 'Sales-M9', 'Tooltip': 'Vendas totais do Mês 9 (M-9) nos últimos 6 meses.'},
    {'Tabela': 'df_main', 'Variavel': 'Sales-M10', 'Tooltip': 'Vendas totais do Mês 10 (M-10) nos últimos 6 meses.'},
    {'Tabela': 'df_main', 'Variavel': 'Sales-M11', 'Tooltip': 'Vendas totais do Mês 11 (M-11) nos últimos 6 meses.'},
    
    # Colunas dinâmicas de Trânsito (Transit_YYYY-MM)
    {'Tabela': 'df_main', 'Variavel': 'Transit_2025-11', 'Tooltip': 'Quantidade total em trânsito com entrega prevista para 2025-11.'},
    {'Tabela': 'df_main', 'Variavel': 'Transit_2025-12', 'Tooltip': 'Quantidade total em trânsito com entrega prevista para 2025-12.'},
    {'Tabela': 'df_main', 'Variavel': 'Transit_2026-01', 'Tooltip': 'Quantidade total em trânsito com entrega prevista para 2026-01.'},
    {'Tabela': 'df_main', 'Variavel': 'Transit_2026-02', 'Tooltip': 'Quantidade total em trânsito com entrega prevista para 2026-02.'},
    
    # Métricas de Estoque (Stock)
    {'Tabela': 'df_main', 'Variavel': 'Stock', 'Tooltip': 'A quantidade atual em estoque físico.'},
    {'Tabela': 'df_main', 'Variavel': 'Transit', 'Tooltip': 'A quantidade total de produto que já foi comprada e está em trânsito.'},
    {'Tabela': 'df_main', 'Variavel': 'Inspection', 'Tooltip': 'A quantidade de produto que está em processo de inspeção.'},
    {'Tabela': 'df_main', 'Variavel': 'Reserved', 'Tooltip': 'A quantidade de produto que está reservada para pedidos.'},
    {'Tabela': 'df_main', 'Variavel': 'Total_Stock', 'Tooltip': 'O estoque total disponível, calculado como `Stock + Transit + Inspection`.'},
    {'Tabela': 'df_main', 'Variavel': 'Inventory_level', 'Tooltip': 'O nível de estoque na data base do cálculo.'},
    
    # Sugestões de Compra (Order Suggestions)
    {'Tabela': 'df_main', 'Variavel': 'Final_order', 'Tooltip': 'A sugestão final de compra (Híbrida v2). Este é o valor final recomendado para o pedido.'},
    {'Tabela': 'df_main', 'Variavel': 'Final_order_baseline', 'Tooltip': 'A sugestão de compra *baseline*, calculada por regra simples para atingir a cobertura mínima.'},
    {'Tabela': 'df_main', 'Variavel': 'Cost', 'Tooltip': 'O custo unitário do componente.'},
    {'Tabela': 'df_main', 'Variavel': 'Total_Cost', 'Tooltip': 'O custo total do pedido sugerido, calculado como `Final_order * Cost`.'},
    {'Tabela': 'df_main', 'Variavel': 'Obs', 'Tooltip': 'Observações. Adiciona uma marcação (ex: `[Cap V2]`) se a sugestão foi limitada por um teto.'},
    
    # Classificação e Cobertura (ABC/XYZ & Coverage)
    {'Tabela': 'df_main', 'Variavel': 'ABC', 'Tooltip': 'A classificação ABC original do produto, carregada dos dados mestres.'},
    {'Tabela': 'df_main', 'Variavel': 'New_ABC', 'Tooltip': 'A nova classificação ABC (A, B, C, D) calculada pelo script, com base na participação nas `Sales_12M`.'},
    {'Tabela': 'df_main', 'Variavel': 'check_abc', 'Tooltip': 'Uma verificação booleana (True/False) se `ABC == New_ABC`.'},
    {'Tabela': 'df_main', 'Variavel': 'Participacao', 'Tooltip': 'O percentual de participação do item no total de `Sales_12M`.'},
    {'Tabela': 'df_main', 'Variavel': 'Participacao_Acumulada', 'Tooltip': 'A soma acumulada da `Participacao` (usada para definir o `New_ABC`).'},
    {'Tabela': 'df_main', 'Variavel': 'Alcance_-_Estoque_Atual', 'Tooltip': 'Cobertura em meses do estoque físico atual (`Stock / Venda_mensal`).'},
    {'Tabela': 'df_main', 'Variavel': 'Alcance_-_Estoque_Total', 'Tooltip': 'Cobertura em meses do estoque total (`Total_Stock / Venda_mensal`).'},
    {'Tabela': 'df_main', 'Variavel': 'Alcance_-_Estoque_total_+_Novo_pedido', 'Tooltip': 'Cobertura em meses que o estoque atingirá se o `Final_order` for comprado.'},
    {'Tabela': 'df_main', 'Variavel': 'Cobertura', 'Tooltip': 'O mesmo que `Alcance_-_Estoque_Total`, mas como um número decimal (float).'},
    
    # Parâmetros de Compra
    {'Tabela': 'df_main', 'Variavel': 'LT', 'Tooltip': 'Lead Time (tempo de entrega) em dias.'},
    {'Tabela': 'df_main', 'Variavel': 'RP', 'Tooltip': 'Review Period (período de revisão de estoque) em dias.'},
    {'Tabela': 'df_main', 'Variavel': 'LT+RP', 'Tooltip': 'A soma de `LT` e `RP` em dias.'},
    {'Tabela': 'df_main', 'Variavel': 'KanBan_Min', 'Tooltip': 'O nível mínimo de estoque definido no KanBan.'},
    {'Tabela': 'df_main', 'Variavel': 'KanBan_Max', 'Tooltip': 'O nível máximo de estoque definido no KanBan.'},
    {'Tabela': 'df_main', 'Variavel': 'Safety_stock', 'Tooltip': 'O valor do estoque de segurança carregado dos dados mestres.'},
    
    # Sinalizadores e Flags (Alerts)
    {'Tabela': 'df_main', 'Variavel': 'Alert', 'Tooltip': 'Alerta de performance da previsão (Convertido para 2=🔴, 1=🟡, 0=🟢) com base no erro histórico.'},
    {'Tabela': 'df_main', 'Variavel': 'Flag', 'Tooltip': 'A regra de decisão de cobertura ("Comprar" ou "Não Comprar") baseada na `Venda_mensal` e no `Alcance_-_Estoque_Total`.'},
    {'Tabela': 'df_main', 'Variavel': 'NewProduct', 'Tooltip': 'Flag (True/False) que indica se o produto é novo.'},
    {'Tabela': 'df_main', 'Variavel': 'IsException', 'Tooltip': 'Flag (True/False) que indica se o componente está na lista de exceções (ex: produtos sem fornecedor).'},
    {'Tabela': 'df_main', 'Variavel': 'Check_Suggestion', 'Tooltip': 'Flag (True/False) que sinaliza sugestões de compra que parecem muito altas ou baixas.'},
    {'Tabela': 'df_main', 'Variavel': 'Min_Order_Value_Warning', 'Tooltip': 'Flag (True/False) que indica se o fornecedor deste item já atingiu o valor mínimo de faturamento.'},
    
    # Cálculos de Volatilidade (CV & Z-Score)
    {'Tabela': 'df_main', 'Variavel': 'M_dia', 'Tooltip': 'A média de vendas dos últimos 6 meses (baseada nas colunas `Sales-M...`).'},
    {'Tabela': 'df_main', 'Variavel': 'DP', 'Tooltip': 'O desvio padrão (volatilidade) das vendas dos últimos 6 meses.'},
    {'Tabela': 'df_main', 'Variavel': 'CV', 'Tooltip': 'O Coeficiente de Variação (`DP / M_dia`), que mede a volatilidade relativa.'},
    {'Tabela': 'df_main', 'Variavel': 'CV_Flag', 'Tooltip': 'Classificação da volatilidade ("BAIXO", "MÉDIO", "ALTO") com base no `CV`.'},
    {'Tabela': 'df_main', 'Variavel': 'Nivel_de_Servico', 'Tooltip': 'O nível de serviço estatístico desejado (ex: 0.95) com base na volatilidade (`CV_Flag`).'},
    {'Tabela': 'df_main', 'Variavel': 'Valor_cr_tico_da_normal_(Z)', 'Tooltip': 'O Z-score (valor Z) correspondente ao `Nivel_de_Servico`.'},
    {'Tabela': 'df_main', 'Variavel': 'Z_Flag', 'Tooltip': 'Descrição textual do nível de serviço (ex: "95% de chance de não faltar").'},
    {'Tabela': 'df_main', 'Variavel': 'Sigma_no_per_odo', 'Tooltip': 'O desvio padrão da demanda (risco) durante o `LT+RP`.'},
    {'Tabela': 'df_main', 'Variavel': 'Demanda_M_dia_no_Periodo', 'Tooltip': 'A demanda média esperada durante o `LT+RP`.'},
    {'Tabela': 'df_main', 'Variavel': 'Estoque_de_Seguran_a', 'Tooltip': 'O estoque de segurança *calculado* pelo script (baseado em Z, DP, LT, RP).'},
    {'Tabela': 'df_main', 'Variavel': 'ROP', 'Tooltip': 'Reorder Point (Ponto de Ressuprimento) calculado pelo script.'},
    {'Tabela': 'df_main', 'Variavel': 'Order_Up_To', 'Tooltip': 'O nível de "pedir até" (Order Up To Level) calculado.'},
    {'Tabela': 'df_main', 'Variavel': 'Service_Level', 'Tooltip': 'Descrição textual da volatilidade da demanda (ex: "DEMANDA PREVISÍVEL").'},
]

df_main_tooltip = pd.DataFrame(df_main_tooltip)

# Unindo os tooltips
lista_de_dfs = [df_main_tooltip]
df_tooltip = pd.concat(lista_de_dfs)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_tooltip

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
Helpers.save_output_dataset(context=context, output_name='df_tooltip', data_frame=df_tooltip)