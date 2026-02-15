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
# ==================================================================================
# ANÁLISE VISUAL: COMPARAÇÃO OBSERVADO VS PREDITO (MODELO) VS BASELINE
# COM CLASSIFICAÇÃO ABC-XYZ
# ==================================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
from sklearn.metrics import mean_absolute_error, r2_score

print("="*80)
print("📊 ANÁLISE VISUAL: OBSERVADO VS PREDITO (MODELO) VS BASELINE + ABC-XYZ")
print("="*80)

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# CARREGAR DATAFRAMES SALVOS
COLOR_XGBOOST = '#06A77D'  # Verde para XGBoost
COLOR_BASELINE = '#E63946'  # Vermelho para Baseline
COLOR_OBSERVADO = '#000000' # Preto para Observado 

print("\n📥 Carregando dataframes...")

df_products_ma = Helpers.getEntityData(context, 'predictions_products_train_test')
df_components_ma = Helpers.getEntityData(context, 'predictions_components_train_test')

print(f"✅ Produtos: {df_products_ma.shape}")
print(f"✅ Componentes: {df_components_ma.shape}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# FUNÇÕES AUXILIARES
# ==================================================================================

def wmape(y_true, y_pred):
    """
    Calcula o Weighted Mean Absolute Percentage Error (WMAE).
    Trata casos de divisão por zero.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    sum_abs_true = np.sum(np.abs(y_true))
    if sum_abs_true == 0:
        return 0.0 if np.sum(np.abs(y_pred)) == 0 else 1.0
    return np.sum(np.abs(y_true - y_pred)) / sum_abs_true

def calcular_metricas_comparacao(df, y_true_col, y_model_col, y_baseline_col):
    """Calcula métricas para modelo e baseline"""
    
    df_clean = df
    
    if len(df_clean) == 0:
        return None
    
    y_true = df_clean[y_true_col].values
    y_pred = df_clean[y_model_col].values
    y_base = df_clean[y_baseline_col].values
    
    metricas = {
        # Modelo XGBoost
        'modelo_mae': mean_absolute_error(y_true, y_pred),
        'modelo_wmape': wmape(y_true, y_pred),
        'modelo_r2': r2_score(y_true, y_pred),
        'modelo_bias': np.sum(y_true - y_pred),
        'modelo_bias_pct': (np.sum(y_true - y_pred) / np.sum(y_true) * 100) if np.sum(y_true) != 0 else 0,
        
        # Baseline (MM12)
        'ma_mae': mean_absolute_error(y_true, y_base),
        'ma_wmape': wmape(y_true, y_base),
        'ma_r2': r2_score(y_true, y_base),
        'ma_bias': np.sum(y_true - y_base),
        'ma_bias_pct': (np.sum(y_true - y_base) / np.sum(y_true) * 100) if np.sum(y_true) != 0 else 0,
    }
    
    # Calcular melhoria
    metricas['melhoria_mae'] = ((metricas['ma_mae'] - metricas['modelo_mae']) / metricas['ma_mae'] * 100) if metricas['ma_mae'] != 0 else 0
    metricas['melhoria_wmape'] = ((metricas['ma_wmape'] - metricas['modelo_wmape']) / metricas['ma_wmape'] * 100) if metricas['ma_wmape'] != 0 else 0
    
    return metricas

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# FUNÇÃO DE CLASSIFICAÇÃO ABC-XYZ
# ==================================================================================
def classify_abc_xyz(df, id_col, value_col='observado', abc_thresholds=(80, 95)):
    """
    Classifica itens (produtos ou componentes) em ABC e XYZ.
    
    Classificação ABC (baseada no valor total):
    - A: itens que representam até 80% do valor acumulado
    - B: itens que representam de 80% a 95% do valor acumulado
    - C: itens que representam acima de 95% do valor acumulado
    
    Classificação XYZ (baseada na variabilidade da demanda):
    - X: CV <= 0.5 (demanda estável)
    - Y: 0.5 < CV <= 1.0 (demanda moderadamente variável)
    - Z: CV > 1.0 (demanda altamente variável)
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame com dados históricos
    id_col : str
        Nome da coluna de identificação ('produto' ou 'componente')
    value_col : str
        Coluna com valores para análise (default: 'observado')
    abc_thresholds : tuple
        Limites percentuais para classificação ABC (default: (80, 95))
    
    Returns:
    --------
    df : DataFrame
        DataFrame original com colunas 'classe_abc' e 'classe_xyz' atualizadas
    """
    
    print(f"\n{'='*80}")
    print(f"📊 CLASSIFICAÇÃO ABC-XYZ PARA {id_col.upper()}")
    print(f"{'='*80}")
    
    # ========================================================================
    # 1. CLASSIFICAÇÃO ABC (Baseada no Valor Total)
    # ========================================================================
    print("\n1️⃣ Calculando classificação ABC (Pareto)...")
    
    # Agregar valor total por item
    df_agg = df.groupby(id_col).agg({
        value_col: 'sum',
        'VALOR_UNITARIO': 'mean'  # Média do valor unitário
    }).reset_index()
    
    # Calcular valor monetário total
    df_agg['valor_total'] = df_agg[value_col] * df_agg['VALOR_UNITARIO']
    
    # Ordenar por valor decrescente
    df_agg = df_agg.sort_values('valor_total', ascending=False).reset_index(drop=True)
    
    # Calcular Pareto
    df_agg['cumsum'] = df_agg['valor_total'].cumsum()
    total_value = df_agg['valor_total'].sum()
    df_agg['cumsum_pct'] = (df_agg['cumsum'] / total_value) * 100
    
    # Classificar ABC
    def get_abc_class(cumsum_pct):
        if cumsum_pct <= abc_thresholds[0]:
            return 'A'
        elif cumsum_pct <= abc_thresholds[1]:
            return 'B'
        else:
            return 'C'
    
    df_agg['classe_abc'] = df_agg['cumsum_pct'].apply(get_abc_class)
    
    # Estatísticas ABC
    abc_stats = df_agg.groupby('classe_abc').agg({
        id_col: 'count',
        'valor_total': 'sum'
    })
    abc_stats['pct_itens'] = (abc_stats[id_col] / len(df_agg)) * 100
    abc_stats['pct_valor'] = (abc_stats['valor_total'] / total_value) * 100
    
    print("\n   📈 Distribuição ABC:")
    for classe in ['A', 'B', 'C']:
        if classe in abc_stats.index:
            n_itens = abc_stats.loc[classe, id_col]
            pct_itens = abc_stats.loc[classe, 'pct_itens']
            pct_valor = abc_stats.loc[classe, 'pct_valor']
            print(f"      Classe {classe}: {n_itens:,} itens ({pct_itens:.1f}%) = {pct_valor:.1f}% do valor")
    
    # ========================================================================
    # 2. CLASSIFICAÇÃO XYZ (Baseada na Variabilidade)
    # ========================================================================
    print("\n2️⃣ Calculando classificação XYZ (Coeficiente de Variação)...")
    
    # Calcular média e desvio padrão por item
    df_variability = df.groupby(id_col)[value_col].agg(['mean', 'std']).reset_index()
    
    # Calcular Coeficiente de Variação (CV)
    df_variability['cv'] = df_variability['std'] / df_variability['mean']
    
    # Tratar casos especiais
    df_variability['cv'] = df_variability['cv'].replace([np.inf, -np.inf], np.nan)
    df_variability['cv'] = df_variability['cv'].fillna(0)  # Itens sem venda = 0
    
    # Classificar XYZ
    def get_xyz_class(cv):
        if cv <= 0.5:
            return 'X'  # Estável
        elif cv <= 1.0:
            return 'Y'  # Moderado
        else:
            return 'Z'  # Altamente variável
    
    df_variability['classe_xyz'] = df_variability['cv'].apply(get_xyz_class)
    
    # Estatísticas XYZ
    xyz_stats = df_variability.groupby('classe_xyz').agg({
        id_col: 'count',
        'cv': 'mean'
    })
    xyz_stats['pct_itens'] = (xyz_stats[id_col] / len(df_variability)) * 100
    
    print("\n   📊 Distribuição XYZ:")
    for classe in ['X', 'Y', 'Z']:
        if classe in xyz_stats.index:
            n_itens = xyz_stats.loc[classe, id_col]
            pct_itens = xyz_stats.loc[classe, 'pct_itens']
            cv_medio = xyz_stats.loc[classe, 'cv']
            print(f"      Classe {classe}: {n_itens:,} itens ({pct_itens:.1f}%) | CV médio: {cv_medio:.2f}")
    
    # ========================================================================
    # 3. CRIAR MAPAS DE CLASSIFICAÇÃO
    # ========================================================================
    abc_map = df_agg.set_index(id_col)['classe_abc'].to_dict()
    xyz_map = df_variability.set_index(id_col)['classe_xyz'].to_dict()
    
    # ========================================================================
    # 4. APLICAR CLASSIFICAÇÕES AO DATAFRAME ORIGINAL
    # ========================================================================
    print("\n3️⃣ Aplicando classificações ao DataFrame...")
    
    df['classe_abc'] = df[id_col].map(abc_map).fillna('C')
    df['classe_xyz'] = df[id_col].map(xyz_map).fillna('Z')
    
    # ========================================================================
    # 5. MATRIZ ABC-XYZ
    # ========================================================================
    print("\n4️⃣ Matriz ABC-XYZ:")
    df_unique = df.groupby(id_col).agg({
        'classe_abc': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'C',
        'classe_xyz': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Z'
    }).reset_index()
    
    matriz = pd.crosstab(
        df_unique['classe_abc'], 
        df_unique['classe_xyz'], 
        margins=True, 
        margins_name='Total'
    )
    print(f"\n{matriz}")
    
    print(f"\n{'='*80}")
    print(f"✅ Classificação concluída!")
    print(f"{'='*80}\n")
    
    return df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# MÉTRICAS GERAIS E POR CLASSIFICAÇÃO ABC-XYZ
# ==================================================================================
print("\n📊 Calculando métricas gerais e por classificação ABC-XYZ...")

# MÉTRICAS GERAIS - COMPONENTES
print("\n🔹 COMPONENTES (GERAL):")
metricas_comp_geral = calcular_metricas_comparacao(df_components_ma, 'observado', 'predito_model', 'predito_baseline')

print(f"\n   MODELO XGBOOST:")
print(f"       MAE:   {metricas_comp_geral['modelo_mae']:,.2f}")
print(f"       WMAPE: {metricas_comp_geral['modelo_wmape']:.2%}")
print(f"       R²:    {metricas_comp_geral['modelo_r2']:.4f}")

print(f"\n   BASELINE (MM12):")
print(f"       MAE:   {metricas_comp_geral['ma_mae']:,.2f}")
print(f"       WMAPE: {metricas_comp_geral['ma_wmape']:.2%}")
print(f"       R²:    {metricas_comp_geral['ma_r2']:.4f}")

print(f"\n   MELHORIA DO MODELO:")
print(f"       MAE:   {metricas_comp_geral['melhoria_mae']:+.1f}%")
print(f"       WMAPE: {metricas_comp_geral['melhoria_wmape']:+.1f}%")

# MÉTRICAS POR CLASSE ABC
print("\n🔹 COMPONENTES POR CLASSE ABC:")
metricas_abc = {}
for classe_abc in ['A', 'B', 'C']:
    df_classe = df_components_ma[df_components_ma['classe_abc'] == classe_abc]
    if len(df_classe) > 0:
        metricas = calcular_metricas_comparacao(df_classe, 'observado', 'predito_model', 'predito_baseline')
        metricas_abc[classe_abc] = metricas
        
        print(f"\n   CLASSE {classe_abc}:")
        print(f"       Modelo - MAE: {metricas['modelo_mae']:,.2f} | WMAPE: {metricas['modelo_wmape']:.2%}")
        print(f"       Baseline - MAE: {metricas['ma_mae']:,.2f} | WMAPE: {metricas['ma_wmape']:.2%}")
        print(f"       Melhoria - MAE: {metricas['melhoria_mae']:+.1f}% | WMAPE: {metricas['melhoria_wmape']:+.1f}%")

# MÉTRICAS POR CLASSE XYZ
print("\n🔹 COMPONENTES POR CLASSE XYZ:")
metricas_xyz = {}
for classe_xyz in ['X', 'Y', 'Z']:
    df_classe = df_components_ma[df_components_ma['classe_xyz'] == classe_xyz]
    if len(df_classe) > 0:
        metricas = calcular_metricas_comparacao(df_classe, 'observado', 'predito_model', 'predito_baseline')
        metricas_xyz[classe_xyz] = metricas
        
        print(f"\n   CLASSE {classe_xyz}:")
        print(f"       Modelo - MAE: {metricas['modelo_mae']:,.2f} | WMAPE: {metricas['modelo_wmape']:.2%}")
        print(f"       Baseline - MAE: {metricas['ma_mae']:,.2f} | WMAPE: {metricas['ma_wmape']:.2%}")
        print(f"       Melhoria - MAE: {metricas['melhoria_mae']:+.1f}% | WMAPE: {metricas['melhoria_wmape']:+.1f}%")

# MÉTRICAS POR COMBINAÇÃO ABC-XYZ (Apenas principais)
print("\n🔹 COMPONENTES POR COMBINAÇÃO ABC-XYZ (Top 6):")
metricas_abc_xyz = {}
combinacoes_principais = ['AX', 'AY', 'AZ', 'BX', 'BY', 'CX']

for comb in combinacoes_principais:
    classe_abc = comb[0]
    classe_xyz = comb[1]
    df_classe = df_components_ma[
        (df_components_ma['classe_abc'] == classe_abc) & 
        (df_components_ma['classe_xyz'] == classe_xyz)
    ]
    if len(df_classe) > 0:
        metricas = calcular_metricas_comparacao(df_classe, 'observado', 'predito_model', 'predito_baseline')
        metricas_abc_xyz[comb] = metricas
        
        print(f"\n   CLASSE {comb}:")
        print(f"       Modelo - MAE: {metricas['modelo_mae']:,.2f} | WMAPE: {metricas['modelo_wmape']:.2%}")
        print(f"       Baseline - MAE: {metricas['ma_mae']:,.2f} | WMAPE: {metricas['ma_wmape']:.2%}")
        print(f"       Melhoria - MAE: {metricas['melhoria_mae']:+.1f}% | WMAPE: {metricas['melhoria_wmape']:+.1f}%")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# GRÁFICO 1: SCATTER PLOT - 3 MÉTODOS (Componentes)
# ==================================================================================
print("\n📊 Gerando Gráfico 1: Scatter Plot - 3 Metodologias (Componentes)...")

# Layout com GridSpec: 2 linhas × 2 colunas (terceiro gráfico ocupa toda a segunda linha)
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8])  # Linha 1 maior, linha 2 menor

# Linhas de layout
ax1 = fig.add_subplot(gs[0, 0])  # XGBoost
ax2 = fig.add_subplot(gs[0, 1])  # Baseline
ax3 = fig.add_subplot(gs[1, :])  # Comparação ocupa toda a largura

df_comp_clean = df_components_ma
colors = df_comp_clean['tipo'].map({'treino': COLOR_BASELINE, 'teste': '#A23B72'})

# =======================================================
# 1.1 - MODELO XGBOOST
# =======================================================
ax = ax1
ax.scatter(df_comp_clean['observado'], df_comp_clean['predito_model'],
           alpha=0.4, s=15, c=colors, edgecolors='none')

max_val = max(df_comp_clean['observado'].max(), df_comp_clean['predito_model'].max())
ax.plot([0, max_val], [0, max_val], 'k--', lw=2, alpha=0.5)

ax.set_xlabel('Observado', fontsize=12, fontweight='bold')
ax.set_ylabel('Predito (Modelo)', fontsize=12, fontweight='bold')
ax.set_title(f'MODELO XGBOOST\nMAE: {metricas_comp_geral["modelo_mae"]:.0f} | '
             f'WMAPE: {metricas_comp_geral["modelo_wmape"]:.1%} | '
             f'R²: {metricas_comp_geral["modelo_r2"]:.3f}',
             fontsize=13, fontweight='bold')

legend_elements = [
    Patch(facecolor=COLOR_BASELINE, label='Treino'),
    Patch(facecolor='#A23B72', label='Teste'),
    plt.Line2D([0], [0], color='k', linestyle='--', label='Perfeito', alpha=0.5)
]
ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3)

# =======================================================
# 1.2 - BASELINE (MM12)
# =======================================================
ax = ax2
ax.scatter(df_comp_clean['observado'], df_comp_clean['predito_baseline'],
           alpha=0.4, s=15, c=colors, edgecolors='none')

max_val_ma = max(df_comp_clean['observado'].max(), df_comp_clean['predito_baseline'].max())
ax.plot([0, max_val_ma], [0, max_val_ma], 'k--', lw=2, alpha=0.5)

ax.set_xlabel('Observado', fontsize=12, fontweight='bold')
ax.set_ylabel('Baseline (MM12)', fontsize=12, fontweight='bold')
ax.set_title(f'BASELINE (MM12)\nMAE: {metricas_comp_geral["ma_mae"]:.0f} | '
             f'WMAPE: {metricas_comp_geral["ma_wmape"]:.1%} | '
             f'R²: {metricas_comp_geral["ma_r2"]:.3f}',
             fontsize=13, fontweight='bold')
ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3)

# =======================================================
# 1.3 - COMPARAÇÃO DE PERFORMANCE
# =======================================================
ax = ax3

# Valores
mae_xgb = metricas_comp_geral['modelo_mae']
mae_baseline = metricas_comp_geral['ma_mae']
wmape_xgb = metricas_comp_geral['modelo_wmape'] * 100
wmape_baseline = metricas_comp_geral['ma_wmape'] * 100

metodos = ['XGBoost', 'Baseline (MM12)']
x = np.arange(len(metodos))
width = 0.35

# Cria barras lado a lado
bars1 = ax.bar(x - width/2, mae_xgb, width, label='MAE', color=COLOR_BASELINE, alpha=0.9, edgecolor='black')
bars2 = ax.bar(x + width/2, wmape_xgb, width, label='WMAPE (%)', color=COLOR_XGBOOST, alpha=0.9, edgecolor='black')

# Aqui é o segredo: usamos os valores corretos para cada método
# Em vez de passar listas fixas, montamos dinamicamente
mae_valores = [mae_xgb, mae_baseline]
wmape_valores = [wmape_xgb, wmape_baseline]

# Redesenha as barras corretamente (duas por método)
ax.clear()
bars1 = ax.bar(x - width/2, mae_valores, width, label='MAE', color=COLOR_BASELINE, alpha=0.9, edgecolor='black')
bars2 = ax.bar(x + width/2, wmape_valores, width, label='WMAPE (%)', color=COLOR_XGBOOST, alpha=0.9, edgecolor='black')

# Adiciona os valores acima das barras
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + (bar1.get_height()*0.03),
            f'{mae_valores[i]:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + (bar2.get_height()*0.03),
            f'{wmape_valores[i]:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color=COLOR_XGBOOST)

# Eixos e título
ax.set_xlabel('Metodologia', fontsize=12, fontweight='bold')
ax.set_ylabel('Valor', fontsize=12, fontweight='bold')
ax.set_title('Comparação de Performance\n(MAE vs WMAPE%)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metodos)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
print("   ✅ Gráfico 1 exibido")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# GRÁFICO 2: EVOLUÇÃO TEMPORAL - 3 MÉTODOS
# ==================================================================================
print("\n📊 Gerando Gráfico 2: Evolução Temporal - 3 Metodologias...")

fig, axes = plt.subplots(3, 1, figsize=(18, 14))

# Agregar por mês
# <<< MODIFICADO: 'predito' -> 'predito_model', 'media_movel' -> 'predito_baseline' >>>
df_mensal = df_components_ma.groupby(['periodo', 'tipo']).agg({
    'observado': 'sum',
    'predito_model': 'sum',
    'predito_baseline': 'sum'
}).reset_index()

df_mensal_treino = df_mensal[df_mensal['tipo'] == 'treino']
df_mensal_teste = df_mensal[df_mensal['tipo'] == 'teste']

# 2.1 - Volume Total
ax = axes[0]

# Plotar dados completos (treino + teste) em uma linha contínua
ax.plot(df_mensal['periodo'], df_mensal['observado'], 
        marker='o', label='Observado', linewidth=2.5, markersize=7, color='#000000')
# <<< MODIFICADO: 'predito_model' >>>
ax.plot(df_mensal['periodo'], df_mensal['predito_model'], 
        marker='s', label='XGBoost', linewidth=2.5, markersize=7, 
        color=COLOR_XGBOOST, linestyle='--', alpha=0.8)
# <<< MODIFICADO: 'predito_baseline' >>>
ax.plot(df_mensal['periodo'], df_mensal['predito_baseline'], 
        marker='^', label='Baseline (MM12)', linewidth=2.5, markersize=7, 
        color=COLOR_BASELINE, linestyle=':', alpha=0.8)

# Adicionar linha de split se houver dados de teste
if len(df_mensal_teste) > 0:
    split_date = df_mensal_teste['periodo'].min()
    ax.axvline(x=split_date, color='red', linestyle=':', linewidth=2, 
               label='Split Treino/Teste', alpha=0.6)


ax.set_xlabel('Período', fontsize=12, fontweight='bold')
ax.set_ylabel('Volume Total', fontsize=12, fontweight='bold')
ax.set_title('Volume Total: Observado vs XGBoost vs Baseline (MM12)', # <<< MODIFICADO
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)

# 2.2 - Erro Absoluto (MAE) por Mês
ax = axes[1]

mae_modelo_mensal = []
mae_ma_mensal = []
periodos = []

for periodo in df_mensal['periodo'].unique():
    # <<< MODIFICADO: Não é mais necessário o dropna >>>
    df_per = df_components_ma[df_components_ma['periodo'] == periodo]
    if len(df_per) > 0:
        # <<< MODIFICADO: 'predito_model' e 'predito_baseline' >>>
        mae_modelo = mean_absolute_error(df_per['observado'], df_per['predito_model'])
        mae_ma = mean_absolute_error(df_per['observado'], df_per['predito_baseline'])
        mae_modelo_mensal.append(mae_modelo)
        mae_ma_mensal.append(mae_ma)
        periodos.append(periodo)

ax.plot(periodos, mae_modelo_mensal, marker='s', linewidth=2.5, markersize=7, 
        label='XGBoost', color=COLOR_XGBOOST)
ax.plot(periodos, mae_ma_mensal, marker='^', linewidth=2.5, markersize=7, 
        label='Baseline (MM12)', color=COLOR_BASELINE) # <<< MODIFICADO

if 'split_date' in locals():
    ax.axvline(x=split_date, color='red', linestyle=':', linewidth=2, alpha=0.6)

ax.set_xlabel('Período', fontsize=12, fontweight='bold')
ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax.set_title('MAE por Mês: XGBoost vs Baseline (MM12)\n(Menor = Melhor)', # <<< MODIFICADO
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)

# 2.3 - WMAPE por Mês
ax = axes[2]

wmape_modelo_mensal = []
wmape_ma_mensal = []

for periodo in periodos: # Reutiliza 'periodos' da célula anterior
    # <<< MODIFICADO: Não é mais necessário o dropna >>>
    df_per = df_components_ma[df_components_ma['periodo'] == periodo]
    if len(df_per) > 0:
        # <<< MODIFICADO: 'predito_model' e 'predito_baseline' >>>
        wmape_modelo = wmape(df_per['observado'].values, df_per['predito_model'].values) * 100
        wmape_ma = wmape(df_per['observado'].values, df_per['predito_baseline'].values) * 100
        wmape_modelo_mensal.append(wmape_modelo)
        wmape_ma_mensal.append(wmape_ma)

ax.plot(periodos, wmape_modelo_mensal, marker='s', linewidth=2.5, markersize=7, 
        label='XGBoost', color=COLOR_XGBOOST)
ax.plot(periodos, wmape_ma_mensal, marker='^', linewidth=2.5, markersize=7, 
        label='Baseline (MM12)', color=COLOR_BASELINE) # <<< MODIFICADO

# Linhas de referência
ax.axhline(y=10, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
ax.axhline(y=20, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
ax.axhline(y=30, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

if 'split_date' in locals():
    ax.axvline(x=split_date, color='red', linestyle=':', linewidth=2, alpha=0.6)

ax.set_xlabel('Período', fontsize=12, fontweight='bold')
ax.set_ylabel('WMAPE (%)', fontsize=12, fontweight='bold')
ax.set_title('WMAPE por Mês: XGBoost vs Baseline (MM12)\n(Menor = Melhor)', # <<< MODIFICADO
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
print("   ✅ Gráfico 2 exibido")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# GRÁFICO 3: TOP 10 COMPONENTES - COMPARAÇÃO DE MÉTODOS
# ==================================================================================
print("\n📊 Gerando Gráfico 3: Top 10 Componentes - Comparação...")

# Calcular métricas por componente
top_10_comp = df_components_ma.groupby('componente')['observado'].sum().nlargest(10).index

metricas_por_comp = []
for comp in top_10_comp:
    # <<< MODIFICADO: Não é mais necessário o dropna >>>
    df_comp = df_components_ma[df_components_ma['componente'] == comp]
    if len(df_comp) > 0:
        metricas_por_comp.append({
            'componente': comp,
            # <<< MODIFICADO: 'predito_model' e 'predito_baseline' >>>
            'modelo_wmape': wmape(df_comp['observado'].values, df_comp['predito_model'].values) * 100,
            'ma_wmape': wmape(df_comp['observado'].values, df_comp['predito_baseline'].values) * 100,
            'volume': df_comp['observado'].sum()
        })

df_metricas_comp = pd.DataFrame(metricas_por_comp)
df_metricas_comp['melhoria'] = df_metricas_comp['ma_wmape'] - df_metricas_comp['modelo_wmape']

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# 3.1 - WMAPE Comparado (Barras Horizontais Agrupadas)
ax = axes[0]
y_pos = np.arange(len(df_metricas_comp))
height = 0.35

# Criar barras horizontais lado a lado
bars1 = ax.barh(y_pos - height/2, df_metricas_comp['modelo_wmape'], height, 
                label='XGBoost', color=COLOR_XGBOOST, alpha=0.8, edgecolor='black')
bars2 = ax.barh(y_pos + height/2, df_metricas_comp['ma_wmape'], height, 
                label='Baseline (MM12)', color=COLOR_BASELINE, alpha=0.8, edgecolor='black')

# Adicionar valores nas barras
for bar in bars1:
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
            f'{width:.1f}%', va='center', ha='left', fontsize=9, fontweight='bold')

for bar in bars2:
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
            f'{width:.1f}%', va='center', ha='left', fontsize=9, fontweight='bold')

# Configurações do eixo
ax.set_yticks(y_pos)
ax.set_yticklabels(df_metricas_comp['componente'], fontsize=10)
ax.set_xlabel('WMAPE (%)', fontsize=12, fontweight='bold')
ax.set_title('WMAPE por Componente: XGBoost vs Baseline (MM12)\n(Menor = Melhor)', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, framealpha=0.9, loc='upper right')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()  # Componente com maior volume no topo

# 3.2 - Melhoria (ou piora) do XGBoost vs Baseline (MM12)
ax = axes[1]

# Cor verde se XGBoost é melhor (melhoria > 0), vermelho se Baseline é melhor (melhoria < 0)
colors_melhoria = [COLOR_XGBOOST if x > 0 else COLOR_BASELINE for x in df_metricas_comp['melhoria']]
bars = ax.barh(y_pos, df_metricas_comp['melhoria'], color=colors_melhoria, 
               alpha=0.8, edgecolor='black')

# Adicionar valores nas barras
for bar in bars:
    width = bar.get_width()
    # Posicionar texto dependendo do sinal
    if width > 0:
        ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
                f'+{width:.1f}pp', va='center', ha='left', 
                fontsize=9, fontweight='bold', color=COLOR_XGBOOST)
    else:
        ax.text(width - 0.3, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}pp', va='center', ha='right', 
                fontsize=9, fontweight='bold', color=COLOR_BASELINE)

# Linha de referência no zero
ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)

# Configurações do eixo
ax.set_yticks(y_pos)
ax.set_yticklabels(df_metricas_comp['componente'], fontsize=10)
ax.set_xlabel('Diferença WMAPE (pp)', fontsize=12, fontweight='bold')
ax.set_title('Melhoria do XGBoost vs Baseline (MM12)\n(Verde = XGBoost Melhor | Vermelho = Baseline Melhor)', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()  # Manter mesma ordem do primeiro gráfico

plt.tight_layout()
plt.show()
print("   ✅ Gráfico 3 exibido")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# GRÁFICO 4: SÉRIES TEMPORAIS - TOP 5 COMPONENTES
# ==================================================================================
print("\n📊 Gerando Gráfico 4: Séries Temporais - Top 5 Componentes...")

top_5_comp = df_components_ma.groupby('componente')['observado'].sum().nlargest(5).index

fig, axes = plt.subplots(3, 2, figsize=(18, 15))
axes = axes.flatten()

for idx, comp in enumerate(top_5_comp):
    if idx >= 5:
        break
    
    ax = axes[idx]
    df_comp = df_components_ma[df_components_ma['componente'] == comp].sort_values('periodo')
    
    # Série observada
    ax.plot(df_comp['periodo'], df_comp['observado'], 
            marker='o', label='Observado', linewidth=2.5, markersize=6, color='#000000')
    
    # Série predita pelo modelo (XGBoost)
    ax.plot(df_comp['periodo'], df_comp['predito_model'], 
            marker='s', label='XGBoost', linewidth=2, markersize=5, 
            color=COLOR_XGBOOST, linestyle='--', alpha=0.8)
    
    # Série predita pelo baseline (MM12)
    ax.plot(df_comp['periodo'], df_comp['predito_baseline'], 
            marker='^', label='Baseline (MM12)', linewidth=2, markersize=5, 
            color=COLOR_BASELINE, linestyle=':', alpha=0.8)
    
    # Calcular métricas
    if len(df_comp) > 0:
        mae_modelo = mean_absolute_error(df_comp['observado'], df_comp['predito_model'])
        mae_ma = mean_absolute_error(df_comp['observado'], df_comp['predito_baseline'])
        wmape_modelo = wmape(df_comp['observado'].values, df_comp['predito_model'].values)
        wmape_ma = wmape(df_comp['observado'].values, df_comp['predito_baseline'].values)
        
        # Título com métricas
        ax.set_title(
            f'{comp}\nXGBoost: MAE={mae_modelo:.0f}, WMAPE={wmape_modelo:.1%} | '
            f'Baseline: MAE={mae_ma:.0f}, WMAPE={wmape_ma:.1%}',
            fontsize=10, fontweight='bold'
        )
    else:
        ax.set_title(f'{comp}', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Período', fontsize=10, fontweight='bold')
    ax.set_ylabel('Consumo', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

# Remover subplot extra se houver
if len(top_5_comp) < 6:
    fig.delaxes(axes[5])

plt.tight_layout()
plt.show()
print("   ✅ Gráfico 4 exibido")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# GRÁFICO 5: HEATMAP - MELHORIA POR MÊS E COMPONENTE
# ==================================================================================
print("\n📊 Gerando Gráfico 5: Heatmap de Melhoria...")

# Calcular melhoria por componente e mês (top 15 componentes)
top_15_comp = df_components_ma.groupby('componente')['observado'].sum().nlargest(15).index

melhoria_data = []
for comp in top_15_comp:
    for periodo in df_components_ma['periodo'].unique():
        df_subset = df_components_ma[
            (df_components_ma['componente'] == comp) & 
            (df_components_ma['periodo'] == periodo)
        ]
        
        if len(df_subset) > 0:
            wmape_modelo = wmape(df_subset['observado'].values, df_subset['predito_model'].values) * 100
            wmape_ma = wmape(df_subset['observado'].values, df_subset['predito_baseline'].values) * 100
            melhoria = wmape_ma - wmape_modelo  # Positivo = XGBoost melhor
            
            melhoria_data.append({
                'componente': comp,
                'periodo': pd.Timestamp(periodo).strftime('%Y-%m'),
                'melhoria': melhoria
            })

df_melhoria = pd.DataFrame(melhoria_data)

# Criar pivot para heatmap
pivot_melhoria = df_melhoria.pivot(index='componente', columns='periodo', values='melhoria')

# Plot
fig, ax = plt.subplots(figsize=(18, 10))

sns.heatmap(pivot_melhoria, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            cbar_kws={'label': 'Melhoria (pp)'},
            linewidths=0.5, ax=ax, vmin=-20, vmax=20)

ax.set_xlabel('Período', fontsize=12, fontweight='bold')
ax.set_ylabel('Componente', fontsize=12, fontweight='bold')
ax.set_title('Melhoria do XGBoost vs Baseline (MM12) por Componente e Mês\n(Verde = XGBoost Melhor | Vermelho = Baseline Melhor)',
             fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()
print("   ✅ Gráfico 5 exibido")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("\n📊 Gerando Gráfico 6: Distribuição de Erros...")

df_comp_clean = df_components_ma.copy()
df_comp_clean['erro_ma'] = df_comp_clean['observado'] - df_comp_clean['predito_baseline']
df_comp_clean['erro_modelo'] = df_comp_clean['observado'] - df_comp_clean['predito_model']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 6.1 - Histograma: Erro do Modelo
ax = axes[0, 0]
ax.hist(df_comp_clean['erro_modelo'], bins=50, edgecolor='black', alpha=0.7, color=COLOR_XGBOOST)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Erro XGBoost (Observado - Predito)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequência', fontsize=11, fontweight='bold')
ax.set_title('Distribuição do Erro: XGBoost', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 6.2 - Histograma: Erro da Baseline
ax = axes[0, 1]
ax.hist(df_comp_clean['erro_ma'], bins=50, edgecolor='black', alpha=0.7, color=COLOR_BASELINE)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Erro Baseline (Observado - Baseline)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequência', fontsize=11, fontweight='bold')
ax.set_title('Distribuição do Erro: Baseline (MM12)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 6.3 - Box Plot Comparativo
ax = axes[1, 0]
data_boxplot = [
    df_comp_clean['erro_modelo'],
    df_comp_clean['erro_ma']
]
bp = ax.boxplot(data_boxplot, labels=['XGBoost', 'Baseline (MM12)'], patch_artist=True)

# Colorir boxes
colors_box = [COLOR_XGBOOST, COLOR_BASELINE]
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
ax.set_ylabel('Erro', fontsize=11, fontweight='bold')
ax.set_title('Comparação da Distribuição de Erros', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 6.4 - Estatísticas Resumidas
ax = axes[1, 1]
ax.axis('off')

stats_modelo = {
    'Média': df_comp_clean['erro_modelo'].mean(),
    'Desvio Padrão': df_comp_clean['erro_modelo'].std(),
    'MAE': df_comp_clean['erro_modelo'].abs().mean(),
    'Mediana': df_comp_clean['erro_modelo'].median()
}

stats_ma = {
    'Média': df_comp_clean['erro_ma'].mean(),
    'Desvio Padrão': df_comp_clean['erro_ma'].std(),
    'MAE': df_comp_clean['erro_ma'].abs().mean(),
    'Mediana': df_comp_clean['erro_ma'].median()
}

# Criar tabela
table_data = []
for key in stats_modelo.keys():
    table_data.append([key, f"{stats_modelo[key]:.2f}", f"{stats_ma[key]:.2f}"])

table = ax.table(cellText=table_data, 
                 colLabels=['Métrica', 'XGBoost', 'Baseline (MM12)'], 
                 cellLoc='center', loc='center',
                 colWidths=[0.3, 0.35, 0.35])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Colorir header
for i in range(3):
    table[(0, i)].set_facecolor('#D3D3D3')
    table[(0, i)].set_text_props(weight='bold')

# Colorir colunas
for i in range(1, len(table_data) + 1):
    table[(i, 1)].set_facecolor('#E3F2FD')
    table[(i, 2)].set_facecolor('#E8F5E9')

ax.set_title('Estatísticas dos Erros', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()
print("   ✅ Gráfico 6 exibido")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# GRÁFICO 7: ACURÁCIA POR FAIXA - COMPARAÇÃO
# ==================================================================================
print("\n📊 Gerando Gráfico 7: Acurácia por Faixa - Comparação...")

df_comp_clean = df_components_ma.copy()
df_comp_clean['erro_abs_pct_modelo'] = np.abs((df_comp_clean['observado'] - df_comp_clean['predito_model']) / np.maximum(df_comp_clean['observado'], 1)) * 100
df_comp_clean['erro_abs_pct_ma'] = np.abs((df_comp_clean['observado'] - df_comp_clean['predito_baseline']) / np.maximum(df_comp_clean['observado'], 1)) * 100

faixas = [5, 10, 20, 30, 50]
acuracia_modelo = []
acuracia_ma = []

for faixa in faixas:
    acc_modelo = (df_comp_clean['erro_abs_pct_modelo'] <= faixa).sum() / len(df_comp_clean) * 100
    acc_ma = (df_comp_clean['erro_abs_pct_ma'] <= faixa).sum() / len(df_comp_clean) * 100
    acuracia_modelo.append(acc_modelo)
    acuracia_ma.append(acc_ma)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 7.1 - Barras Comparativas
ax = axes[0]
x = np.arange(len(faixas))
width = 0.35

bars1 = ax.bar(x - width/2, acuracia_modelo, width, label='XGBoost', 
               color=COLOR_XGBOOST, alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, acuracia_ma, width, label='Baseline (MM12)',
               color=COLOR_BASELINE, alpha=0.8, edgecolor='black')

# Adicionar valores
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Faixa de Erro (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('% de Previsões Dentro da Faixa', fontsize=12, fontweight='bold')
ax.set_title('Acurácia por Faixa: XGBoost vs Baseline (MM12)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'±{f}%' for f in faixas])
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 105])

# 7.2 - Curva Acumulada
ax = axes[1]
ax.plot(faixas, acuracia_modelo, marker='s', linewidth=2.5, markersize=10, 
        label='XGBoost', color=COLOR_XGBOOST)
ax.plot(faixas, acuracia_ma, marker='^', linewidth=2.5, markersize=10, 
        label='Baseline (MM12)', color=COLOR_BASELINE)

ax.axhline(y=80, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Meta 80%')

ax.set_xlabel('Faixa de Erro (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('% de Previsões Dentro da Faixa', fontsize=12, fontweight='bold')
ax.set_title('Curva de Acurácia Acumulada', fontsize=13, fontweight='bold')
ax.set_xticks(faixas)
ax.set_xticklabels([f'±{f}%' for f in faixas])
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 105])

plt.tight_layout()
plt.show()
print("   ✅ Gráfico 7 exibido")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# RESUMO FINAL
# ==================================================================================
print("\n" + "="*80)
print("✅ ANÁLISE COMPARATIVA COMPLETA!")
print("="*80)

print("\n📊 RESUMO DE PERFORMANCE (COMPONENTES):")
print(f"\n   MODELO XGBOOST:")
print(f"       MAE:   {metricas_comp_geral['modelo_mae']:,.0f}")
print(f"       WMAPE: {metricas_comp_geral['modelo_wmape']:.2%}")
print(f"       R²:    {metricas_comp_geral['modelo_r2']:.4f}")
print(f"       Bias:  {metricas_comp_geral['modelo_bias_pct']:+.2f}%")

print(f"\n   BASELINE (MM12):")
print(f"       MAE:   {metricas_comp_geral['ma_mae']:,.0f}")
print(f"       WMAPE: {metricas_comp_geral['ma_wmape']:.2%}")
print(f"       R²:    {metricas_comp_geral['ma_r2']:.4f}")
print(f"       Bias:  {metricas_comp_geral['ma_bias_pct']:+.2f}%")

print(f"\n   ⚡ MELHORIA DO XGBOOST:")
print(f"       MAE:   {metricas_comp_geral['melhoria_mae']:+.1f}%")
print(f"       WMAPE: {metricas_comp_geral['melhoria_wmape']:+.1f}%")

if metricas_comp_geral['melhoria_mae'] > 0:
    print(f"\n   ✅ O modelo XGBoost é {metricas_comp_geral['melhoria_mae']:.1f}% melhor que o Baseline (MAE)")
else:
    print(f"\n   ⚠️   O Baseline é {-metricas_comp_geral['melhoria_mae']:.1f}% melhor que o XGBoost (MAE)")

print("\n📊 Gráficos gerados:")
print("   1. Scatter Plot - 3 Metodologias")
print("   2. Evolução Temporal - 3 Metodologias")
print("   3. Top 10 Componentes - Comparação de Métodos")
print("   4. Séries Temporais - Top 5 Componentes")
print("   5. Heatmap de Melhoria por Mês e Componente")
print("   6. Distribuição de Erros - Comparação")
print("   7. Acurácia por Faixa - Comparação")

print("\n" + "="*80)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# 9. ANÁLISE FINANCEIRA COMPARATIVA (COMPONENTES)
# ==================================================================================

print("\n" + "="*80)
print("💲 ANÁLISE FINANCEIRA COMPARATIVA (COMPONENTES)")
print("="*80)

# Verificar se a coluna de valor existe
if 'VALOR_UNITARIO' not in df_components_ma.columns:
    print("\n⚠️ AVISO: Coluna 'VALOR_UNITARIO' não encontrada em 'df_components_ma'.")
    print("   A análise financeira não pode ser executada.")
    
else:
    print("\n🔄 Calculando impacto financeiro...")
    
    # 1. Preparar Dataframe Financeiro
    df_financial = df_components_ma.copy()
    df_financial['VALOR_UNITARIO'] = df_financial['VALOR_UNITARIO'].fillna(0)

    # 2. Calcular Valores Monetários
    # Valor total de cada período (Observado e Predito)
    df_financial['valor_observado'] = df_financial['observado'] * df_financial['VALOR_UNITARIO']

    df_financial['valor_predito_xgb'] = df_financial['predito_model'] * df_financial['VALOR_UNITARIO']
    df_financial['valor_predito_mm'] = df_financial['predito_baseline'] * df_financial['VALOR_UNITARIO']
    
    # Erro Absoluto Financeiro (Quanto errou em R$)
    df_financial['erro_abs_financeiro_xgb'] = (df_financial['observado'] - df_financial['predito_model']).abs() * df_financial['VALOR_UNITARIO']
    df_financial['erro_abs_financeiro_mm'] = (df_financial['observado'] - df_financial['predito_baseline']).abs() * df_financial['VALOR_UNITARIO']
    
    # Bias Financeiro (Erro líquido em R$ - Positivo = previu a menos, Negativo = previu a mais)
    df_financial['bias_financeiro_xgb'] = (df_financial['observado'] - df_financial['predito_model']) * df_financial['VALOR_UNITARIO']
    df_financial['bias_financeiro_mm'] = (df_financial['observado'] - df_financial['predito_baseline']) * df_financial['VALOR_UNITARIO']
    
    print("   ✅ Cálculos financeiros concluídos.")

    # 3. Agregar Métricas por Tipo (Treino/Teste)
    print("   Agregando resultados...")
    df_agg_financial = df_financial.groupby('tipo').agg(
        # Volumes Totais
        valor_observado_total=('valor_observado', 'sum'),
        valor_predito_xgb_total=('valor_predito_xgb', 'sum'),
        valor_predito_mm_total=('valor_predito_mm', 'sum'),
        
        # Erros Absolutos Totais (Soma do erro em R$ de cada previsão)
        erro_abs_financeiro_xgb_total=('erro_abs_financeiro_xgb', 'sum'),
        erro_abs_financeiro_mm_total=('erro_abs_financeiro_mm', 'sum'),
        
        # Bias Líquido Total (Quanto sobrou ou faltou em R$ no total)
        bias_financeiro_xgb_total=('bias_financeiro_xgb', 'sum'),
        bias_financeiro_mm_total=('bias_financeiro_mm', 'sum')
    ).reset_index()

    # 4. Calcular Métricas Financeiras Finais (WMAPE e BIAS Ponderado)
    
    # WMAE Financeiro: Erro Absoluto Total / Valor Observado Total
    df_agg_financial['wmape_financeiro_xgb'] = df_agg_financial['erro_abs_financeiro_xgb_total'] / df_agg_financial['valor_observado_total']
    df_agg_financial['wmape_financeiro_mm'] = df_agg_financial['erro_abs_financeiro_mm_total'] / df_agg_financial['valor_observado_total']
    
    # BIAS Financeiro Ponderado: Bias Líquido Total / Valor Observado Total
    df_agg_financial['bias_financeiro_pct_xgb'] = df_agg_financial['bias_financeiro_xgb_total'] / df_agg_financial['valor_observado_total']
    df_agg_financial['bias_financeiro_pct_mm'] = df_agg_financial['bias_financeiro_mm_total'] / df_agg_financial['valor_observado_total']

    # Melhoria (quanto WMAE financeiro do XGBoost é menor que o da MM)
    df_agg_financial['melhoria_wmape_financeira'] = (
        (df_agg_financial['wmape_financeiro_mm'] - df_agg_financial['wmape_financeiro_xgb']) / 
         df_agg_financial['wmape_financeiro_mm']
    ) * 100
    
    print("   ✅ Métricas financeiras calculadas.")

    # 5. Exibir Resultados (Foco no Teste)
    print("\n" + "-"*80)
    print("📊 RESUMO FINANCEIRO - FOCO NO CONJUNTO DE TESTE")
    print("   (Comparando Erro Financeiro Total)")
    print("-" * 80)
    
    pd.options.display.float_format = '{:,.2f}'.format
    
    df_display_financial = df_agg_financial[df_agg_financial['tipo'] == 'teste'].copy()
    
    if df_display_financial.empty:
        print("   Não há dados de 'teste' para exibir análise financeira.")
    else:
        res = df_display_financial.iloc[0]
        
        print(f"   Valor Observado Total (Teste): R$ {res['valor_observado_total']:,.2f}")
        print("-" * 60)
        
        print("   MODELO XGBOOST:")
        print(f"     Valor Predito Total:       R$ {res['valor_predito_xgb_total']:,.2f}")
        print(f"     Erro Absoluto (MAE $):     R$ {res['erro_abs_financeiro_xgb_total']:,.2f}")
        print(f"     Erro Ponderado (WMAE $):   {res['wmape_financeiro_xgb']:.2%}")
        print(f"     Bias (Viés) Líquido:       R$ {res['bias_financeiro_xgb_total']:,.2f} ({res['bias_financeiro_pct_xgb']:.2%})")
        
        print(f"\n   BASELINE (MM12):")
        print(f"     Valor Predito Total:       R$ {res['valor_predito_mm_total']:,.2f}")
        print(f"     Erro Absoluto (MAE $):     R$ {res['erro_abs_financeiro_mm_total']:,.2f}")
        print(f"     Erro Ponderado (WMAE $):   {res['wmape_financeiro_mm']:.2%}")
        print(f"     Bias (Viés) Líquido:       R$ {res['bias_financeiro_mm_total']:,.2f} ({res['bias_financeiro_pct_mm']:.2%})")
        
        print("-" * 60)
        print("   ⚡ MELHORIA DO XGBOOST (em R$):")
        
        reducao_erro_abs = res['erro_abs_financeiro_mm_total'] - res['erro_abs_financeiro_xgb_total']
        
        print(f"     Redução do Erro Absoluto:   R$ {reducao_erro_abs:,.2f}")
        print(f"     Melhoria WMAE Financeiro:   {res['melhoria_wmape_financeira']:+.2f}%")
        
        if reducao_erro_abs > 0:
            print(f"\n   ✅ O Modelo XGBoost foi R$ {reducao_erro_abs:,.2f} mais preciso que o Baseline.")
        else:
            print(f"\n   ⚠️   O Baseline foi R$ {abs(reducao_erro_abs):,.2f} mais preciso que o Modelo XGBoost.")
    
    print("="*80)
    
    # Salvar a análise financeira
    Helpers.save_output_dataset(
        context=context,
        output_name="financial_analysis_components_comparative",
        data_frame=df_agg_financial
    )
    print("\n   ✅ financial_analysis_components_comparative salvo")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# APLICAR CLASSIFICAÇÃO ABC-XYZ
print("\n🔄 Aplicando classificação ABC-XYZ...")

# Para produtos
df_products_ma = classify_abc_xyz(
    df_products_ma,
    id_col='produto',
    value_col='observado'
)

# Para componentes
df_components_ma = classify_abc_xyz(
    df_components_ma,
    id_col='componente',
    value_col='observado'
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# GRÁFICO ABC-8: PERFORMANCE POR CLASSE ABC
# ==================================================================================
print("\n📊 Gerando Gráfico ABC-1: Performance por Classe ABC...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 1.1 - Comparação WMAPE por Classe ABC
ax = axes[0, 0]
classes = list(metricas_abc.keys())
wmape_modelo = [metricas_abc[c]['modelo_wmape'] * 100 for c in classes]
wmape_baseline = [metricas_abc[c]['ma_wmape'] * 100 for c in classes]

x = np.arange(len(classes))
width = 0.35

bars1 = ax.bar(x - width/2, wmape_modelo, width, label='XGBoost', 
               color=COLOR_XGBOOST, alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, wmape_baseline, width, label='Baseline (MM12)', 
               color=COLOR_BASELINE, alpha=0.8, edgecolor='black')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Classe ABC', fontsize=12, fontweight='bold')
ax.set_ylabel('WMAPE (%)', fontsize=12, fontweight='bold')
ax.set_title('WMAPE por Classe ABC\n(Menor = Melhor)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[0, 1]
melhoria = [metricas_abc[c]['melhoria_wmape'] for c in classes]

colors_melhoria = [COLOR_XGBOOST if x > 0 else COLOR_BASELINE for x in melhoria]

bars = ax.bar(classes, melhoria, color=colors_melhoria, alpha=0.8, edgecolor='black')

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2., 
        height + (0.5 if height > 0 else -0.5),
        f'{height:.1f}%', 
        ha='center', 
        va='bottom' if height > 0 else 'top', 
        fontsize=10, 
        fontweight='bold'
    )

ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax.set_xlabel('Classe ABC', fontsize=12, fontweight='bold')
ax.set_ylabel('Melhoria WMAPE (%)', fontsize=12, fontweight='bold')
ax.set_title('Melhoria do XGBoost vs Baseline por Classe ABC\n(Verde = XGBoost Melhor, Vermelho = Pior)', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 1.3 - Distribuição de Volume por Classe ABC
ax = axes[1, 0]
volume_abc = df_components_ma.groupby('classe_abc')['observado'].sum()
colors_abc = [COLOR_BASELINE, '#F4A261', COLOR_XGBOOST]

wedges, texts, autotexts = ax.pie(volume_abc, labels=volume_abc.index, autopct='%1.1f%%',
                                    colors=colors_abc, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)

ax.set_title('Distribuição de Volume por Classe ABC', fontsize=13, fontweight='bold')

# 1.4 - Scatter: Classe ABC
ax = axes[1, 1]
df_comp_clean = df_components_ma.copy()

colors_map = {'A': COLOR_BASELINE, 'B': '#F4A261', 'C': COLOR_XGBOOST}
for classe in ['A', 'B', 'C']:
    df_classe = df_comp_clean[df_comp_clean['classe_abc'] == classe]
    if len(df_classe) > 0:
        ax.scatter(df_classe['observado'], df_classe['predito_model'], 
                   alpha=0.4, s=20, c=colors_map[classe], label=f'Classe {classe}', edgecolors='none')

max_val = max(df_comp_clean['observado'].max(), df_comp_clean['predito_model'].max())
ax.plot([0, max_val], [0, max_val], 'k--', lw=2, alpha=0.5, label='Perfeito')

ax.set_xlabel('Observado', fontsize=12, fontweight='bold')
ax.set_ylabel('Predito (Modelo)', fontsize=12, fontweight='bold')
ax.set_title('Observado vs Predito por Classe ABC', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("   ✅ Gráfico ABC-1 exibido")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# GRÁFICO XYZ-1: PERFORMANCE POR CLASSE XYZ
# ==================================================================================
print("\n📊 Gerando Gráfico XYZ-1: Performance por Classe XYZ...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 2.1 - Comparação WMAPE por Classe XYZ
ax = axes[0, 0]
classes = list(metricas_xyz.keys())
wmape_modelo = [metricas_xyz[c]['modelo_wmape'] * 100 for c in classes]
wmape_baseline = [metricas_xyz[c]['ma_wmape'] * 100 for c in classes]

x = np.arange(len(classes))
width = 0.35

bars1 = ax.bar(x - width/2, wmape_modelo, width, label='XGBoost', 
               color=COLOR_XGBOOST, alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, wmape_baseline, width, label='Baseline (MM12)', 
               color=COLOR_BASELINE, alpha=0.8, edgecolor='black')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Classe XYZ', fontsize=12, fontweight='bold')
ax.set_ylabel('WMAPE (%)', fontsize=12, fontweight='bold')
ax.set_title('WMAPE por Classe XYZ\n(Menor = Melhor)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')

# 2.2 - Comparação WMAPE Lado a Lado (Valores Absolutos)
ax = axes[0, 1]

x = np.arange(len(classes))
width = 0.35

bars1 = ax.bar(x - width/2, wmape_modelo, width, label='XGBoost', 
               color=COLOR_XGBOOST, alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, wmape_baseline, width, label='Baseline (MM12)', 
               color=COLOR_BASELINE, alpha=0.8, edgecolor='black')

# Adicionar valores nas barras
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Classe XYZ', fontsize=12, fontweight='bold')
ax.set_ylabel('WMAPE (%)', fontsize=12, fontweight='bold')
ax.set_title('Comparação Direta: XGBoost vs Baseline por Classe XYZ\n(Verde = XGBoost | Vermelho = Baseline)', 
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')

# 2.3 - Distribuição de Volume por Classe XYZ
ax = axes[1, 0]
volume_xyz = df_components_ma.groupby('classe_xyz')['observado'].sum()
colors_xyz = [COLOR_BASELINE, '#F4A261', '#8338EC']

wedges, texts, autotexts = ax.pie(volume_xyz, labels=volume_xyz.index, autopct='%1.1f%%',
                                    colors=colors_xyz, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)

ax.set_title('Distribuição de Volume por Classe XYZ', fontsize=13, fontweight='bold')

# 2.4 - Scatter: Classe XYZ
ax = axes[1, 1]
colors_map_xyz = {'X': COLOR_BASELINE, 'Y': '#F4A261', 'Z': '#8338EC'}
for classe in ['X', 'Y', 'Z']:
    df_classe = df_comp_clean[df_comp_clean['classe_xyz'] == classe]
    if len(df_classe) > 0:
        ax.scatter(df_classe['observado'], df_classe['predito_model'], 
                   alpha=0.4, s=20, c=colors_map_xyz[classe], label=f'Classe {classe}', edgecolors='none')

max_val = max(df_comp_clean['observado'].max(), df_comp_clean['predito_model'].max())
ax.plot([0, max_val], [0, max_val], 'k--', lw=2, alpha=0.5, label='Perfeito')

ax.set_xlabel('Observado', fontsize=12, fontweight='bold')
ax.set_ylabel('Predito (Modelo)', fontsize=12, fontweight='bold')
ax.set_title('Observado vs Predito por Classe XYZ', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("   ✅ Gráfico XYZ-1 exibido")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# GRÁFICO ABC-XYZ-1: MATRIZ DE PERFORMANCE
# ==================================================================================
print("\n📊 Gerando Gráfico ABC-XYZ-1: Matriz de Performance...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 3.1 - Heatmap WMAPE XGBoost por ABC-XYZ
ax = axes[0, 0]
df_unique = df_components_ma.groupby(['componente']).agg({
    'classe_abc': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'C',
    'classe_xyz': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Z'
}).reset_index()

pivot_data = []
for abc in ['A', 'B', 'C']:
    for xyz in ['X', 'Y', 'Z']:
        df_subset = df_components_ma[
            (df_components_ma['classe_abc'] == abc) & 
            (df_components_ma['classe_xyz'] == xyz)
        ]
        if len(df_subset) > 0:
            wmape_val = wmape(df_subset['observado'].values, df_subset['predito_model'].values) * 100
            pivot_data.append({'ABC': abc, 'XYZ': xyz, 'WMAPE': wmape_val})

df_pivot = pd.DataFrame(pivot_data)
if len(df_pivot) > 0:
    pivot_table = df_pivot.pivot(index='ABC', columns='XYZ', values='WMAPE')
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax, 
                cbar_kws={'label': 'WMAPE (%)'}, vmin=0, vmax=50)
    ax.set_title('WMAPE XGBoost por Classe ABC-XYZ\n(Verde = Melhor)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Classe XYZ', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classe ABC', fontsize=12, fontweight='bold')

# 3.2 - Heatmap WMAPE Baseline por ABC-XYZ
ax = axes[0, 1]
pivot_data_baseline = []
for abc in ['A', 'B', 'C']:
    for xyz in ['X', 'Y', 'Z']:
        df_subset = df_components_ma[
            (df_components_ma['classe_abc'] == abc) & 
            (df_components_ma['classe_xyz'] == xyz)
        ]
        if len(df_subset) > 0:
            wmape_val = wmape(df_subset['observado'].values, df_subset['predito_baseline'].values) * 100
            pivot_data_baseline.append({'ABC': abc, 'XYZ': xyz, 'WMAPE': wmape_val})

df_pivot_baseline = pd.DataFrame(pivot_data_baseline)
if len(df_pivot_baseline) > 0:
    pivot_table_baseline = df_pivot_baseline.pivot(index='ABC', columns='XYZ', values='WMAPE')
    sns.heatmap(pivot_table_baseline, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax, 
                cbar_kws={'label': 'WMAPE (%)'}, vmin=0, vmax=50)
    ax.set_title('WMAPE Baseline (MM12) por Classe ABC-XYZ\n(Verde = Melhor)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Classe XYZ', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classe ABC', fontsize=12, fontweight='bold')

# 3.3 - Heatmap Melhoria por ABC-XYZ
ax = axes[1, 0]
pivot_data_melhoria = []
for abc in ['A', 'B', 'C']:
    for xyz in ['X', 'Y', 'Z']:
        df_subset = df_components_ma[
            (df_components_ma['classe_abc'] == abc) & 
            (df_components_ma['classe_xyz'] == xyz)
        ]
        if len(df_subset) > 0:
            wmape_modelo = wmape(df_subset['observado'].values, df_subset['predito_model'].values) * 100
            wmape_baseline = wmape(df_subset['observado'].values, df_subset['predito_baseline'].values) * 100
            melhoria = wmape_baseline - wmape_modelo
            pivot_data_melhoria.append({'ABC': abc, 'XYZ': xyz, 'Melhoria': melhoria})

df_pivot_melhoria = pd.DataFrame(pivot_data_melhoria)
if len(df_pivot_melhoria) > 0:
    pivot_table_melhoria = df_pivot_melhoria.pivot(index='ABC', columns='XYZ', values='Melhoria')
    sns.heatmap(pivot_table_melhoria, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax, 
                center=0, cbar_kws={'label': 'Melhoria (pp)'}, vmin=-10, vmax=10)
    ax.set_title('Melhoria XGBoost vs Baseline por ABC-XYZ\n(Verde = XGBoost Melhor)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Classe XYZ', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classe ABC', fontsize=12, fontweight='bold')

# 3.4 - Distribuição de Itens por ABC-XYZ
ax = axes[1, 1]
matriz = pd.crosstab(df_unique['classe_abc'], df_unique['classe_xyz'])
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', ax=ax, 
            cbar_kws={'label': 'Quantidade'})
ax.set_title('Distribuição de Componentes por ABC-XYZ', 
             fontsize=13, fontweight='bold')
ax.set_xlabel('Classe XYZ', fontsize=12, fontweight='bold')
ax.set_ylabel('Classe ABC', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
print("   ✅ Gráfico ABC-XYZ-1 exibido")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# GRÁFICO ABC-XYZ-2: ANÁLISE DETALHADA DAS PRINCIPAIS COMBINAÇÕES
# ==================================================================================
print("\n📊 Gerando Gráfico ABC-XYZ-2: Análise Detalhada...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

combinacoes = ['AX', 'AY', 'AZ', 'BX', 'BY', 'CX']

for idx, comb in enumerate(combinacoes):
    ax = axes[idx]
    classe_abc = comb[0]
    classe_xyz = comb[1]
    
    df_comb = df_components_ma[
        (df_components_ma['classe_abc'] == classe_abc) & 
        (df_components_ma['classe_xyz'] == classe_xyz)
    ]
    
    if len(df_comb) > 0:
        # Agregar por mês
        df_mensal = df_comb.groupby('periodo').agg({
            'observado': 'sum',
            'predito_model': 'sum',
            'predito_baseline': 'sum'
        }).reset_index()
        
        # Plot
        ax.plot(df_mensal['periodo'], df_mensal['observado'], 
                marker='o', label='Observado', linewidth=2, markersize=5, color='#000000')
        ax.plot(df_mensal['periodo'], df_mensal['predito_model'], 
                marker='s', label='XGBoost', linewidth=2, markersize=4, 
                color=COLOR_XGBOOST, linestyle='--', alpha=0.8)
        ax.plot(df_mensal['periodo'], df_mensal['predito_baseline'], 
                marker='^', label='Baseline (MM12)', linewidth=2, markersize=4, 
                color=COLOR_BASELINE, linestyle=':', alpha=0.8)
        
        # Métricas
        metricas = calcular_metricas_comparacao(df_comb, 'observado', 'predito_model', 'predito_baseline')
        
        ax.set_title(
            f'Classe {comb}\nXGB: WMAPE={metricas["modelo_wmape"]:.1%} | Base: WMAPE={metricas["ma_wmape"]:.1%}', 
            fontsize=10, fontweight='bold'
        )
        ax.set_xlabel('Período', fontsize=9)
        ax.set_ylabel('Volume', fontsize=9)
        ax.legend(fontsize=8, framealpha=0.9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
    else:
        ax.text(0.5, 0.5, f'Sem dados\npara {comb}', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()
print("   ✅ Gráfico ABC-XYZ-2 exibido")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# TABELA RESUMO: MÉTRICAS POR CLASSIFICAÇÃO
# ==================================================================================
print("\n" + "="*80)
print("📊 TABELA RESUMO: MÉTRICAS POR CLASSIFICAÇÃO")
print("="*80)

# Criar DataFrame resumo
resumo_data = []

# Geral
resumo_data.append({
    'Classificação': 'GERAL',
    'XGB_WMAPE': metricas_comp_geral['modelo_wmape'] * 100,
    'BASE_WMAPE': metricas_comp_geral['ma_wmape'] * 100,
    'Melhoria': metricas_comp_geral['melhoria_wmape'],
    'XGB_R2': metricas_comp_geral['modelo_r2']
})

# Por ABC
for classe in ['A', 'B', 'C']:
    if classe in metricas_abc:
        resumo_data.append({
            'Classificação': f'ABC-{classe}',
            'XGB_WMAPE': metricas_abc[classe]['modelo_wmape'] * 100,
            'BASE_WMAPE': metricas_abc[classe]['ma_wmape'] * 100,
            'Melhoria': metricas_abc[classe]['melhoria_wmape'],
            'XGB_R2': metricas_abc[classe]['modelo_r2']
        })

# Por XYZ
for classe in ['X', 'Y', 'Z']:
    if classe in metricas_xyz:
        resumo_data.append({
            'Classificação': f'XYZ-{classe}',
            'XGB_WMAPE': metricas_xyz[classe]['modelo_wmape'] * 100,
            'BASE_WMAPE': metricas_xyz[classe]['ma_wmape'] * 100,
            'Melhoria': metricas_xyz[classe]['melhoria_wmape'],
            'XGB_R2': metricas_xyz[classe]['modelo_r2']
        })

# Por ABC-XYZ
for comb in combinacoes_principais:
    if comb in metricas_abc_xyz:
        resumo_data.append({
            'Classificação': f'ABC-XYZ-{comb}',
            'XGB_WMAPE': metricas_abc_xyz[comb]['modelo_wmape'] * 100,
            'BASE_WMAPE': metricas_abc_xyz[comb]['ma_wmape'] * 100,
            'Melhoria': metricas_abc_xyz[comb]['melhoria_wmape'],
            'XGB_R2': metricas_abc_xyz[comb]['modelo_r2']
        })

df_resumo = pd.DataFrame(resumo_data)

# Formatar para exibição
print("\n")
print(df_resumo.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

# Salvar resumo
Helpers.save_output_dataset(
    context=context,
    output_name="metricas_por_classificacao",
    data_frame=df_resumo
)
print("\n✅ Tabela salva como 'metricas_por_classificacao'")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# ANÁLISE FINANCEIRA POR CLASSIFICAÇÃO ABC-XYZ
# ==================================================================================
print("\n" + "="*80)
print("💲 ANÁLISE FINANCEIRA POR CLASSIFICAÇÃO ABC-XYZ")
print("="*80)

if 'VALOR_UNITARIO' in df_components_ma.columns:
    
    # Preparar dados financeiros
    df_financial = df_components_ma.copy()
    df_financial['VALOR_UNITARIO'] = df_financial['VALOR_UNITARIO'].fillna(0)
    
    df_financial['valor_observado'] = df_financial['observado'] * df_financial['VALOR_UNITARIO']
    df_financial['valor_predito_xgb'] = df_financial['predito_model'] * df_financial['VALOR_UNITARIO']
    df_financial['valor_predito_mm'] = df_financial['predito_baseline'] * df_financial['VALOR_UNITARIO']
    
    df_financial['erro_abs_financeiro_xgb'] = (df_financial['observado'] - df_financial['predito_model']).abs() * df_financial['VALOR_UNITARIO']
    df_financial['erro_abs_financeiro_mm'] = (df_financial['observado'] - df_financial['predito_baseline']).abs() * df_financial['VALOR_UNITARIO']
    
    # Agregar por ABC
    print("\n🔹 POR CLASSE ABC:")
    for classe in ['A', 'B', 'C']:
        df_classe = df_financial[df_financial['classe_abc'] == classe]
        if len(df_classe) > 0:
            valor_obs = df_classe['valor_observado'].sum()
            erro_xgb = df_classe['erro_abs_financeiro_xgb'].sum()
            erro_mm = df_classe['erro_abs_financeiro_mm'].sum()
            economia = erro_mm - erro_xgb
            
            print(f"\n   Classe {classe}:")
            print(f"       Valor Total: R$ {valor_obs:,.2f}")
            print(f"       Erro XGB:    R$ {erro_xgb:,.2f}")
            print(f"       Erro Base:   R$ {erro_mm:,.2f}")
            print(f"       Economia:    R$ {economia:,.2f} ({(economia/erro_mm*100):.1f}%)")
    
    # Agregar por XYZ
    print("\n🔹 POR CLASSE XYZ:")
    for classe in ['X', 'Y', 'Z']:
        df_classe = df_financial[df_financial['classe_xyz'] == classe]
        if len(df_classe) > 0:
            valor_obs = df_classe['valor_observado'].sum()
            erro_xgb = df_classe['erro_abs_financeiro_xgb'].sum()
            erro_mm = df_classe['erro_abs_financeiro_mm'].sum()
            economia = erro_mm - erro_xgb
            
            print(f"\n   Classe {classe}:")
            print(f"       Valor Total: R$ {valor_obs:,.2f}")
            print(f"       Erro XGB:    R$ {erro_xgb:,.2f}")
            print(f"       Erro Base:   R$ {erro_mm:,.2f}")
            print(f"       Economia:    R$ {economia:,.2f} ({(economia/erro_mm*100):.1f}%)")
    
    print("\n" + "="*80)
else:
    print("\n⚠️ Coluna VALOR_UNITARIO não encontrada. Análise financeira não executada.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# TABELA FINANCEIRA DETALHADA POR ABC-XYZ
# ==================================================================================
print("\n" + "="*80)
print("💰 TABELA FINANCEIRA DETALHADA POR CLASSIFICAÇÃO ABC-XYZ")
print("="*80)

if 'VALOR_UNITARIO' in df_components_ma.columns:
    
    # Preparar dados financeiros
    df_financial = df_components_ma.copy()
    df_financial['VALOR_UNITARIO'] = df_financial['VALOR_UNITARIO'].fillna(0)
    
    df_financial['erro_abs_financeiro_xgb'] = (df_financial['observado'] - df_financial['predito_model']).abs() * df_financial['VALOR_UNITARIO']
    df_financial['erro_abs_financeiro_mm'] = (df_financial['observado'] - df_financial['predito_baseline']).abs() * df_financial['VALOR_UNITARIO']
    
    # Lista para armazenar resultados
    financial_results = []
    
    # 1. GERAL (Total)
    erro_xgb_total = df_financial['erro_abs_financeiro_xgb'].sum()
    erro_mm_total = df_financial['erro_abs_financeiro_mm'].sum()
    reducao_total = erro_mm_total - erro_xgb_total
    pct_reducao_total = (reducao_total / erro_mm_total * 100) if erro_mm_total > 0 else 0
    
    financial_results.append({
        'Classificação': 'GERAL',
        'Erro_XGBoost': erro_xgb_total,
        'Erro_Baseline': erro_mm_total,
        'Redução_Absoluta': reducao_total,
        'Redução_Percentual': pct_reducao_total
    })
    
    # 2. POR CLASSE ABC
    for classe in ['A', 'B', 'C']:
        df_classe = df_financial[df_financial['classe_abc'] == classe]
        if len(df_classe) > 0:
            erro_xgb = df_classe['erro_abs_financeiro_xgb'].sum()
            erro_mm = df_classe['erro_abs_financeiro_mm'].sum()
            reducao = erro_mm - erro_xgb
            pct_reducao = (reducao / erro_mm * 100) if erro_mm > 0 else 0
            
            financial_results.append({
                'Classificação': f'ABC-{classe}',
                'Erro_XGBoost': erro_xgb,
                'Erro_Baseline': erro_mm,
                'Redução_Absoluta': reducao,
                'Redução_Percentual': pct_reducao
            })
    
    # 3. POR CLASSE XYZ
    for classe in ['X', 'Y', 'Z']:
        df_classe = df_financial[df_financial['classe_xyz'] == classe]
        if len(df_classe) > 0:
            erro_xgb = df_classe['erro_abs_financeiro_xgb'].sum()
            erro_mm = df_classe['erro_abs_financeiro_mm'].sum()
            reducao = erro_mm - erro_xgb
            pct_reducao = (reducao / erro_mm * 100) if erro_mm > 0 else 0
            
            financial_results.append({
                'Classificação': f'XYZ-{classe}',
                'Erro_XGBoost': erro_xgb,
                'Erro_Baseline': erro_mm,
                'Redução_Absoluta': reducao,
                'Redução_Percentual': pct_reducao
            })
    
    # 4. POR COMBINAÇÃO ABC-XYZ
    for abc in ['A', 'B', 'C']:
        for xyz in ['X', 'Y', 'Z']:
            df_classe = df_financial[
                (df_financial['classe_abc'] == abc) & 
                (df_financial['classe_xyz'] == xyz)
            ]
            if len(df_classe) > 0:
                erro_xgb = df_classe['erro_abs_financeiro_xgb'].sum()
                erro_mm = df_classe['erro_abs_financeiro_mm'].sum()
                reducao = erro_mm - erro_xgb
                pct_reducao = (reducao / erro_mm * 100) if erro_mm > 0 else 0
                
                financial_results.append({
                    'Classificação': f'{abc}{xyz}',
                    'Erro_XGBoost': erro_xgb,
                    'Erro_Baseline': erro_mm,
                    'Redução_Absoluta': reducao,
                    'Redução_Percentual': pct_reducao
                })
    
    # Criar DataFrame
    df_financial_summary = pd.DataFrame(financial_results)
    
    # Formatar valores monetários
    df_financial_summary['Erro_XGBoost_fmt'] = df_financial_summary['Erro_XGBoost'].apply(lambda x: f'R$ {x:,.2f}')
    df_financial_summary['Erro_Baseline_fmt'] = df_financial_summary['Erro_Baseline'].apply(lambda x: f'R$ {x:,.2f}')
    df_financial_summary['Redução_Absoluta_fmt'] = df_financial_summary['Redução_Absoluta'].apply(
        lambda x: f'R$ {x:,.2f}' if x >= 0 else f'-R$ {abs(x):,.2f}'
    )
    df_financial_summary['Redução_Percentual_fmt'] = df_financial_summary['Redução_Percentual'].apply(
        lambda x: f'{x:.2f}%'
    )
    
    # Exibir tabela formatada
    print("\n📊 RESUMO FINANCEIRO - ECONOMIA DO XGBOOST vs BASELINE")
    print("="*120)
    print(f"{'Classificação':<15} | {'Erro XGBoost':>20} | {'Erro Baseline':>20} | {'Redução':>20} | {'Redução %':>12}")
    print("-"*120)
    
    for _, row in df_financial_summary.iterrows():
        print(f"{row['Classificação']:<15} | {row['Erro_XGBoost_fmt']:>20} | {row['Erro_Baseline_fmt']:>20} | {row['Redução_Absoluta_fmt']:>20} | {row['Redução_Percentual_fmt']:>12}")
    
    print("="*120)
    
    # Salvar tabela
    df_financial_export = df_financial_summary[[
        'Classificação', 'Erro_XGBoost', 'Erro_Baseline', 'Redução_Absoluta', 'Redução_Percentual'
    ]].copy()
    
    Helpers.save_output_dataset(
        context=context,
        output_name="financial_summary_abc_xyz",
        data_frame=df_financial_export
    )
    print("\n✅ Tabela salva como 'financial_summary_abc_xyz'")
    
    # ==================================================================================
    # VISUALIZAÇÃO: GRÁFICO DE ECONOMIA POR ABC-XYZ
    # ==================================================================================
    print("\n📊 Gerando visualizações da economia...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    # Gráfico 1: Comparação Geral (ABC, XYZ, GERAL)
    ax = axes[0, 0]
    df_plot1 = df_financial_summary[df_financial_summary['Classificação'].isin([
        'GERAL', 'ABC-A', 'ABC-B', 'ABC-C', 'XYZ-X', 'XYZ-Y', 'XYZ-Z'
    ])].copy()
    
    x = np.arange(len(df_plot1))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df_plot1['Erro_XGBoost'], width, 
                   label='XGBoost', color=COLOR_XGBOOST, alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, df_plot1['Erro_Baseline'], width, 
                   label='Baseline', color=COLOR_BASELINE, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Classificação', fontsize=12, fontweight='bold')
    ax.set_ylabel('Erro Absoluto (R$)', fontsize=12, fontweight='bold')
    ax.set_title('Comparação de Erro Financeiro: XGBoost vs Baseline\n(Por Classificação)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot1['Classificação'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Gráfico 2: Redução Absoluta por Classificação
    ax = axes[0, 1]
    df_plot2 = df_financial_summary[df_financial_summary['Classificação'].isin([
        'ABC-A', 'ABC-B', 'ABC-C', 'XYZ-X', 'XYZ-Y', 'XYZ-Z'
    ])].copy()
    
    colors_reducao = [COLOR_XGBOOST if x >= 0 else COLOR_BASELINE for x in df_plot2['Redução_Absoluta']]
    bars = ax.barh(df_plot2['Classificação'], df_plot2['Redução_Absoluta'], 
                   color=colors_reducao, alpha=0.8, edgecolor='black')
    
    # Adicionar valores
    for bar in bars:
        width = bar.get_width()
        label = f'R$ {width:,.0f}'
        if width >= 0:
            ax.text(width + (width*0.02), bar.get_y() + bar.get_height()/2, 
                   label, va='center', ha='left', fontsize=9, fontweight='bold')
        else:
            ax.text(width - (abs(width)*0.02), bar.get_y() + bar.get_height()/2, 
                   label, va='center', ha='right', fontsize=9, fontweight='bold')
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax.set_xlabel('Redução de Erro (R$)', fontsize=12, fontweight='bold')
    ax.set_title('Economia por Classificação ABC e XYZ\n(Verde = Economia | Vermelho = Aumento)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Gráfico 3: Matriz ABC-XYZ - Redução Absoluta
    ax = axes[1, 0]
    df_matriz = df_financial_summary[
        df_financial_summary['Classificação'].str.len() == 2
    ].copy()
    df_matriz['ABC'] = df_matriz['Classificação'].str[0]
    df_matriz['XYZ'] = df_matriz['Classificação'].str[1]
    
    pivot_reducao = df_matriz.pivot(index='ABC', columns='XYZ', values='Redução_Absoluta')
    
    sns.heatmap(pivot_reducao, annot=True, fmt='.0f', cmap='RdYlGn', center=0,
                ax=ax, cbar_kws={'label': 'Redução (R$)'},
                linewidths=0.5)
    
    ax.set_title('Economia Financeira por Combinação ABC-XYZ\n(Verde = Maior Economia)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Classe XYZ', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classe ABC', fontsize=12, fontweight='bold')
    
    # Gráfico 4: Matriz ABC-XYZ - Redução Percentual
    ax = axes[1, 1]
    pivot_reducao_pct = df_matriz.pivot(index='ABC', columns='XYZ', values='Redução_Percentual')
    
    sns.heatmap(pivot_reducao_pct, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                ax=ax, cbar_kws={'label': 'Redução (%)'},
                linewidths=0.5, vmin=-20, vmax=20)
    
    ax.set_title('Redução Percentual por Combinação ABC-XYZ\n(Verde = Maior % de Economia)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Classe XYZ', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classe ABC', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    print("   ✅ Gráficos de economia financeira exibidos")
    
    # ==================================================================================
    # INSIGHTS FINAIS
    # ==================================================================================
    print("\n" + "="*80)
    print("🎯 PRINCIPAIS INSIGHTS FINANCEIROS")
    print("="*80)
    
    # Melhor combinação ABC-XYZ
    df_combinacoes = df_financial_summary[df_financial_summary['Classificação'].str.len() == 2].copy()
    melhor_comb = df_combinacoes.loc[df_combinacoes['Redução_Absoluta'].idxmax()]
    pior_comb = df_combinacoes.loc[df_combinacoes['Redução_Absoluta'].idxmin()]
    
    print(f"\n✅ MELHOR DESEMPENHO:")
    print(f"   Classe {melhor_comb['Classificação']}: Economia de R$ {melhor_comb['Redução_Absoluta']:,.2f} ({melhor_comb['Redução_Percentual']:.1f}%)")
    
    print(f"\n⚠️  PIOR DESEMPENHO:")
    if pior_comb['Redução_Absoluta'] < 0:
        print(f"   Classe {pior_comb['Classificação']}: Aumento de R$ {abs(pior_comb['Redução_Absoluta']):,.2f} ({abs(pior_comb['Redução_Percentual']):.1f}%)")
    else:
        print(f"   Classe {pior_comb['Classificação']}: Economia de R$ {pior_comb['Redução_Absoluta']:,.2f} ({pior_comb['Redução_Percentual']:.1f}%)")
    
    print(f"\n💰 ECONOMIA TOTAL:")
    print(f"   R$ {reducao_total:,.2f} ({pct_reducao_total:.1f}% de redução)")
    
    print("\n" + "="*80)
    
else:
    print("\n⚠️ Coluna VALOR_UNITARIO não encontrada. Análise financeira não executada.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# ANÁLISE FINANCEIRA MENSAL POR ABC-XYZ (ÚLTIMO ANO)
# ==================================================================================
print("\n" + "="*80)
print("📅 ANÁLISE FINANCEIRA MENSAL POR ABC-XYZ - ÚLTIMO ANO")
print("="*80)

if 'VALOR_UNITARIO' in df_components_ma.columns:
    
    # Preparar dados financeiros
    df_financial = df_components_ma.copy()
    df_financial['VALOR_UNITARIO'] = df_financial['VALOR_UNITARIO'].fillna(0)
    
    # Converter periodo para datetime se necessário
    df_financial['periodo'] = pd.to_datetime(df_financial['periodo'])
    
    # Calcular erros financeiros
    df_financial['erro_abs_financeiro_xgb'] = (df_financial['observado'] - df_financial['predito_model']).abs() * df_financial['VALOR_UNITARIO']
    df_financial['erro_abs_financeiro_mm'] = (df_financial['observado'] - df_financial['predito_baseline']).abs() * df_financial['VALOR_UNITARIO']
    
    # ========================================================================
    # 1. FILTRAR ÚLTIMO ANO (12 MESES MAIS RECENTES)
    # ========================================================================
    ultimo_periodo = pd.Timestamp(df_financial['periodo'].max())
    data_inicio_ano = pd.Timestamp(ultimo_periodo) - pd.DateOffset(months=11)  # 12 meses incluindo o atual    
    df_ultimo_ano = df_financial[df_financial['periodo'] >= data_inicio_ano].copy()
    
    print(f"\n📆 Período analisado: {pd.Timestamp(data_inicio_ano).strftime('%Y-%m')} até {pd.Timestamp(ultimo_periodo).strftime('%Y-%m')}")
    print(f"   Total de meses: {df_ultimo_ano['periodo'].nunique()}")
    
    # ========================================================================
    # 2. CALCULAR MÉTRICAS MENSAIS - GERAL
    # ========================================================================
    print("\n🔄 Calculando métricas mensais...")
    
    financial_mensal = []
    
    # Para cada mês
    for periodo in sorted(df_ultimo_ano['periodo'].unique()):
        df_mes = df_ultimo_ano[df_ultimo_ano['periodo'] == periodo]
        
        # GERAL
        erro_xgb = df_mes['erro_abs_financeiro_xgb'].sum()
        erro_mm = df_mes['erro_abs_financeiro_mm'].sum()
        reducao = erro_mm - erro_xgb
        pct_reducao = (reducao / erro_mm * 100) if erro_mm > 0 else 0
        
        financial_mensal.append({
            'Periodo': periodo,
            'Classificacao': 'GERAL',
            'Erro_XGBoost': erro_xgb,
            'Erro_Baseline': erro_mm,
            'Reducao_Absoluta': reducao,
            'Reducao_Percentual': pct_reducao
        })
        
        # POR CLASSE ABC
        for classe_abc in ['A', 'B', 'C']:
            df_classe = df_mes[df_mes['classe_abc'] == classe_abc]
            if len(df_classe) > 0:
                erro_xgb = df_classe['erro_abs_financeiro_xgb'].sum()
                erro_mm = df_classe['erro_abs_financeiro_mm'].sum()
                reducao = erro_mm - erro_xgb
                pct_reducao = (reducao / erro_mm * 100) if erro_mm > 0 else 0
                
                financial_mensal.append({
                    'Periodo': periodo,
                    'Classificacao': f'ABC-{classe_abc}',
                    'Erro_XGBoost': erro_xgb,
                    'Erro_Baseline': erro_mm,
                    'Reducao_Absoluta': reducao,
                    'Reducao_Percentual': pct_reducao
                })
        
        # POR CLASSE XYZ
        for classe_xyz in ['X', 'Y', 'Z']:
            df_classe = df_mes[df_mes['classe_xyz'] == classe_xyz]
            if len(df_classe) > 0:
                erro_xgb = df_classe['erro_abs_financeiro_xgb'].sum()
                erro_mm = df_classe['erro_abs_financeiro_mm'].sum()
                reducao = erro_mm - erro_xgb
                pct_reducao = (reducao / erro_mm * 100) if erro_mm > 0 else 0
                
                financial_mensal.append({
                    'Periodo': periodo,
                    'Classificacao': f'XYZ-{classe_xyz}',
                    'Erro_XGBoost': erro_xgb,
                    'Erro_Baseline': erro_mm,
                    'Reducao_Absoluta': reducao,
                    'Reducao_Percentual': pct_reducao
                })
        
        # POR COMBINAÇÃO ABC-XYZ
        for abc in ['A', 'B', 'C']:
            for xyz in ['X', 'Y', 'Z']:
                df_classe = df_mes[
                    (df_mes['classe_abc'] == abc) & 
                    (df_mes['classe_xyz'] == xyz)
                ]
                if len(df_classe) > 0:
                    erro_xgb = df_classe['erro_abs_financeiro_xgb'].sum()
                    erro_mm = df_classe['erro_abs_financeiro_mm'].sum()
                    reducao = erro_mm - erro_xgb
                    pct_reducao = (reducao / erro_mm * 100) if erro_mm > 0 else 0
                    
                    financial_mensal.append({
                        'Periodo': periodo,
                        'Classificacao': f'{abc}{xyz}',
                        'Erro_XGBoost': erro_xgb,
                        'Erro_Baseline': erro_mm,
                        'Reducao_Absoluta': reducao,
                        'Reducao_Percentual': pct_reducao
                    })
    
    # Criar DataFrame
    df_financial_mensal = pd.DataFrame(financial_mensal)
    df_financial_mensal['Periodo_fmt'] = df_financial_mensal['Periodo'].dt.strftime('%Y-%m')
    
    print(f"   ✅ {len(df_financial_mensal)} registros calculados")
    
    # ========================================================================
    # 3. EXIBIR TABELA RESUMIDA (ÚLTIMOS 3 MESES)
    # ========================================================================
    print("\n📊 RESUMO FINANCEIRO - ÚLTIMOS 3 MESES")
    print("="*120)
    
    ultimos_3_meses = sorted(df_ultimo_ano['periodo'].unique())[-3:]
    df_resumo_3m = df_financial_mensal[df_financial_mensal['Periodo'].isin(ultimos_3_meses)].copy()
    
    for periodo in ultimos_3_meses:
        df_mes = df_resumo_3m[df_resumo_3m['Periodo'] == periodo]
        periodo_fmt = df_mes['Periodo_fmt'].iloc[0]
        
        print(f"\n🗓️  {periodo_fmt}")
        print("-"*120)
        print(f"{'Classificação':<15} | {'Erro XGBoost':>20} | {'Erro Baseline':>20} | {'Redução':>20} | {'Redução %':>12}")
        print("-"*120)
        
        # Mostrar apenas as principais classificações
        classes_principais = ['GERAL', 'ABC-A', 'ABC-B', 'ABC-C', 'AX', 'AY', 'BX', 'CX']
        df_mes_filtrado = df_mes[df_mes['Classificacao'].isin(classes_principais)]
        
        for _, row in df_mes_filtrado.iterrows():
            erro_xgb_fmt = f"R$ {row['Erro_XGBoost']:,.2f}"
            erro_mm_fmt = f"R$ {row['Erro_Baseline']:,.2f}"
            reducao_fmt = f"R$ {row['Reducao_Absoluta']:,.2f}" if row['Reducao_Absoluta'] >= 0 else f"-R$ {abs(row['Reducao_Absoluta']):,.2f}"
            pct_fmt = f"{row['Reducao_Percentual']:.2f}%"
            
            print(f"{row['Classificacao']:<15} | {erro_xgb_fmt:>20} | {erro_mm_fmt:>20} | {reducao_fmt:>20} | {pct_fmt:>12}")
    
    print("="*120)
    
    # ========================================================================
    # 4. SALVAR DATASET COMPLETO
    # ========================================================================
    # df_financial_export = df_financial_mensal[[
    #     'Periodo_fmt', 'Classificacao', 'Erro_XGBoost', 'Erro_Baseline', 
    #     'Reducao_Absoluta', 'Reducao_Percentual'
    # ]].copy()
    # df_financial_export.rename(columns={'Periodo_fmt': 'Periodo'}, inplace=True)
    
    # Helpers.save_output_dataset(
    #     context=context,
    #     output_name="financial_mensal_abc_xyz",
    #     data_frame=df_financial_export
    # )
    # print("\n✅ Tabela mensal completa salva como 'financial_mensal_abc_xyz'")
    
    # ========================================================================
    # 5. VISUALIZAÇÕES
    # ========================================================================
    print("\n📊 Gerando visualizações da evolução mensal...")
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # -----------------------------------------------------------------------
    # GRÁFICO 1: EVOLUÇÃO DA REDUÇÃO TOTAL (GERAL)
    # -----------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    
    df_geral = df_financial_mensal[df_financial_mensal['Classificacao'] == 'GERAL'].sort_values('Periodo')
    
    ax1_twin = ax1.twinx()
    
    # Linha: Redução Absoluta
    line1 = ax1.plot(df_geral['Periodo_fmt'], df_geral['Reducao_Absoluta'], 
                     marker='o', linewidth=3, markersize=8, color=COLOR_XGBOOST, 
                     label='Redução (R$)')
    
    # Barras: Redução Percentual
    colors_bars = [COLOR_XGBOOST if x >= 0 else COLOR_BASELINE for x in df_geral['Reducao_Percentual']]
    bars = ax1_twin.bar(df_geral['Periodo_fmt'], df_geral['Reducao_Percentual'], 
                        alpha=0.3, color=colors_bars, edgecolor='black', linewidth=1)
    
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax1_twin.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    ax1.set_xlabel('Período', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Redução Absoluta (R$)', fontsize=12, fontweight='bold', color=COLOR_XGBOOST)
    ax1_twin.set_ylabel('Redução Percentual (%)', fontsize=12, fontweight='bold', color='gray')
    ax1.set_title('Evolução Mensal da Economia: XGBoost vs Baseline\n(Última 12 Meses)', 
                  fontsize=14, fontweight='bold')
    
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    
    # -----------------------------------------------------------------------
    # GRÁFICO 2: EVOLUÇÃO POR CLASSE ABC
    # -----------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    
    for classe in ['ABC-A', 'ABC-B', 'ABC-C']:
        df_classe = df_financial_mensal[df_financial_mensal['Classificacao'] == classe].sort_values('Periodo')
        ax2.plot(df_classe['Periodo_fmt'], df_classe['Reducao_Absoluta'], 
                marker='o', linewidth=2.5, markersize=6, label=classe)
    
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('Período', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Redução (R$)', fontsize=11, fontweight='bold')
    ax2.set_title('Evolução da Economia por Classe ABC', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    # -----------------------------------------------------------------------
    # GRÁFICO 3: EVOLUÇÃO POR CLASSE XYZ
    # -----------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    
    for classe in ['XYZ-X', 'XYZ-Y', 'XYZ-Z']:
        df_classe = df_financial_mensal[df_financial_mensal['Classificacao'] == classe].sort_values('Periodo')
        ax3.plot(df_classe['Periodo_fmt'], df_classe['Reducao_Absoluta'], 
                marker='s', linewidth=2.5, markersize=6, label=classe)
    
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax3.set_xlabel('Período', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Redução (R$)', fontsize=11, fontweight='bold')
    ax3.set_title('Evolução da Economia por Classe XYZ', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(fontsize=10, framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    
    # -----------------------------------------------------------------------
    # GRÁFICO 4: HEATMAP - REDUÇÃO POR MÊS E ABC
    # -----------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[2, 0])
    
    df_abc = df_financial_mensal[df_financial_mensal['Classificacao'].str.startswith('ABC-')].copy()
    df_abc['Classe'] = df_abc['Classificacao'].str.replace('ABC-', '')
    pivot_abc = df_abc.pivot(index='Classe', columns='Periodo_fmt', values='Reducao_Absoluta')
    
    sns.heatmap(pivot_abc, annot=True, fmt='.0f', cmap='RdYlGn', center=0,
                ax=ax4, cbar_kws={'label': 'Redução (R$)'},
                linewidths=0.5)
    
    ax4.set_title('Heatmap: Economia por Classe ABC ao Longo dos Meses\n(Verde = Maior Economia)', 
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('Período', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Classe ABC', fontsize=11, fontweight='bold')
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    # -----------------------------------------------------------------------
    # GRÁFICO 5: HEATMAP - REDUÇÃO POR MÊS E XYZ
    # -----------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[2, 1])
    
    df_xyz = df_financial_mensal[df_financial_mensal['Classificacao'].str.startswith('XYZ-')].copy()
    df_xyz['Classe'] = df_xyz['Classificacao'].str.replace('XYZ-', '')
    pivot_xyz = df_xyz.pivot(index='Classe', columns='Periodo_fmt', values='Reducao_Absoluta')
    
    sns.heatmap(pivot_xyz, annot=True, fmt='.0f', cmap='RdYlGn', center=0,
                ax=ax5, cbar_kws={'label': 'Redução (R$)'},
                linewidths=0.5)
    
    ax5.set_title('Heatmap: Economia por Classe XYZ ao Longo dos Meses\n(Verde = Maior Economia)', 
                  fontsize=12, fontweight='bold')
    ax5.set_xlabel('Período', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Classe XYZ', fontsize=11, fontweight='bold')
    plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
    
    plt.show()
    print("   ✅ Gráficos de evolução mensal exibidos")
    
    # ========================================================================
    # 6. ANÁLISE DE TENDÊNCIAS
    # ========================================================================
    print("\n" + "="*80)
    print("📈 ANÁLISE DE TENDÊNCIAS")
    print("="*80)
    
    df_geral_sorted = df_geral.sort_values('Periodo')
    
    # Calcular variação entre primeiro e último mês
    primeiro_mes = df_geral_sorted.iloc[0]
    ultimo_mes = df_geral_sorted.iloc[-1]
    
    var_reducao = ultimo_mes['Reducao_Absoluta'] - primeiro_mes['Reducao_Absoluta']
    var_reducao_pct = (var_reducao / abs(primeiro_mes['Reducao_Absoluta']) * 100) if primeiro_mes['Reducao_Absoluta'] != 0 else 0
    
    print(f"\n🗓️  Primeiro mês ({primeiro_mes['Periodo_fmt']}):")
    print(f"   Redução: R$ {primeiro_mes['Reducao_Absoluta']:,.2f} ({primeiro_mes['Reducao_Percentual']:.1f}%)")
    
    print(f"\n🗓️  Último mês ({ultimo_mes['Periodo_fmt']}):")
    print(f"   Redução: R$ {ultimo_mes['Reducao_Absoluta']:,.2f} ({ultimo_mes['Reducao_Percentual']:.1f}%)")
    
    print(f"\n📊 VARIAÇÃO:")
    if var_reducao > 0:
        print(f"   ✅ Economia AUMENTOU R$ {var_reducao:,.2f} ({var_reducao_pct:+.1f}%)")
        print(f"   O modelo está melhorando ao longo do tempo!")
    elif var_reducao < 0:
        print(f"   ⚠️  Economia DIMINUIU R$ {abs(var_reducao):,.2f} ({var_reducao_pct:.1f}%)")
        print(f"   O modelo pode estar degradando!")
    else:
        print(f"   ➡️  Economia ESTÁVEL")
    
    # Média mensal
    media_reducao = df_geral_sorted['Reducao_Absoluta'].mean()
    print(f"\n💰 ECONOMIA MÉDIA MENSAL: R$ {media_reducao:,.2f}")
    
    # Mês com melhor/pior performance
    melhor_mes = df_geral_sorted.loc[df_geral_sorted['Reducao_Absoluta'].idxmax()]
    pior_mes = df_geral_sorted.loc[df_geral_sorted['Reducao_Absoluta'].idxmin()]
    
    print(f"\n🏆 MELHOR MÊS: {melhor_mes['Periodo_fmt']}")
    print(f"   Economia: R$ {melhor_mes['Reducao_Absoluta']:,.2f} ({melhor_mes['Reducao_Percentual']:.1f}%)")
    
    print(f"\n⚠️  PIOR MÊS: {pior_mes['Periodo_fmt']}")
    print(f"   Economia: R$ {pior_mes['Reducao_Absoluta']:,.2f} ({pior_mes['Reducao_Percentual']:.1f}%)")
    
    # Consistência
    desvio_padrao = df_geral_sorted['Reducao_Absoluta'].std()
    cv = (desvio_padrao / abs(media_reducao) * 100) if media_reducao != 0 else 0
    
    print(f"\n📊 CONSISTÊNCIA:")
    print(f"   Desvio Padrão: R$ {desvio_padrao:,.2f}")
    print(f"   Coeficiente de Variação: {cv:.1f}%")
    if cv < 20:
        print(f"   ✅ Performance MUITO CONSISTENTE")
    elif cv < 50:
        print(f"   ⚠️  Performance MODERADAMENTE VARIÁVEL")
    else:
        print(f"   ❌ Performance MUITO VARIÁVEL")
    
    print("\n" + "="*80)
    
else:
    print("\n⚠️ Coluna VALOR_UNITARIO não encontrada. Análise financeira mensal não executada.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# SALVAR DATASETS COM CLASSIFICAÇÃO ABC-XYZ
# ==================================================================================
print("\n💾 Salvando datasets com classificação ABC-XYZ...")

# Salvar componentes com classificação
Helpers.save_output_dataset(
    context=context,
    output_name="predictions_components_abc_xyz",
    data_frame=df_components_ma
)
print("   ✅ predictions_components_abc_xyz salvo")

# Salvar produtos com classificação
Helpers.save_output_dataset(
    context=context,
    output_name="predictions_products_abc_xyz",
    data_frame=df_products_ma
)
print("   ✅ predictions_products_abc_xyz salvo")

print("\n" + "="*80)
print("✅ ANÁLISE COMPLETA COM CLASSIFICAÇÃO ABC-XYZ FINALIZADA!")
print("="*80)