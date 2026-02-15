# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Required imports
from utils.notebookhelpers.helpers import Helpers
import pandas as pd
import os
from pathlib import Path

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Função para normalizar nomes de colunas
def normalize_column_names(df):
    """
    Normaliza nomes de colunas:
    - Substitui espaços por underscores
    - Remove espaços extras
    """
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    return df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Download all files from Consolidated-Historical-Data folder
print("📂 Baixando arquivos da pasta 'Consolidated-Historical-Data'...")

try:
    consolidated_files = Helpers.downloadArtifacts(context, 'Consolidated-Historical-Data')
    
    # Filtrar apenas arquivos CSV e XLSX
    data_files = {filename: file_obj for filename, file_obj in consolidated_files.items() 
                  if filename.lower().endswith(('.csv', '.xlsx', '.xls'))}
    
    if not data_files:
        raise ValueError("Nenhum arquivo CSV ou XLSX encontrado na pasta.")
    
    print(f"✅ {len(data_files)} arquivo(s) encontrado(s): {list(data_files.keys())}")
    
except Exception as e:
    raise RuntimeError(f"❌ Erro ao baixar arquivos: {e}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Análise de colunas de cada arquivo (COM NORMALIZAÇÃO)
print("\n" + "="*60)
print("🔍 ANÁLISE DE ESTRUTURA DOS ARQUIVOS")
print("="*60)

file_columns = {}
file_info = {}

for filename, file_obj in data_files.items():
    try:
        # Ler apenas o header para análise
        if filename.lower().endswith('.csv'):
            try:
                df_sample = pd.read_csv(file_obj, nrows=5, encoding='utf-8')
            except UnicodeDecodeError:
                file_obj.seek(0)
                df_sample = pd.read_csv(file_obj, nrows=5, encoding='latin-1')
        elif filename.lower().endswith(('.xlsx', '.xls')):
            df_sample = pd.read_excel(file_obj, nrows=5)
        
        # NORMALIZAR NOMES DE COLUNAS
        df_sample = normalize_column_names(df_sample)
        
        file_obj.seek(0)  # Reset para leitura completa depois
        
        file_columns[filename] = list(df_sample.columns)
        file_info[filename] = {
            'num_columns': len(df_sample.columns),
            'columns': list(df_sample.columns)
        }
        
        print(f"\n📄 {filename}")
        print(f"   📊 Número de colunas: {len(df_sample.columns)}")
        print(f"   📋 Colunas (normalizadas): {list(df_sample.columns)[:5]}{'...' if len(df_sample.columns) > 5 else ''}")
        
    except Exception as e:
        print(f"\n❌ Erro ao analisar {filename}: {e}")

# Identificar colunas comuns e exclusivas
print("\n" + "="*60)
print("📊 ANÁLISE DE COMPATIBILIDADE DE COLUNAS")
print("="*60)

if file_columns:
    all_columns = set()
    for cols in file_columns.values():
        all_columns.update(cols)
    
    # Colunas que aparecem em todos os arquivos
    common_columns = set(file_columns[list(file_columns.keys())[0]])
    for cols in file_columns.values():
        common_columns = common_columns.intersection(set(cols))
    
    print(f"\n✅ Colunas COMUNS a todos os arquivos ({len(common_columns)}):")
    for col in sorted(common_columns):
        print(f"   - {col}")
    
    print(f"\n⚠️  Colunas que NÃO aparecem em todos os arquivos:")
    for filename, cols in file_columns.items():
        unique_cols = set(cols) - common_columns
        if unique_cols:
            print(f"\n   📄 {filename} ({len(unique_cols)} colunas exclusivas):")
            for col in sorted(unique_cols):
                print(f"      - {col}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# OPÇÃO 1: Merge mantendo TODAS as colunas (com NaN onde não houver dados)
# OPÇÃO 2: Merge mantendo APENAS colunas comuns

# Escolha aqui qual opção usar:
MERGE_OPTION = "ALL_COLUMNS"  # ou "COMMON_ONLY"

print("\n" + "="*60)
print(f"🔄 MERGE DE DADOS - Modo: {MERGE_OPTION}")
print("="*60)

try:
    dataframes = []
    
    for filename, file_obj in data_files.items():
        try:
            print(f"\n📄 Processando: {filename}")
            
            # Ler arquivo completo
            if filename.lower().endswith('.csv'):
                try:
                    df = pd.read_csv(file_obj, encoding='utf-8')
                    encoding_used = 'utf-8'
                except UnicodeDecodeError:
                    file_obj.seek(0)
                    df = pd.read_csv(file_obj, encoding='latin-1')
                    encoding_used = 'latin-1'
                print(f"   Tipo: CSV (encoding: {encoding_used})")
                
            elif filename.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_obj)
                print(f"   Tipo: Excel")
            
            # NORMALIZAR NOMES DE COLUNAS IMEDIATAMENTE
            print(f"   📋 Colunas antes da normalização: {list(df.columns)[:3]}...")
            df = normalize_column_names(df)
            print(f"   📋 Colunas após normalização: {list(df.columns)[:3]}...")
            
            # Se escolheu apenas colunas comuns, filtrar
            if MERGE_OPTION == "COMMON_ONLY":
                available_common = [col for col in common_columns if col in df.columns]
                df = df[available_common]
                print(f"   📋 Mantendo apenas {len(available_common)} colunas comuns")
            
            print(f"   📊 Linhas: {len(df):,}")
            print(f"   📋 Colunas após filtro: {len(df.columns)}")
            
            # Adicionar coluna de origem
            df['source_file'] = filename
            
            dataframes.append(df)
            print(f"   ✅ Sucesso!")
            
        except Exception as e:
            print(f"   ❌ ERRO: {e}")
            continue
    
    if not dataframes:
        raise ValueError("❌ Nenhum arquivo pôde ser lido com sucesso.")
    
    # Combinar todos os dataframes
    print(f"\n{'='*60}")
    print(f"🔄 Combinando {len(dataframes)} dataframe(s)...")
    
    df_machine_sales_raw = pd.concat(dataframes, ignore_index=True)
    
    print(f"✅ Combinação concluída!")
    print(f"   📊 Total de linhas: {len(df_machine_sales_raw):,}")
    print(f"   📋 Total de colunas: {len(df_machine_sales_raw.columns)}")
    
    # Análise de dados faltantes por coluna
    print(f"\n📊 Análise de valores nulos por coluna:")
    null_analysis = df_machine_sales_raw.isnull().sum()
    null_analysis = null_analysis[null_analysis > 0].sort_values(ascending=False)
    
    if len(null_analysis) > 0:
        print(f"\n⚠️  {len(null_analysis)} colunas têm valores nulos:")
        for col, count in null_analysis.head(20).items():  # Top 20
            pct = (count / len(df_machine_sales_raw)) * 100
            print(f"   - {col}: {count:,} nulos ({pct:.1f}%)")
        
        if len(null_analysis) > 20:
            print(f"   ... e mais {len(null_analysis) - 20} colunas")
    else:
        print(f"   ✅ Nenhum valor nulo encontrado!")
    
    # Análise por arquivo de origem
    print(f"\n📊 Distribuição de dados por arquivo:")
    source_dist = df_machine_sales_raw['source_file'].value_counts()
    for source, count in source_dist.items():
        pct = (count / len(df_machine_sales_raw)) * 100
        print(f"   - {source}: {count:,} ({pct:.1f}%)")
    
except Exception as e:
    raise RuntimeError(f"❌ Erro ao processar os arquivos: {e}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Validações e limpeza de dados
if df_machine_sales_raw is not None:
    print("\n" + "="*60)
    print("🔍 VALIDAÇÕES E LIMPEZA DE DADOS")
    print("="*60)
    
    # 1. Verificar e remover duplicatas
    print("\n1️⃣ Verificando duplicatas...")
    
    # Agora procurar por Sales_ID (normalizado)
    if 'Sales_ID' in df_machine_sales_raw.columns:
        initial_rows = len(df_machine_sales_raw)
        duplicates = df_machine_sales_raw.duplicated(subset=['Sales_ID']).sum()
        print(f"   📊 Duplicatas por Sales_ID: {duplicates:,}")
        
        if duplicates > 0:
            df_machine_sales_raw = df_machine_sales_raw.drop_duplicates(subset=['Sales_ID'], keep='first')
            final_rows = len(df_machine_sales_raw)
            print(f"   ✅ Duplicatas removidas: {initial_rows - final_rows:,}")
    else:
        print("   ⚠️  Coluna 'Sales_ID' não encontrada")
        # Remover duplicatas completas
        initial_rows = len(df_machine_sales_raw)
        df_machine_sales_raw = df_machine_sales_raw.drop_duplicates()
        final_rows = len(df_machine_sales_raw)
        print(f"   Removendo linhas completamente duplicadas: {initial_rows - final_rows:,}")
    
    # 2. Processar datas (normalizado: Sell_Date)
    print("\n2️⃣ Processando datas...")
    if 'Sell_Date' in df_machine_sales_raw.columns:
        try:
            df_machine_sales_raw['Sell_Date'] = pd.to_datetime(
                df_machine_sales_raw['Sell_Date'], 
                errors='coerce'
            )
            
            valid_dates = df_machine_sales_raw['Sell_Date'].dropna()
            if len(valid_dates) > 0:
                min_date = valid_dates.min()
                max_date = valid_dates.max()
                print(f"   📅 Range de datas: {min_date.strftime('%Y-%m-%d')} até {max_date.strftime('%Y-%m-%d')}")
                
                months_available = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month)
                print(f"   📊 Meses de histórico: {months_available}")
                
                if months_available < 27:
                    print(f"   ⚠️  Menos de 27 meses de dados (recomendado: 27+)")
                else:
                    print(f"   ✅ Histórico suficiente")
        except Exception as e:
            print(f"   ❌ Erro ao processar datas: {e}")
    else:
        print(f"   ⚠️  Coluna 'Sell_Date' não encontrada")
        print(f"   💡 Colunas disponíveis: {list(df_machine_sales_raw.columns)[:10]}...")
    
    # 3. Verificar Fleet
    print("\n3️⃣ Verificando Fleet...")
    if 'Fleet' in df_machine_sales_raw.columns:
        fleet_values = df_machine_sales_raw['Fleet'].value_counts()
        print(f"   📊 Valores em Fleet:")
        for fleet, count in fleet_values.items():
            pct = (count / len(df_machine_sales_raw)) * 100
            print(f"      - {fleet}: {count:,} ({pct:.1f}%)")
        
        if 'New' in fleet_values.index:
            print(f"   ✅ Fleet='New' disponível")
        else:
            print(f"   ⚠️  Fleet='New' NÃO encontrado")
    else:
        print(f"   ⚠️  Coluna 'Fleet' não encontrada")
    
    # 4. Verificar Model
    print("\n4️⃣ Verificando Model...")
    if 'Model' in df_machine_sales_raw.columns:
        total_models = df_machine_sales_raw['Model'].nunique()
        print(f"   📊 Total de modelos únicos: {total_models:,}")
        
        top_models = df_machine_sales_raw['Model'].value_counts().head(10)
        print(f"   📋 Top 10 modelos:")
        for model, count in top_models.items():
            print(f"      - {model}: {count:,} registros")
    else:
        print(f"   ⚠️  Coluna 'Model' não encontrada")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Preview do DataFrame final
if df_machine_sales_raw is not None:
    print("\n" + "="*60)
    print("📊 PREVIEW DO DATAFRAME CONSOLIDADO")
    print("="*60)
    
    print(f"\n📋 Colunas finais ({len(df_machine_sales_raw.columns)}):")
    print(list(df_machine_sales_raw.columns))
    
    print("\n📋 Primeiras 5 linhas:")
    print(df_machine_sales_raw.head())
    
    print("\n📊 Informações do DataFrame:")
    print(df_machine_sales_raw.info())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Salvar o DataFrame consolidado como dataset
if df_machine_sales_raw is not None:
    print("\n💾 Salvando DataFrame consolidado...")
    
    df_machine_sales_final = df_machine_sales_raw.copy()
    
    try:
        Helpers.save_output_dataset(
            context=context, 
            output_name='Machine Sales New',
            data_frame=df_machine_sales_final
        )
        print(f"✅ Dataset 'Machine Sales' salvo com sucesso!")
        print(f"   📊 {len(df_machine_sales_final):,} linhas")
        print(f"   📋 {len(df_machine_sales_final.columns)} colunas")
    except Exception as e:
        print(f"❌ Erro ao salvar dataset: {e}")
else:
    print("❌ Nenhum dado para salvar")