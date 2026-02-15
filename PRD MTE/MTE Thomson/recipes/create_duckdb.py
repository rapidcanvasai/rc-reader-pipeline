"""
Recipe RapidCanvas: Criar banco DuckDB a partir de CSV(s).

Inputs:
    - input_data: DataFrame(s) CSV carregado(s) pelo Canvas

Outputs:
    - Artifact: Arquivo DuckDB com as tabelas criadas
"""

from utils.notebookhelpers.helpers import Helpers
import duckdb
import os
import re
import shutil


def sanitize_name(name: str) -> str:
    """Converte nome para snake_case válido."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    name = name.lower()
    return name


def rename_columns_to_snake_case(con, table_name: str) -> int:
    """Renomeia todas as colunas de uma tabela para snake_case."""
    columns = con.execute(f"DESCRIBE {table_name}").fetchall()
    renamed = 0

    for col in columns:
        old_name = col[0]
        new_name = sanitize_name(old_name)

        if old_name != new_name:
            try:
                con.execute(f'ALTER TABLE {table_name} RENAME COLUMN "{old_name}" TO {new_name}')
                renamed += 1
            except Exception:
                pass

    return renamed


def create_duckdb_from_dataframes(dataframes: dict, db_path: str) -> dict:
    """
    Cria banco DuckDB a partir de um dicionário de DataFrames.

    Args:
        dataframes: Dict com {nome_tabela: DataFrame}
        db_path: Caminho para salvar o arquivo .duckdb

    Returns:
        Dict com estatísticas de criação
    """
    con = duckdb.connect(db_path)

    success = 0
    errors = []
    table_stats = []

    for table_name, df in dataframes.items():
        safe_table_name = sanitize_name(table_name)

        try:
            con.execute(f"CREATE TABLE {safe_table_name} AS SELECT * FROM df")
            renamed = rename_columns_to_snake_case(con, safe_table_name)
            count = con.execute(f"SELECT COUNT(*) FROM {safe_table_name}").fetchone()[0]

            table_stats.append({
                "table_name": safe_table_name,
                "original_name": table_name,
                "row_count": count,
                "columns_renamed": renamed
            })
            success += 1

        except Exception as e:
            errors.append({"table": table_name, "error": str(e)})

    con.close()

    return {
        "tables_created": success,
        "total_tables": len(dataframes),
        "errors": errors,
        "table_stats": table_stats,
        "db_path": db_path
    }


# ============================================================================
# RECIPE PRINCIPAL
# ============================================================================

# Inicializa contexto RapidCanvas
# O internalContext é injetado pelo runtime do Canvas
context = Helpers.getOrCreateContext("create_duckdb", locals())

# Parâmetros configuráveis
db_filename = Helpers.getParam(context, "db_filename") or "output.duckdb"
artifact_name = Helpers.getParam(context, "artifact_name") or "database"

# Carrega datasets
df_main = Helpers.getEntityData(context, "df_main_confidence_interval")
df_vendas = Helpers.getEntityData(context, "df_vendas")
df_produtos = Helpers.getEntityData(context, "df_produtos")

dataframes = {
    "df_main": df_main,
    "df_vendas": df_vendas,
    "df_produtos": df_produtos
}

if not dataframes:
    raise ValueError("Nenhum DataFrame válido encontrado nos inputs")

print(f"DataFrames carregados: {list(dataframes.keys())}")

# Cria diretório de artefato
artifact_dir = Helpers.getOrCreateArtifactsDir(context, artifact_name, purgeOld=True)
print(f"Diretório do artefato: {artifact_dir}")

# Caminho do DuckDB dentro do artefato
db_path = os.path.join(artifact_dir, db_filename)

# Remove arquivo existente se houver
if os.path.exists(db_path):
    os.remove(db_path)

# Cria o banco DuckDB
result = create_duckdb_from_dataframes(dataframes, db_path)

# Log do resultado
print(f"\n{'='*60}")
print(f"RESUMO DA CRIAÇÃO DO DUCKDB")
print(f"{'='*60}")
print(f"Tabelas criadas: {result['tables_created']}/{result['total_tables']}")

if result['table_stats']:
    print(f"\nTabelas:")
    for stat in result['table_stats']:
        print(f"  - {stat['table_name']}: {stat['row_count']:,} linhas, {stat['columns_renamed']} colunas renomeadas")

if result['errors']:
    print(f"\nErros:")
    for err in result['errors']:
        print(f"  - {err['table']}: {err['error']}")

db_size_mb = os.path.getsize(db_path) / 1024 / 1024
print(f"\nArquivo: {db_path}")
print(f"Tamanho: {db_size_mb:.2f} MB")

# Salva o artefato
Helpers.save_output_artifacts(context, artifact_name)