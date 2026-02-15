# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Required imports
from utils.notebookhelpers.helpers import Helpers
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Carregar os dados
test_entity = Helpers.getEntityData(context, 'df_future_count_sales')

# Converter para DataFrame se necessário
if not isinstance(test_entity, pd.DataFrame):
    output_df = pd.DataFrame(test_entity)
else:
    output_df = test_entity

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Configuração do banco de dados MySQL
DB_CONFIG = {
    'host': '35.202.242.146',
    'port': '3306',
    'database': 'milton_cat_db',
    'user': 'milton_cat_db_user',
    'password': '_ai1I@l:}r]Qe}nK'
}

# Codificar senha (necessário por causa dos caracteres especiais)
encoded_password = quote_plus(DB_CONFIG['password'])

connection_string = f"mysql+pymysql://{DB_CONFIG['user']}:{encoded_password}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Salvar no banco de dados MySQL
TABLE_NAME = 'df_future_count_sales'

try:
    engine = create_engine(connection_string)
    
    output_df.to_sql(
        name=TABLE_NAME,
        con=engine,
        if_exists='replace',
        index=False,
        chunksize=1000
    )
    
    print(f"✅ Dados salvos com sucesso na tabela '{TABLE_NAME}'")
    print(f"   Total de registros: {len(output_df)}")
    
except Exception as e:
    print(f"❌ Erro ao salvar no banco: {e}")
    raise

finally:
    engine.dispose()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Salvar output para o framework (obrigatório para a recipe funcionar)
Helpers.save_output_dataset(context=context, output_name='outputDataset', data_frame=output_df)