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
# Your code goes here
import pandas as pd
output_df_1 = pd.DataFrame()

# Azure Blob Storage connection string
anthropic = Helpers.get_secret(context, "anthropic")
print(f"✅ anthropic '{anthropic}''")

openai = Helpers.get_secret(context, "openai")
print(f"✅ openai '{openai}''")



# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

Helpers.save_output_dataset(context=context, output_name='outputDataset', data_frame=output_df_1)