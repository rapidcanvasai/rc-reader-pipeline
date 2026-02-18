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
from datetime import datetime  # Import datetime for dynamic file naming

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

print(f"Context : {context}")
token = Helpers.get_user_token(context)
print(f"Token: {token}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Your code goes here
output_df_1 = Helpers.getEntityData(context, 'ProfitabilityExtract')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Get or create artifacts directory
artifactsDir = Helpers.getOrCreateArtifactsDir(context, artifactsId="Historical-Data")

# Generate dynamic filename with current date (format: YYYY-MM-DD)
current_date = datetime.now().strftime('%Y-%m-%d')
filename = f'Last_Day_Sales_{current_date}.csv'

# Save the DataFrame as CSV file in the artifacts directory
output_df_1.to_csv(artifactsDir + '/' + filename, index=False)

# Add artifact to output
Helpers.save_output_artifacts(context=context, artifact_name='Historical-Data')