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
Inventory = Helpers.getEntityData(context, 'NewInventoryExtract')


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

Helpers.save_output_dataset(context=context, output_name='Inventory', data_frame=Inventory)