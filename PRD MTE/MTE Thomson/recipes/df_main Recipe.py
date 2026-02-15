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
import numpy as np

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Your code goes here
test_entity = Helpers.getEntityData(context, 'df_main')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# mocado enquanto o modelo com conformal prediction não está na plataforma
test_entity['min'] = 0.95 * test_entity['Final_order']
test_entity['max'] = 1.05 * test_entity['Final_order']

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# mocado enquanto o modelo com conformal prediction não está na plataforma
precision = [np.random.choice(['high', 'medium', 'low']) for i in range(len(test_entity))]
test_entity['precision'] = precision


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
test_entity['precision'].unique()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
Helpers.save_output_dataset(context=context, output_name='df_main_confidence_interval', data_frame=test_entity)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE


