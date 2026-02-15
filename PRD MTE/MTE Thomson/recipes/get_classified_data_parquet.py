import sys
if "/app" in sys.path:
    sys.path.remove("/app")

from utils.notebookhelpers.helpers import Helpers
from utils.dtos.rc_ml_model import RCMLModel

import pandas as pd
import pickle
import tempfile
import base64
import gzip
import io


class GetDatasetParquetEndpoint(RCMLModel):

    def __init__(self):
        self.datasets = {}

    def load(self, artifacts: dict):
        if 'model_artifacts' in artifacts:
            with open(artifacts['model_artifacts'], 'rb') as f:
                self.datasets = pickle.load(f)
            print(f"✓ Loaded {len(self.datasets)} datasets: {list(self.datasets.keys())}")

    def predict(self, input_data: dict) -> list:
        try:
            data = input_data.get('data', [{}])[0]
            dataset_name = data.get('dataset_name')

            if not dataset_name:
                return [{'status': 'error', 'message': f'Informe "dataset_name". Disponíveis: {list(self.datasets.keys())}'}]

            if dataset_name not in self.datasets:
                return [{'status': 'error', 'message': f'Dataset "{dataset_name}" não encontrado. Disponíveis: {list(self.datasets.keys())}'}]

            df = self.datasets[dataset_name]

            buf = io.BytesIO()
            df.to_parquet(buf, engine='pyarrow', compression='gzip', index=False)
            compressed = gzip.compress(buf.getvalue())
            b64 = base64.b64encode(compressed).decode('utf-8')

            print(f"✓ {dataset_name}: {len(df):,} rows -> {len(b64):,} chars")
            return [{'data': b64}]

        except Exception as e:
            return [{'status': 'error', 'message': str(e)}]


context = Helpers.getOrCreateContext(contextId='endpoint_get_dataset_parquet', localVars=locals())

# Carregar TODOS os datasets conectados ao nó
datasets = {}
for name in Helpers.getAllEntities(context):
    try:
        df = Helpers.getEntityData(context, name)
        datasets[name] = df
        print(f"✓ {name}: {len(df):,} rows")
    except Exception as e:
        print(f"✗ {name}: {e}")

print(f"✓ Total: {len(datasets)} datasets")

# Salvar todos como pickle
artifacts_path = tempfile.mktemp(suffix='.pkl')
with open(artifacts_path, 'wb') as f:
    pickle.dump(datasets, f)

# Registrar endpoint
outputCollection = Helpers.createOutputCollection(context)
model_output = Helpers.create_template_output_rc_ml_model(
    context,
    'get_dataset_parquet',
    GetDatasetParquetEndpoint,
    {'model_artifacts': artifacts_path}
)
outputCollection.addTemplateOutput(model_output)
Helpers.save(context)
print("✓ Endpoint registered")