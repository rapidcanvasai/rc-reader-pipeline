# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# API Operations:
# op == 'analyze-inventory': Analyzes inventory and returns consolidated metrics

from utils.notebookhelpers.helpers import Helpers
from utils.dtos.rc_ml_model import RCMLModel
from utils.rc.client.requests import Requests
from utils.rc.dtos.project import Project
from utils.rc.dtos.dataset import Dataset
from utils.rcclient.enums import DatasetFileType as FileType

import pandas as pd
import json
import os

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
class InventoryAnalysisModel(RCMLModel):
    CACHE_NAMESPACE = "INVENTORY_ANALYSIS"
    PROJECT_ID = "469fe89c-b494-48ae-8177-eb4759ebd134"
    DATASET_NAME = "Inventory"
    
    def load(self, artifacts):
        """Model initialization (empty; no external dependencies)."""
        pass
    
    def _load_dataset(self, dataset_name):
        """Loads a project dataset and returns it as a DataFrame."""
        try:
            # Fetch the project
            project = Project.find_by_id(self.PROJECT_ID)
            all_datasets = project.getAllDatasets()
            
            # Ensure the dataset exists
            if dataset_name not in all_datasets:
                raise ValueError(f"Dataset '{dataset_name}' not found in the project")
            
            # Download the dataset
            file_path = all_datasets[dataset_name].download_dataset(
                folder_path="/app/data",
                scenario_id=None,
                project_run_entry_id=None,
                file_type=FileType.PARQUET
            )
            
            # Load into a DataFrame
            df = pd.read_parquet(file_path)
            
            # Remove the temporary file
            os.unlink(file_path)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading dataset '{dataset_name}': {str(e)}")
    
    def _analyze_inventory(self, df, price_column='ASK_PRICE', model_column='MODEL'):
        """
        Analyzes equipment inventory and returns consolidated metrics.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with inventory data.
        price_column : str, optional
            Price column name (default: 'ASK_PRICE').
        model_column : str, optional
            Model column name (default: 'MODEL').
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'total_value': Total inventory value
            - 'total_quantity': Total number of equipment
            - 'average_equipment_value': Average value per equipment
            - 'by_model': Per-model analysis (total value, quantity, average value)
        """
        # Work on a copy to avoid mutating the original DataFrame
        df_temp = df.copy()
        
        # Convert price to numeric
        df_temp['PRICE_NUM'] = pd.to_numeric(df_temp[price_column], errors='coerce')
        
        # Overall metrics
        total_value = df_temp['PRICE_NUM'].sum()
        total_quantity = len(df_temp)
        average_equipment_value = df_temp['PRICE_NUM'].mean()
        
        # Per-model analysis
        by_model = df_temp.groupby(model_column)['PRICE_NUM'].agg([
            ('total_value', 'sum'),
            ('quantity', 'count'),
            ('average_value', 'mean')
        ]).sort_values('total_value', ascending=False)
        
        # Round values for readability
        by_model_formatted = by_model.copy()
        by_model_formatted['total_value'] = by_model_formatted['total_value'].round(2)
        by_model_formatted['average_value'] = by_model_formatted['average_value'].round(2)
        
        # Convert DataFrame to a JSON-serializable structure
        by_model_json = json.loads(
            by_model_formatted.reset_index().to_json(orient='records', date_format='iso')
        )
        
        # Build the result payload
        result = {
            'total_value': float(round(total_value, 2)),
            'total_quantity': int(total_quantity),
            'average_equipment_value': float(round(average_equipment_value, 2)),
            'by_model': by_model_json
        }
        
        return result
    
    def predict(self, model_input, context):
        """Processes the inventory analysis operation."""
        token = Helpers.get_user_token(context)
        Requests.setToken(token)
        
        if "op" not in model_input:
            raise ValueError(f"No 'op' provided in input: {model_input}")
        
        operation = model_input["op"]
        
        if operation == 'analyze-inventory':
            # Read parameters from model_input (dataset fixed to "Inventory")
            price_column = model_input.get('price_column', 'ASK_PRICE')
            model_column = model_input.get('model_column', 'MODEL')
            
            # Load the Inventory dataset
            df = self._load_dataset(self.DATASET_NAME)
            
            # Run analysis
            result = self._analyze_inventory(df, price_column, model_column)
            
            return {
                "success": True,
                "dataset_used": self.DATASET_NAME,
                "rows_analyzed": result['total_quantity'],
                "analysis": result
            }
        
        raise ValueError(f"Invalid operation '{operation}'. Valid operation: 'analyze-inventory'")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Initialize model
model_obj = InventoryAnalysisModel()
model_obj.load({})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Test operation
print("="*80)
print("TESTING OPERATION: analyze-inventory")
print("="*80)
result_analysis = model_obj.predict({
    "op": "analyze-inventory",
    "price_column": "ASK_PRICE",
    "model_column": "MODEL"
}, context)

print(f"Success: {result_analysis['success']}")
print(f"Dataset used: {result_analysis['dataset_used']}")
print(f"Rows analyzed: {result_analysis['rows_analyzed']:,}")
print(f"\nAnalysis Results:")
print(f"  Total Value: ${result_analysis['analysis']['total_value']:,.2f}")
print(f"  Total Quantity: {result_analysis['analysis']['total_quantity']:,}")
print(f"  Average Value per Equipment: ${result_analysis['analysis']['average_equipment_value']:,.2f}")
print(f"  Models Analyzed: {len(result_analysis['analysis']['by_model'])}")

if result_analysis['analysis']['by_model']:
    print(f"\nTop 5 Models by Total Value:")
    for i, model in enumerate(result_analysis['analysis']['by_model'][:5], 1):
        print(f"  {i}. {model['MODEL']}: ${model['total_value']:,.2f} ({model['quantity']} units)")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Save model as a prediction service
Helpers.save_output_rc_ml_model(
    context=context, 
    model_name='inventory-analysis-api', 
    model_obj=InventoryAnalysisModel, 
    artifacts={}
)

print("\n" + "="*80)
print("✅ MODEL SAVED AS: 'inventory-analysis-api'")
print("="*80)
print("\nAvailable operation:")
print("  • analyze-inventory: Analyzes inventory with consolidated metrics")
print("\nDataset used: Inventory")
print("\nExample usage:")
print('  {"op": "analyze-inventory"}')
print('  {"op": "analyze-inventory", "price_column": "ASK_PRICE", "model_column": "MODEL"}')