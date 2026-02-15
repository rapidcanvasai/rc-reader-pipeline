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

import pandas as pd
import numpy as np

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Load data from entity

df_results = Helpers.getEntityData(context, 'df_results_count_sales')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Calculate summary metrics for all equipment models

def calculate_metrics_by_equipment(df):
    """
    Calculate summary metrics for each equipment model and ML model combination
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with results from df_results_count_sales
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with one row per (equipment, model) combination and metrics as columns
    """
    # Copy data
    validation_data = df.copy()
    
    # Filter by validation period if column exists
    if 'period_type' in validation_data.columns:
        validation_data = validation_data[validation_data['period_type'] == 'validation']
    
    # Get unique combinations of equipment and model
    equipment_model_combinations = validation_data[['Model_Eq.', 'Model']].drop_duplicates()
    
    # List to store results
    results = []
    
    # Calculate metrics for each equipment and model combination
    for _, row in equipment_model_combinations.iterrows():
        equipment = row['Model_Eq.']
        model = row['Model']
        
        # Filter data for this equipment and model
        equipment_data = validation_data[
            (validation_data['Model_Eq.'] == equipment) & 
            (validation_data['Model'] == model)
        ]
        
        # Aggregate by date (sum all values)
        equipment_agg = equipment_data.groupby('Sell_Date').agg({
            'Count': 'sum',
            'Pred': 'sum'
        }).reset_index()
        
        # Get arrays
        y_true = equipment_agg['Count'].values
        y_pred = equipment_agg['Pred'].values
        
        # Skip if no data
        if len(y_true) == 0:
            continue
        
        # Calculate metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # Percentage error
        total_error = np.sum(y_pred - y_true)
        percentage_error = (total_error / np.sum(y_true) * 100) if np.sum(y_true) != 0 else 0
        
        # MAPE (filtered for values >= 5)
        mask = y_true >= 5
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
        
        # WMAPE
        wmape = (np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100) if np.sum(np.abs(y_true)) != 0 else 0
        
        # Calculate accuracy (Predicted / Actual) as percentage
        total_actual = np.sum(y_true)
        total_predicted = np.sum(y_pred)
        accuracy = (total_predicted / total_actual) if total_actual != 0 else 0
        
        # Use MAPE for confidence calculation, fallback to WMAPE if MAPE is NaN
        mape_for_confidence = mape if not np.isnan(mape) else wmape
        
        # Calculate confidence level based on MAPE
        if mape_for_confidence < 30:
            confidence = 'high'
        elif mape_for_confidence <= 60:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Store results
        results.append({
            'Model_Equipment': equipment,
            'Model': model,
            'MAE': mae,
            'Percentage_Error': percentage_error,
            'MAPE': mape if not np.isnan(mape) else wmape,
            'WMAPE': wmape,
            'Accuracy': accuracy,
            'RMSE': rmse,
            'Confidence': confidence,
            'Total_Actual': total_actual,
            'Total_Predicted': total_predicted,
            'N_Observations': len(y_true)
        })
    
    # Create DataFrame
    metrics_df = pd.DataFrame(results)
    
    # Sort by equipment model and model
    metrics_df = metrics_df.sort_values(['Model_Equipment', 'Model']).reset_index(drop=True)
    
    return metrics_df


# Execute function
output_metrics_df = calculate_metrics_by_equipment(df_results)

# Display summary
print(f"Total equipment-model combinations analyzed: {len(output_metrics_df)}")
print(f"Unique equipment models: {output_metrics_df['Model_Equipment'].nunique()}")
print(f"Unique ML models: {output_metrics_df['Model'].nunique()}")
print(f"ML models found: {output_metrics_df['Model'].unique().tolist()}")
print(f"\nMetrics summary:")
print(output_metrics_df[['Model_Equipment', 'Model', 'MAE', 'MAPE', 'Accuracy', 'RMSE', 'Confidence']].head(10))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Save output dataset

Helpers.save_output_dataset(
    context=context, 
    output_name='equipment_metrics_summary', 
    data_frame=output_metrics_df
)