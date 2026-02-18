# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# API Operations:
# op == 'get-forecast-data': Returns forecast data for a specific equipment or all aggregated
# op == 'save-demand-adjustment': Saves a demand adjustment to MySQL database
# op == 'get-demand-adjustments': Retrieves demand adjustments from MySQL database

from utils.notebookhelpers.helpers import Helpers
from utils.dtos.rc_ml_model import RCMLModel
from utils.rc.client.requests import Requests
from utils.rc.dtos.project import Project
from utils.rc.dtos.dataset import Dataset
from utils.rcclient.enums import DatasetFileType as FileType

import pandas as pd
import numpy as np
import json
import os
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# MySQL Database Configuration
DB_CONFIG = {
    'host': '35.202.242.146',
    'port': '3306',
    'database': 'milton_cat_db',
    'user': 'milton_cat_db_user',
    'password': '_ai1I@l:}r]Qe}nK'
}

encoded_password = quote_plus(DB_CONFIG['password'])
MYSQL_CONNECTION_STRING = f"mysql+pymysql://{DB_CONFIG['user']}:{encoded_password}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
class ForecastDataModel(RCMLModel):
    CACHE_NAMESPACE = "FORECAST_DATA"
    PROJECT_ID = "469fe89c-b494-48ae-8177-eb4759ebd134"
    TEST_MONTHS = 6  # Fixed test months parameter
    
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
    
    def _load_forecast_data(self):
        """
        Loads all necessary datasets for forecast
        
        Returns:
        --------
        tuple : (df_sales, df_results, df_future, df_crm)
        """
        # Load historical sales data
        df_sales = self._load_dataset('df_count_machine_sales')
        df_sales['Sell_Date'] = pd.to_datetime(df_sales['Sell_Date'])
        
        # Load validation results
        df_results = self._load_dataset('df_results_count_sales')
        df_results['Sell_Date'] = pd.to_datetime(df_results['Sell_Date'])
        
        # Load future predictions
        df_future = self._load_dataset('df_future_count_sales')
        df_future['Sell_Date'] = pd.to_datetime(df_future['Sell_Date'])
        
        # Load CRM data
        try:
            df_crm = self._load_dataset('crm_data')
            df_crm['Sell_Date'] = pd.to_datetime(df_crm['Sell_Date'])
            # Keep only relevant columns
            df_crm = df_crm[['Model_Eq.', 'Sell_Date', 'Count']].copy()
        except Exception as e:
            print(f"Warning: Could not load CRM data: {e}")
            df_crm = pd.DataFrame(columns=['Model_Eq.', 'Sell_Date', 'Count'])
        
        return df_sales, df_results, df_future, df_crm
    
    def _get_forecast_data_by_equipment(self, model_eq, model_name, df_sales, df_results, df_future, df_crm):
        """
        Extracts all necessary data to build the forecast chart for a specific Model_Eq. or all aggregated
        
        Parameters:
        -----------
        model_eq : str
            Equipment model (e.g., 'D6K', 'Cat320D', etc.) or 'All' to aggregate all equipment
        model_name : str
            ML model name used (e.g., 'XGBoost', 'LightGBM', etc.)
        df_sales : DataFrame
            DataFrame with historical sales data
        df_results : DataFrame
            DataFrame with validation results
        df_future : DataFrame
            DataFrame with future predictions
        df_crm : DataFrame
            DataFrame with CRM data
        
        Returns:
        --------
        DataFrame with date index and columns:
            - Historical_Count: Real historical values
            - Validation_Actual: Real values from validation period
            - Validation_Pred: Predictions from validation period
            - Forecast_Pred: Future predictions
            - Forecast_Upper: Upper confidence interval limit
            - Forecast_Lower: Lower confidence interval limit
            - CRM_Count: CRM values
            - Period_Type: Period type (Historical/Validation/Forecast)
        """
        
        # ========================================================================
        # 1. HISTORICAL DATA
        # ========================================================================
        df_sales_sorted = df_sales.sort_values('Sell_Date').reset_index(drop=True)
        max_date = df_sales_sorted['Sell_Date'].max()
        split_date = max_date - pd.DateOffset(months=self.TEST_MONTHS)
        
        # Filter by Model_Eq. (or all if model_eq == 'All')
        if model_eq.lower() == 'all':
            df_sales_filtered = df_sales_sorted.copy()
        else:
            df_sales_filtered = df_sales_sorted[df_sales_sorted['Model_Eq.'] == model_eq].copy()
        
        # Aggregate by date
        historical_agg = df_sales_filtered.groupby('Sell_Date')['Count'].sum().reset_index()
        historical_agg.columns = ['Date', 'Historical_Count']
        
        # ========================================================================
        # 2. VALIDATION DATA
        # ========================================================================
        # Filter validation data
        if 'period_type' in df_results.columns:
            validation_data = df_results[df_results['period_type'] == 'validation'].copy()
        else:
            validation_data = df_results[df_results['Sell_Date'] > split_date].copy()
        
        # Note: model_name parameter kept for API compatibility, but no longer filtering by Model column
        # (now there's only one model per equipment)
        
        # Filter by Model_Eq. (or all if model_eq == 'All')
        if model_eq.lower() != 'all':
            validation_data = validation_data[validation_data['Model_Eq.'] == model_eq].copy()
        
        if validation_data.empty:
            print(f"Warning: No validation data found for Model_Eq. '{model_eq}' and model '{model_name}'")
            validation_agg = pd.DataFrame(columns=['Date', 'Validation_Actual', 'Validation_Pred'])
        else:
            validation_agg = validation_data.groupby('Sell_Date').agg({
                'Count': 'sum',
                'Pred': 'sum'
            }).reset_index()
            validation_agg.columns = ['Date', 'Validation_Actual', 'Validation_Pred']
        
        # ========================================================================
        # 3. FUTURE PREDICTIONS
        # ========================================================================
        if df_future is not None and not df_future.empty:
            # Note: model_name parameter kept for API compatibility, but no longer filtering by Model column
            future_data = df_future.copy()
            
            # Filter by Model_Eq. (or all if model_eq == 'All')
            if model_eq.lower() != 'all':
                future_data = future_data[future_data['Model_Eq.'] == model_eq].copy()
            
            if not future_data.empty:
                future_agg = future_data.groupby('Sell_Date')['Pred'].sum().reset_index()
                future_agg.columns = ['Date', 'Forecast_Pred']
            else:
                future_agg = pd.DataFrame(columns=['Date', 'Forecast_Pred'])
        else:
            future_agg = pd.DataFrame(columns=['Date', 'Forecast_Pred'])
        
        # ========================================================================
        # 4. CONFIDENCE INTERVAL
        # ========================================================================
        if not validation_agg.empty and not future_agg.empty:
            # Calculate validation residuals
            residuals = validation_agg['Validation_Actual'] - validation_agg['Validation_Pred']
            std_error = np.std(residuals)
            
            # Calculate confidence interval limits
            forecast_values = future_agg['Forecast_Pred'].values
            expansion_factor = np.linspace(1.5, 2.5, len(forecast_values))
            
            future_agg['Forecast_Upper'] = forecast_values + (std_error * expansion_factor)
            future_agg['Forecast_Lower'] = np.maximum(0, forecast_values - (std_error * expansion_factor))
        elif not future_agg.empty:
            future_agg['Forecast_Upper'] = np.nan
            future_agg['Forecast_Lower'] = np.nan
        
        # ========================================================================
        # 5. CRM DATA
        # ========================================================================
        if df_crm is not None and not df_crm.empty:
            # Filter by Model_Eq. (or all if model_eq == 'All')
            if model_eq.lower() == 'all':
                crm_filtered = df_crm.copy()
            else:
                crm_filtered = df_crm[df_crm['Model_Eq.'] == model_eq].copy()
            
            if not crm_filtered.empty:
                crm_agg = crm_filtered.groupby('Sell_Date')['Count'].sum().reset_index()
                crm_agg.columns = ['Date', 'CRM_Count']
            else:
                crm_agg = pd.DataFrame(columns=['Date', 'CRM_Count'])
        else:
            crm_agg = pd.DataFrame(columns=['Date', 'CRM_Count'])
        
        # ========================================================================
        # 6. COMBINE ALL DATA
        # ========================================================================
        # Start with historical data
        result_df = historical_agg.copy()
        
        # Add validation
        if not validation_agg.empty:
            result_df = result_df.merge(validation_agg, on='Date', how='outer')
        else:
            result_df['Validation_Actual'] = np.nan
            result_df['Validation_Pred'] = np.nan
        
        # Add future predictions
        if not future_agg.empty:
            result_df = result_df.merge(future_agg, on='Date', how='outer')
        else:
            result_df['Forecast_Pred'] = np.nan
            result_df['Forecast_Upper'] = np.nan
            result_df['Forecast_Lower'] = np.nan
        
        # Add CRM
        if not crm_agg.empty:
            result_df = result_df.merge(crm_agg, on='Date', how='outer')
        else:
            result_df['CRM_Count'] = np.nan
        
        # Sort by date and set as index
        result_df = result_df.sort_values('Date').set_index('Date')
        
        # Add information about the period
        result_df['Period_Type'] = 'Historical'
        if not validation_agg.empty:
            validation_dates = validation_agg['Date'].values
            result_df.loc[result_df.index.isin(validation_dates), 'Period_Type'] = 'Validation'
        if not future_agg.empty:
            future_dates = future_agg['Date'].values
            result_df.loc[result_df.index.isin(future_dates), 'Period_Type'] = 'Forecast'
        
        return result_df
    
    def _save_demand_adjustment(self, model_eq, month_key, adjustment_type, units, note, created_by):
        """Saves a demand adjustment to MySQL database."""
        engine = None
        try:
            engine = create_engine(MYSQL_CONNECTION_STRING)

            insert_query = text("""
                INSERT INTO demand_adjustment_history
                (model_eq, month_key, adjustment_type, units, note, created_by)
                VALUES (:model_eq, :month_key, :adjustment_type, :units, :note, :created_by)
            """)

            with engine.begin() as conn:
                conn.execute(insert_query, {
                    'model_eq': model_eq,
                    'month_key': month_key,
                    'adjustment_type': adjustment_type,
                    'units': units,
                    'note': note or '',
                    'created_by': created_by or 'unknown'
                })

            return {
                'success': True,
                'model_eq': model_eq,
                'month_key': month_key,
                'adjustment_type': adjustment_type,
                'units': units
            }

        except Exception as e:
            raise Exception(f"Error saving adjustment: {str(e)}")
        finally:
            if engine:
                engine.dispose()

    def _get_demand_adjustments(self, model_eq):
        """Retrieves all demand adjustments for a specific equipment model."""
        engine = None
        try:
            engine = create_engine(MYSQL_CONNECTION_STRING)

            history_query = text("""
                SELECT id, model_eq, month_key, adjustment_type, units, note, created_by, created_at
                FROM demand_adjustment_history
                WHERE model_eq = :model_eq
                ORDER BY month_key, created_at
            """)

            consolidated_query = text("""
                SELECT
                    month_key,
                    SUM(CASE WHEN adjustment_type = 'add' THEN units ELSE -units END) as total_adjustment
                FROM demand_adjustment_history
                WHERE model_eq = :model_eq
                GROUP BY month_key
            """)

            with engine.connect() as conn:
                # Get history
                history_result = conn.execute(history_query, {'model_eq': model_eq})
                adjustments = []
                for row in history_result:
                    adjustments.append({
                        'id': row.id,
                        'model_eq': row.model_eq,
                        'month_key': row.month_key,
                        'adjustment_type': row.adjustment_type,
                        'units': int(row.units) if row.units is not None else 0,
                        'note': row.note,
                        'created_by': row.created_by,
                        'created_at': row.created_at.isoformat() if row.created_at else None
                    })

                # Get consolidated
                consolidated_result = conn.execute(consolidated_query, {'model_eq': model_eq})
                consolidated = {}
                for row in consolidated_result:
                    consolidated[row.month_key] = int(row.total_adjustment) if row.total_adjustment is not None else 0

            return {
                'success': True,
                'model_eq': model_eq,
                'adjustments': adjustments,
                'consolidated': consolidated
            }

        except Exception as e:
            raise Exception(f"Error getting adjustments: {str(e)}")
        finally:
            if engine:
                engine.dispose()

    def _to_json(self, df):
        """
        Converts DataFrame to JSON using the same pattern as InventoryAnalysisModel:
        - Normaliza tipos (evita numpy.* no payload)
        - Substitui NaN por None (vira null em JSON)
        - Converte datas para string ISO sem 'Z'
        """
        df_reset = df.reset_index()

        # Converte datas para string ISO sem timezone (ex.: 2023-01-01T00:00:00)
        if 'Date' in df_reset.columns and pd.api.types.is_datetime64_any_dtype(df_reset['Date']):
            df_reset['Date'] = df_reset['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S')

        # Substitui NaN por None (para sair null em JSON)
        df_reset = df_reset.where(pd.notna(df_reset), None)

        # Serializa com pandas e volta para dict/list Python (sem tipos numpy)
        json_str = df_reset.to_json(orient='records')  # datas já estão formatadas acima
        data = json.loads(json_str)
        return data
    
    def predict(self, model_input, context):
        """Processes the forecast data operation."""
        print(f"Context : {context}")
        # token = Helpers.get_user_token(context)

        import requests
        pipa_api_key = 'rc-1bbbe5e9-c50d-469f-920b-dcfa4a86fd06'
        base_url = Requests.getRootHost()
        url = f"{base_url}access_key/token"
        headers = {
                "X-API-KEY": pipa_api_key,
                "accept": "application/json",
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()
        token = data.get("token")

        print(f"Token: {token}")
        Requests.setToken(token=token)
        
        if "op" not in model_input:
            raise ValueError(f"No 'op' provided in input: {model_input}")
        
        operation = model_input["op"]
        
        if operation == 'get-forecast-data':
            # Read parameters from model_input
            model_eq = model_input.get('model_eq', 'All')
            model_name = model_input.get('model_name', 'ensemble')
            
            # Load all necessary datasets
            df_sales, df_results, df_future, df_crm = self._load_forecast_data()
            
            # Get forecast data
            forecast_df = self._get_forecast_data_by_equipment(
                model_eq=model_eq,
                model_name=model_name,
                df_sales=df_sales,
                df_results=df_results,
                df_future=df_future,
                df_crm=df_crm
            )
            
            # Convert to JSON "limpo" (sem numpy e sem NaN)
            forecast_json = self._to_json(forecast_df)
            
            return {
                "success": True,
                "model_eq": model_eq,
                "model_name": model_name,
                "test_months": self.TEST_MONTHS,
                "data_points": len(forecast_json),
                "date_range": {
                    "start": forecast_df.index.min().isoformat(),
                    "end": forecast_df.index.max().isoformat()
                },
                "columns": list(forecast_df.columns),
                "data": forecast_json
            }

        elif operation == 'save-demand-adjustment':
            # Required parameters
            model_eq = model_input.get('model_eq')
            month_key = model_input.get('month_key')
            adjustment_type = model_input.get('adjustment_type')
            units = model_input.get('units')

            # Optional parameters
            note = model_input.get('note', '')
            created_by = model_input.get('created_by', 'unknown')

            # Validation
            if not model_eq:
                raise ValueError("'model_eq' is required")
            if not month_key:
                raise ValueError("'month_key' is required")
            if adjustment_type not in ['add', 'subtract']:
                raise ValueError("'adjustment_type' must be 'add' or 'subtract'")
            if not isinstance(units, int) or units <= 0:
                raise ValueError("'units' must be a positive integer")

            # Save to MySQL
            return self._save_demand_adjustment(
                model_eq=model_eq,
                month_key=month_key,
                adjustment_type=adjustment_type,
                units=units,
                note=note,
                created_by=created_by
            )

        elif operation == 'get-demand-adjustments':
            model_eq = model_input.get('model_eq')

            if not model_eq:
                raise ValueError("'model_eq' is required")

            return self._get_demand_adjustments(model_eq=model_eq)

        raise ValueError(f"Invalid operation '{operation}'. Valid operations: 'get-forecast-data', 'save-demand-adjustment', 'get-demand-adjustments'")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Initialize model
model_obj = ForecastDataModel()
model_obj.load({})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Test operation 1: Specific equipment
print("="*80)
print("TESTING OPERATION: get-forecast-data (Specific Equipment)")
print("="*80)
result_315 = model_obj.predict({
    "op": "get-forecast-data",
    "model_eq": "315",
    "model_name": "ensemble"
}, context)

print(f"Success: {result_315['success']}")
print(f"Model Equipment: {result_315['model_eq']}")
print(f"ML Model: {result_315['model_name']}")
print(f"Test Months: {result_315['test_months']}")
print(f"Data Points: {result_315['data_points']:,}")
print(f"Date Range: {result_315['date_range']['start']} to {result_315['date_range']['end']}")
print(f"Columns: {result_315['columns']}")
print(f"\nFirst 3 data points:")
for i, point in enumerate(result_315['data'][:3], 1):
    print(f"  {i}. {point}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Test operation 2: All equipment aggregated
print("\n" + "="*80)
print("TESTING OPERATION: get-forecast-data (All Equipment Aggregated)")
print("="*80)
result_all = model_obj.predict({
    "op": "get-forecast-data",
    "model_eq": "All",
    "model_name": "ensemble"
}, context)

print(f"Success: {result_all['success']}")
print(f"Model Equipment: {result_all['model_eq']}")
print(f"ML Model: {result_all['model_name']}")
print(f"Test Months: {result_all['test_months']}")
print(f"Data Points: {result_all['data_points']:,}")
print(f"Date Range: {result_all['date_range']['start']} to {result_all['date_range']['end']}")
print(f"\nLast 3 data points (future forecast):")
for i, point in enumerate(result_all['data'][-3:], 1):
    print(f"  {i}. {point}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Test operation 3: Save demand adjustment
print("\n" + "="*80)
print("TESTING OPERATION: save-demand-adjustment")
print("="*80)
result_save = model_obj.predict({
    "op": "save-demand-adjustment",
    "model_eq": "315",
    "month_key": "January 2025",
    "adjustment_type": "add",
    "units": 5,
    "note": "Test adjustment from API",
    "created_by": "test_user"
}, context)

print(f"Success: {result_save['success']}")
print(f"Model Equipment: {result_save['model_eq']}")
print(f"Month: {result_save['month_key']}")
print(f"Adjustment: {result_save['adjustment_type']} {result_save['units']} units")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Test operation 4: Get demand adjustments
print("\n" + "="*80)
print("TESTING OPERATION: get-demand-adjustments")
print("="*80)
result_get = model_obj.predict({
    "op": "get-demand-adjustments",
    "model_eq": "315"
}, context)

print(f"Success: {result_get['success']}")
print(f"Model Equipment: {result_get['model_eq']}")
print(f"Total adjustments: {len(result_get['adjustments'])}")
print(f"\nConsolidated by month:")
for month, total in result_get['consolidated'].items():
    print(f"  {month}: {'+' if total > 0 else ''}{total} units")
print(f"\nAdjustment history (last 3):")
for adj in result_get['adjustments'][-3:]:
    print(f"  - {adj['month_key']}: {adj['adjustment_type']} {adj['units']} units ({adj['created_by']} at {adj['created_at']})")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Save model as a prediction service
Helpers.save_output_rc_ml_model(
    context=context, 
    model_name='forecast-data-api', 
    model_obj=ForecastDataModel, 
    artifacts={}
)

print("\n" + "="*80)
print("✅ MODEL SAVED AS: 'forecast-data-api'")
print("="*80)
print("\nAvailable operations:")
print("  • get-forecast-data: Returns forecast data for equipment")
print("  • save-demand-adjustment: Saves a demand adjustment to MySQL")
print("  • get-demand-adjustments: Retrieves demand adjustments from MySQL")
print("\nDatasets used: df_count_machine_sales, df_results_count_sales, df_future_count_sales, crm_data")
print("MySQL table: demand_adjustment_history")
print("\n--- get-forecast-data ---")
print('  {"op": "get-forecast-data", "model_eq": "315", "model_name": "ensemble"}')
print("  Parameters:")
print("    - model_eq: Equipment model ('315', 'D6K', etc.) or 'All' (default: 'All')")
print("    - model_name: ML model name ('ensemble', 'xgboost', etc.) (default: 'ensemble')")
print("\n--- save-demand-adjustment ---")
print('  {"op": "save-demand-adjustment", "model_eq": "315", "month_key": "January 2025", "adjustment_type": "add", "units": 5, "note": "Reason", "created_by": "user"}')
print("  Parameters:")
print("    - model_eq: Equipment model (required)")
print("    - month_key: Month identifier, e.g., 'January 2025' (required)")
print("    - adjustment_type: 'add' or 'subtract' (required)")
print("    - units: Positive integer (required)")
print("    - note: Optional justification")
print("    - created_by: User identifier (optional, defaults to 'unknown')")
print("\n--- get-demand-adjustments ---")
print('  {"op": "get-demand-adjustments", "model_eq": "315"}')
print("  Parameters:")
print("    - model_eq: Equipment model (required)")
print("  Returns:")
print("    - adjustments: Full history of all adjustments")
print("    - consolidated: Total adjustment per month")