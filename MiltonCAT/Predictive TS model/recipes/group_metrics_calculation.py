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
# Load data from entities

# Load equipment metrics (already calculated)
df_metrics = Helpers.getEntityData(context, 'equipment_metrics_summary')

# Load SKU hierarchy
sku_hierarchy = Helpers.getEntityData(context, 'hierarchy_v4')

# Load detailed results for volume calculations
df_results = Helpers.getEntityData(context, 'df_results_count_sales')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Filter data to include only SKUs that exist in hierarchy_v4

# Get list of valid SKUs from hierarchy
valid_skus = sku_hierarchy['SKU'].unique().tolist()

print(f"Total SKUs in hierarchy_v4: {len(valid_skus)}")
print(f"Total SKUs in df_metrics (before filter): {len(df_metrics)}")
print(f"Total unique SKUs in df_results (before filter): {df_results['Model_Eq.'].nunique()}")

# Filter df_metrics to keep only SKUs in hierarchy
df_metrics = df_metrics[df_metrics['Model_Equipment'].isin(valid_skus)].copy()

# Filter df_results to keep only SKUs in hierarchy
df_results = df_results[df_results['Model_Eq.'].isin(valid_skus)].copy()

print(f"\nAfter filtering to hierarchy_v4 SKUs only:")
print(f"Total SKUs in df_metrics: {len(df_metrics)}")
print(f"Total unique SKUs in df_results: {df_results['Model_Eq.'].nunique()}")

# SKUs removed
skus_removed_metrics = set(Helpers.getEntityData(context, 'equipment_metrics_summary')['Model_Equipment']) - set(valid_skus)
if skus_removed_metrics:
    print(f"\n⚠️  SKUs removed from metrics (not in hierarchy): {len(skus_removed_metrics)}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Enrich SKU metrics with hierarchy information

def enrich_sku_metrics_with_hierarchy(metrics_df, hierarchy_df):
    """
    Add hierarchy information (Group and SubGroup) to SKU-level metrics
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        DataFrame with equipment metrics (output from calculate_metrics_by_equipment)
        Already filtered to include only SKUs in hierarchy
    hierarchy_df : pd.DataFrame
        DataFrame with hierarchy structure (Group, SubGroup, SKU)
    
    Returns:
    --------
    pd.DataFrame
        Enriched metrics with hierarchy information
    """
    # Create a copy
    enriched_df = metrics_df.copy()
    
    # Merge with hierarchy (inner join to ensure only valid SKUs)
    enriched_df = enriched_df.merge(
        hierarchy_df[['Group', 'SubGroup', 'SKU']],
        left_on='Model_Equipment',
        right_on='SKU',
        how='inner'
    )
    
    # Drop the redundant SKU column
    if 'SKU' in enriched_df.columns:
        enriched_df = enriched_df.drop(columns=['SKU'])
    
    # Reorder columns to put hierarchy first
    cols = ['Group', 'SubGroup', 'Model_Equipment', 'Model', 'MAE', 'Percentage_Error', 
            'MAPE', 'WMAPE', 'Accuracy', 'RMSE', 'Confidence', 
            'Total_Actual', 'Total_Predicted', 'N_Observations']
    
    # Keep only columns that exist
    cols = [c for c in cols if c in enriched_df.columns]
    enriched_df = enriched_df[cols]
    
    # Sort by Group, SubGroup, Equipment
    enriched_df = enriched_df.sort_values(['Group', 'SubGroup', 'Model_Equipment']).reset_index(drop=True)
    
    return enriched_df

# Execute enrichment
sku_metrics_enriched = enrich_sku_metrics_with_hierarchy(df_metrics, sku_hierarchy)

print(f"SKU metrics enriched with hierarchy")
print(f"Total SKUs with metrics: {len(sku_metrics_enriched)}")
print(f"SKUs with Group/SubGroup assigned: {sku_metrics_enriched['Group'].notna().sum()}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Calculate SubGroup-level metrics with coverage information

def calculate_subgroup_metrics(df_results, sku_metrics_enriched, hierarchy_df):
    """
    Calculate aggregated metrics at SubGroup level with coverage analysis
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        Original results data for volume calculations (already filtered to hierarchy SKUs)
    sku_metrics_enriched : pd.DataFrame
        SKU-level metrics enriched with hierarchy
    hierarchy_df : pd.DataFrame
        Complete hierarchy structure
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with SubGroup-level metrics and coverage information
    """
    # Copy validation data
    validation_data = df_results.copy()
    
    # Filter by validation period if column exists
    if 'period_type' in validation_data.columns:
        validation_data = validation_data[validation_data['period_type'] == 'validation']
    
    # Get model name from data
    model_name = validation_data['Model'].iloc[0] if len(validation_data) > 0 else 'unknown'
    
    # Get all SubGroups from hierarchy
    all_subgroups = hierarchy_df.groupby(['Group', 'SubGroup']).size().reset_index()[['Group', 'SubGroup']]
    
    # List to store results
    results = []
    
    # Process each SubGroup
    for _, row in all_subgroups.iterrows():
        group = row['Group']
        subgroup = row['SubGroup']
        
        # Get all SKUs in this SubGroup from hierarchy
        subgroup_skus = hierarchy_df[
            (hierarchy_df['Group'] == group) & 
            (hierarchy_df['SubGroup'] == subgroup)
        ]['SKU'].unique()
        
        # Get SKUs with metrics
        skus_with_metrics = sku_metrics_enriched[
            (sku_metrics_enriched['Group'] == group) & 
            (sku_metrics_enriched['SubGroup'] == subgroup)
        ]['Model_Equipment'].unique()
        
        # Calculate coverage rates
        n_skus_total = len(subgroup_skus)
        n_skus_with_metrics = len(skus_with_metrics)
        coverage_rate = (n_skus_with_metrics / n_skus_total * 100) if n_skus_total > 0 else 0
        
        # Calculate volume coverage
        # Total volume in SubGroup (from validation data)
        total_volume = validation_data[
            validation_data['Model_Eq.'].isin(subgroup_skus)
        ].groupby('Sell_Date')['Count'].sum().sum()
        
        # Volume of SKUs with metrics
        volume_with_metrics = validation_data[
            validation_data['Model_Eq.'].isin(skus_with_metrics)
        ].groupby('Sell_Date')['Count'].sum().sum()
        
        volume_coverage = (volume_with_metrics / total_volume * 100) if total_volume > 0 else 0
        
        # Skip if no SKUs with metrics
        if n_skus_with_metrics == 0:
            # Still record the SubGroup but mark as insufficient data
            results.append({
                'Group': group,
                'SubGroup': subgroup,
                'Model': model_name,
                'N_SKUs_Total': n_skus_total,
                'N_SKUs_With_Metrics': 0,
                'Coverage_Rate': 0.0,
                'Volume_Coverage': 0.0,
                'Total_Actual': total_volume,
                'Total_Predicted': np.nan,
                'MAE': np.nan,
                'Percentage_Error': np.nan,
                'MAPE': np.nan,
                'WMAPE': np.nan,
                'Accuracy': np.nan,
                'RMSE': np.nan,
                'Confidence': 'insufficient_data',
                'Metric_Confidence': 'INSUFFICIENT',
                'N_Observations': 0,
                'Warning': f'No metrics available for any SKU in this SubGroup (0/{n_skus_total})'
            })
            continue
        
        # Calculate aggregated metrics for SKUs with metrics
        subgroup_data = validation_data[validation_data['Model_Eq.'].isin(skus_with_metrics)]
        
        # Aggregate by date
        subgroup_agg = subgroup_data.groupby('Sell_Date').agg({
            'Count': 'sum',
            'Pred': 'sum'
        }).reset_index()
        
        # Get arrays
        y_true = subgroup_agg['Count'].values
        y_pred = subgroup_agg['Pred'].values
        
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
        
        # Calculate prediction confidence level based on MAPE
        if mape_for_confidence < 30:
            confidence = 'high'
        elif mape_for_confidence <= 60:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Calculate metric confidence based on coverage
        if volume_coverage >= 80:
            metric_confidence = 'HIGH'
        elif volume_coverage >= 50:
            metric_confidence = 'MEDIUM'
        elif volume_coverage >= 30:
            metric_confidence = 'LOW'
        else:
            metric_confidence = 'VERY_LOW'
        
        # Generate warning if coverage is not complete
        warning = None
        if coverage_rate < 100:
            missing_skus = n_skus_total - n_skus_with_metrics
            warning = f'{missing_skus} of {n_skus_total} SKUs missing metrics ({coverage_rate:.1f}% SKU coverage, {volume_coverage:.1f}% volume coverage)'
        
        # Store results
        results.append({
            'Group': group,
            'SubGroup': subgroup,
            'Model': model_name,
            'N_SKUs_Total': n_skus_total,
            'N_SKUs_With_Metrics': n_skus_with_metrics,
            'Coverage_Rate': coverage_rate,
            'Volume_Coverage': volume_coverage,
            'Total_Actual': total_actual,
            'Total_Predicted': total_predicted,
            'MAE': mae,
            'Percentage_Error': percentage_error,
            'MAPE': mape if not np.isnan(mape) else wmape,
            'WMAPE': wmape,
            'Accuracy': accuracy,
            'RMSE': rmse,
            'Confidence': confidence,
            'Metric_Confidence': metric_confidence,
            'N_Observations': len(y_true),
            'Warning': warning
        })
    
    # Create DataFrame
    subgroup_metrics_df = pd.DataFrame(results)
    
    # Sort by Group and SubGroup
    subgroup_metrics_df = subgroup_metrics_df.sort_values(['Group', 'SubGroup']).reset_index(drop=True)
    
    return subgroup_metrics_df

# Execute function
subgroup_metrics = calculate_subgroup_metrics(
    df_results=df_results,
    sku_metrics_enriched=sku_metrics_enriched,
    hierarchy_df=sku_hierarchy
)

# Display summary
print(f"\nSubGroup metrics calculated: {len(subgroup_metrics)}")
print(f"SubGroups with sufficient data: {(subgroup_metrics['Metric_Confidence'] != 'INSUFFICIENT').sum()}")
print(f"SubGroups with HIGH metric confidence: {(subgroup_metrics['Metric_Confidence'] == 'HIGH').sum()}")
print(f"SubGroups with warnings: {subgroup_metrics['Warning'].notna().sum()}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Calculate Group-level metrics with coverage information

def calculate_group_metrics(df_results, sku_metrics_enriched, hierarchy_df):
    """
    Calculate aggregated metrics at Group level with coverage analysis
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        Original results data for volume calculations (already filtered to hierarchy SKUs)
    sku_metrics_enriched : pd.DataFrame
        SKU-level metrics enriched with hierarchy
    hierarchy_df : pd.DataFrame
        Complete hierarchy structure
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with Group-level metrics and coverage information
    """
    # Copy validation data
    validation_data = df_results.copy()
    
    # Filter by validation period if column exists
    if 'period_type' in validation_data.columns:
        validation_data = validation_data[validation_data['period_type'] == 'validation']
    
    # Get model name from data
    model_name = validation_data['Model'].iloc[0] if len(validation_data) > 0 else 'unknown'
    
    # Get all Groups from hierarchy
    all_groups = hierarchy_df['Group'].unique()
    
    # List to store results
    results = []
    
    # Process each Group
    for group in all_groups:
        # Get all SKUs in this Group from hierarchy
        group_skus = hierarchy_df[hierarchy_df['Group'] == group]['SKU'].unique()
        
        # Get SKUs with metrics
        skus_with_metrics = sku_metrics_enriched[
            sku_metrics_enriched['Group'] == group
        ]['Model_Equipment'].unique()
        
        # Calculate coverage rates
        n_skus_total = len(group_skus)
        n_skus_with_metrics = len(skus_with_metrics)
        coverage_rate = (n_skus_with_metrics / n_skus_total * 100) if n_skus_total > 0 else 0
        
        # Get number of SubGroups
        n_subgroups_total = hierarchy_df[hierarchy_df['Group'] == group]['SubGroup'].nunique()
        n_subgroups_with_metrics = sku_metrics_enriched[
            sku_metrics_enriched['Group'] == group
        ]['SubGroup'].nunique()
        
        # Calculate volume coverage
        # Total volume in Group (from validation data)
        total_volume = validation_data[
            validation_data['Model_Eq.'].isin(group_skus)
        ].groupby('Sell_Date')['Count'].sum().sum()
        
        # Volume of SKUs with metrics
        volume_with_metrics = validation_data[
            validation_data['Model_Eq.'].isin(skus_with_metrics)
        ].groupby('Sell_Date')['Count'].sum().sum()
        
        volume_coverage = (volume_with_metrics / total_volume * 100) if total_volume > 0 else 0
        
        # Skip if no SKUs with metrics
        if n_skus_with_metrics == 0:
            results.append({
                'Group': group,
                'Model': model_name,
                'N_SubGroups_Total': n_subgroups_total,
                'N_SubGroups_With_Metrics': 0,
                'N_SKUs_Total': n_skus_total,
                'N_SKUs_With_Metrics': 0,
                'Coverage_Rate': 0.0,
                'Volume_Coverage': 0.0,
                'Total_Actual': total_volume,
                'Total_Predicted': np.nan,
                'MAE': np.nan,
                'Percentage_Error': np.nan,
                'MAPE': np.nan,
                'WMAPE': np.nan,
                'Accuracy': np.nan,
                'RMSE': np.nan,
                'Confidence': 'insufficient_data',
                'Metric_Confidence': 'INSUFFICIENT',
                'N_Observations': 0,
                'Warning': f'No metrics available for any SKU in this Group (0/{n_skus_total})'
            })
            continue
        
        # Calculate aggregated metrics for SKUs with metrics
        group_data = validation_data[validation_data['Model_Eq.'].isin(skus_with_metrics)]
        
        # Aggregate by date
        group_agg = group_data.groupby('Sell_Date').agg({
            'Count': 'sum',
            'Pred': 'sum'
        }).reset_index()
        
        # Get arrays
        y_true = group_agg['Count'].values
        y_pred = group_agg['Pred'].values
        
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
        
        # Calculate prediction confidence level based on MAPE
        if mape_for_confidence < 30:
            confidence = 'high'
        elif mape_for_confidence <= 60:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Calculate metric confidence based on coverage
        if volume_coverage >= 80:
            metric_confidence = 'HIGH'
        elif volume_coverage >= 50:
            metric_confidence = 'MEDIUM'
        elif volume_coverage >= 30:
            metric_confidence = 'LOW'
        else:
            metric_confidence = 'VERY_LOW'
        
        # Generate warning if coverage is not complete
        warning = None
        if coverage_rate < 100:
            missing_skus = n_skus_total - n_skus_with_metrics
            warning = f'{missing_skus} of {n_skus_total} SKUs missing metrics ({coverage_rate:.1f}% SKU coverage, {volume_coverage:.1f}% volume coverage)'
        
        # Store results
        results.append({
            'Group': group,
            'Model': model_name,
            'N_SubGroups_Total': n_subgroups_total,
            'N_SubGroups_With_Metrics': n_subgroups_with_metrics,
            'N_SKUs_Total': n_skus_total,
            'N_SKUs_With_Metrics': n_skus_with_metrics,
            'Coverage_Rate': coverage_rate,
            'Volume_Coverage': volume_coverage,
            'Total_Actual': total_actual,
            'Total_Predicted': total_predicted,
            'MAE': mae,
            'Percentage_Error': percentage_error,
            'MAPE': mape if not np.isnan(mape) else wmape,
            'WMAPE': wmape,
            'Accuracy': accuracy,
            'RMSE': rmse,
            'Confidence': confidence,
            'Metric_Confidence': metric_confidence,
            'N_Observations': len(y_true),
            'Warning': warning
        })
    
    # Create DataFrame
    group_metrics_df = pd.DataFrame(results)
    
    # Sort by Group
    group_metrics_df = group_metrics_df.sort_values('Group').reset_index(drop=True)
    
    return group_metrics_df

# Execute function
group_metrics = calculate_group_metrics(
    df_results=df_results,
    sku_metrics_enriched=sku_metrics_enriched,
    hierarchy_df=sku_hierarchy
)

# Display summary
print(f"\nGroup metrics calculated: {len(group_metrics)}")
print(f"Groups with sufficient data: {(group_metrics['Metric_Confidence'] != 'INSUFFICIENT').sum()}")
print(f"Groups with HIGH metric confidence: {(group_metrics['Metric_Confidence'] == 'HIGH').sum()}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Display summary of all levels

print("\n" + "="*80)
print("HIERARCHICAL METRICS SUMMARY")
print("="*80)
print("(Only SKUs present in hierarchy_v4 are included)")

print(f"\n📊 SKU Level (enriched):")
print(f"   - Total SKUs with metrics: {len(sku_metrics_enriched)}")
print(f"   - Average Accuracy: {sku_metrics_enriched['Accuracy'].mean():.2f}%")
print(f"   - High confidence: {(sku_metrics_enriched['Confidence'] == 'high').sum()} SKUs")

print(f"\n📊 SubGroup Level:")
print(f"   - Total SubGroups: {len(subgroup_metrics)}")
print(f"   - SubGroups with data: {(subgroup_metrics['Metric_Confidence'] != 'INSUFFICIENT').sum()}")
print(f"   - Average Coverage Rate: {subgroup_metrics['Coverage_Rate'].mean():.1f}%")
print(f"   - Average Volume Coverage: {subgroup_metrics['Volume_Coverage'].mean():.1f}%")
print(f"   - HIGH confidence: {(subgroup_metrics['Metric_Confidence'] == 'HIGH').sum()}")
print(f"   - MEDIUM confidence: {(subgroup_metrics['Metric_Confidence'] == 'MEDIUM').sum()}")
print(f"   - LOW confidence: {(subgroup_metrics['Metric_Confidence'] == 'LOW').sum()}")

print(f"\n📊 Group Level:")
print(f"   - Total Groups: {len(group_metrics)}")
print(f"   - Groups with data: {(group_metrics['Metric_Confidence'] != 'INSUFFICIENT').sum()}")
print(f"   - Average Coverage Rate: {group_metrics['Coverage_Rate'].mean():.1f}%")
print(f"   - Average Volume Coverage: {group_metrics['Volume_Coverage'].mean():.1f}%")
print(f"   - HIGH confidence: {(group_metrics['Metric_Confidence'] == 'HIGH').sum()}")

# Show examples
print(f"\n📋 Example - SubGroup with highest accuracy:")
best_subgroup = subgroup_metrics[subgroup_metrics['Metric_Confidence'] != 'INSUFFICIENT'].nlargest(1, 'Accuracy')
if len(best_subgroup) > 0:
    print(f"   {best_subgroup.iloc[0]['SubGroup']}: {best_subgroup.iloc[0]['Accuracy']:.2f}% accuracy")
    print(f"   Coverage: {best_subgroup.iloc[0]['Volume_Coverage']:.1f}% volume, {best_subgroup.iloc[0]['Metric_Confidence']} confidence")

print(f"\n⚠️  SubGroups with warnings: {subgroup_metrics['Warning'].notna().sum()}")
if subgroup_metrics['Warning'].notna().sum() > 0:
    print("\nExamples:")
    for _, row in subgroup_metrics[subgroup_metrics['Warning'].notna()].head(3).iterrows():
        print(f"   - {row['SubGroup']}: {row['Warning']}")

print("\n" + "="*80)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Save output datasets

# 1. SKU-level metrics (enriched with hierarchy)
Helpers.save_output_dataset(
    context=context, 
    output_name='sku_metrics_with_hierarchy', 
    data_frame=sku_metrics_enriched
)

# 2. SubGroup-level metrics
Helpers.save_output_dataset(
    context=context, 
    output_name='subgroup_metrics_with_coverage', 
    data_frame=subgroup_metrics
)

# 3. Group-level metrics
Helpers.save_output_dataset(
    context=context, 
    output_name='group_metrics_with_coverage', 
    data_frame=group_metrics
)

print("\n✅ All datasets saved successfully!")
print("   - sku_metrics_with_hierarchy")
print("   - subgroup_metrics_with_coverage")
print("   - group_metrics_with_coverage")