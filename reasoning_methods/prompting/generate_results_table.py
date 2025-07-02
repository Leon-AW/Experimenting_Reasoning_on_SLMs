#!/usr/bin/env python3
"""
Script to generate a comprehensive results table from Llama 3.2 reasoning experiments.
This script parses the result files from the 'results' folder and creates structured tables
showing accuracy across different datasets, prompting methods, and model sizes.
"""

import os
import re
import pandas as pd
from tabulate import tabulate
from collections import defaultdict

def parse_results_files(results_dir="results"):
    """Parse all result files and extract accuracy values."""
    results_data = []
    
    # Valid template names to help with parsing
    valid_templates = ['simple', 'cot', 'role', 'plan']
    
    for filename in os.listdir(results_dir):
        if filename.endswith("_total_accuracy.txt"):
            # Parse filename more carefully
            parts = filename.replace("_total_accuracy.txt", "").split("_")
            
            # Find the template by checking which part matches a valid template
            template_idx = -1
            for i, part in enumerate(parts):
                if part in valid_templates:
                    template_idx = i
                    break
            
            if template_idx == -1:
                print(f"Warning: Could not find valid template in filename {filename}, skipping.")
                continue
                
            # Everything before the template is the dataset
            dataset = "_".join(parts[:template_idx])
            template = parts[template_idx]
            
            # Model size is after the template
            model_size = parts[template_idx + 1]
            
            # Check if self-consistency was used (after model size)
            self_consistency = False
            if len(parts) > template_idx + 2 and parts[template_idx + 2].startswith("sc"):
                self_consistency = True
            
            # Read the file and extract accuracy
            with open(os.path.join(results_dir, filename), 'r') as f:
                content = f.read()
                accuracy_match = re.search(r'Final Accuracy of .+?: (\d+\.\d+)%', content)
                
                if accuracy_match:
                    accuracy = float(accuracy_match.group(1))
                    
                    results_data.append({
                        'dataset': dataset,
                        'template': template,
                        'model_size': model_size,
                        'self_consistency': self_consistency,
                        'accuracy': accuracy,
                    })
    
    return results_data

def generate_main_table(results_data):
    """Generate the main comprehensive table of results."""
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results_data)
    
    # Create a label for the column that combines template and self-consistency
    df['method'] = df.apply(
        lambda row: f"{row['template']}{'_SC' if row['self_consistency'] else ''}",
        axis=1
    )
    
    # Create the pivot table
    pivot_df = df.pivot_table(
        index=['dataset', 'model_size'],
        columns=['method'],
        values=['accuracy'],
        aggfunc='first'
    )
    
    # Flatten the multi-level column index
    pivot_df.columns = [col[1] for col in pivot_df.columns]
    
    # Reset index for better display and sort by dataset name
    table_df = pivot_df.reset_index()
    
    # Define a consistent order for datasets
    dataset_order = [
        'gsm8k', 'gsm8k_2', 'arc', 'race', 'mmlu', 'drop'
    ]
    
    # Create a custom sorting key to use our dataset_order
    def get_dataset_rank(ds):
        try:
            return dataset_order.index(ds)
        except ValueError:
            return len(dataset_order)  # Put unknown datasets at the end
    
    # Sort the table by dataset first, then by model size
    table_df['dataset_rank'] = table_df['dataset'].apply(get_dataset_rank)
    table_df = table_df.sort_values(['dataset_rank', 'model_size'])
    table_df = table_df.drop(columns=['dataset_rank'])
    
    return table_df

def generate_text_tables(results_data):
    """Generate a series of plain text tables with various analyses."""
    formatted_text = []
    
    # Title and description
    formatted_text.append("# Llama 3.2 Reasoning Experiment Results\n")
    formatted_text.append("## Results organized by dataset, model size, and prompting method\n")
    
    # Main comprehensive table
    main_df = generate_main_table(results_data)
    
    # Create a better formatted version of the main table
    pretty_df = main_df.copy()
    
    # Rename columns to be more readable
    column_mapping = {
        'simple': 'Simple',
        'simple_SC': 'Simple+SC',
        'cot': 'CoT',
        'cot_SC': 'CoT+SC',
        'role': 'Role',
        'role_SC': 'Role+SC',
        'plan': 'Plan',
        'plan_SC': 'Plan+SC',
        'dataset': 'Dataset',
        'model_size': 'Model'
    }
    
    pretty_df = pretty_df.rename(columns=column_mapping)
    
    # Format the accuracy values to one decimal place with % symbol
    for col in pretty_df.columns:
        if col not in ['Dataset', 'Model']:
            pretty_df[col] = pretty_df[col].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "-")
    
    # Ensure consistent column ordering
    desired_columns = ['Dataset', 'Model']
    method_columns = ['Simple', 'Simple+SC', 'CoT', 'CoT+SC', 'Role', 'Role+SC', 'Plan', 'Plan+SC']
    for col in method_columns:
        if col in pretty_df.columns:
            desired_columns.append(col)
    
    # Keep only columns that exist
    pretty_df = pretty_df[[col for col in desired_columns if col in pretty_df.columns]]
    
    # Convert to string table
    table_str = tabulate(pretty_df, headers='keys', tablefmt='grid', showindex=False)
    formatted_text.append(table_str)
    formatted_text.append("\n")
    
    # Best performance by dataset
    formatted_text.append("## Best Performance by Dataset\n")
    
    best_by_dataset = defaultdict(list)
    for _, row in pd.DataFrame(results_data).iterrows():
        best_by_dataset[row['dataset']].append((row['model_size'], row['template'], 
                                              row['self_consistency'], row['accuracy']))
    
    best_results = []
    
    # Define dataset order for consistent presentation
    dataset_order = ['gsm8k', 'gsm8k_2', 'arc', 'race', 'mmlu', 'drop']
    ordered_datasets = sorted(best_by_dataset.keys(), 
                             key=lambda ds: dataset_order.index(ds) if ds in dataset_order else len(dataset_order))
    
    for dataset in ordered_datasets:
        results_list = best_by_dataset[dataset]
        best_result = max(results_list, key=lambda x: x[3])
        model, template, sc, accuracy = best_result
        sc_str = "+SC" if sc else ""
        best_results.append({
            'Dataset': dataset.upper(),
            'Best Method': f"{template.capitalize()}{sc_str} ({model.upper()})",
            'Accuracy': f"{accuracy:.1f}%"
        })
    
    best_df = pd.DataFrame(best_results)
    best_table = tabulate(best_df, headers='keys', tablefmt='grid', showindex=False)
    formatted_text.append(best_table)
    formatted_text.append("\n")
    
    # Model improvement analysis
    formatted_text.append("## Model Size Impact (3B vs 1B)\n")
    
    # Filter for paired results (same dataset/template, different model)
    df = pd.DataFrame(results_data)
    df_non_sc = df[~df['self_consistency']]
    
    # Calculate average improvement
    improvements = []
    datasets = df_non_sc['dataset'].unique()
    templates = df_non_sc['template'].unique()
    
    improvement_data = []
    
    for dataset in datasets:
        for template in templates:
            df_subset = df_non_sc[(df_non_sc['dataset'] == dataset) & 
                                 (df_non_sc['template'] == template)]
            
            if len(df_subset) == 2:  # Both 1B and 3B results exist
                acc_1b = df_subset[df_subset['model_size'] == '1b']['accuracy'].values[0]
                acc_3b = df_subset[df_subset['model_size'] == '3b']['accuracy'].values[0]
                improvement = acc_3b - acc_1b
                improvements.append(improvement)
                
                improvement_data.append({
                    'Dataset': dataset,
                    'Template': template.capitalize(),
                    '1B': f"{acc_1b:.1f}%",
                    '3B': f"{acc_3b:.1f}%",
                    'Improvement': f"{improvement:.1f}pp"
                })
    
    # Sort improvement data by dataset
    improvement_df = pd.DataFrame(improvement_data)
    if not improvement_df.empty:
        improvement_df['dataset_rank'] = improvement_df['Dataset'].apply(
            lambda ds: dataset_order.index(ds) if ds in dataset_order else len(dataset_order)
        )
        improvement_df = improvement_df.sort_values(['dataset_rank', 'Template'])
        improvement_df = improvement_df.drop(columns=['dataset_rank'])
    
    # Add the improvement table
    imp_table = tabulate(improvement_df, headers='keys', tablefmt='grid', showindex=False)
    formatted_text.append(imp_table)
    
    avg_improvement = sum(improvements) / len(improvements) if improvements else 0
    formatted_text.append(f"\nAverage improvement across all templates and datasets: {avg_improvement:.1f} percentage points\n")
    
    # Self-consistency effect analysis
    formatted_text.append("## Self-Consistency Effect (1B model)\n")
    
    # Calculate average SC effect
    sc_effects = []
    df_1b = df[df['model_size'] == '1b']
    
    sc_effect_data = []
    
    for dataset in datasets:
        for template in templates:
            df_subset = df_1b[(df_1b['dataset'] == dataset) & 
                             (df_1b['template'] == template)]
            
            if len(df_subset) == 2:  # Both SC and non-SC results exist
                acc_no_sc = df_subset[~df_subset['self_consistency']]['accuracy'].values[0]
                acc_sc = df_subset[df_subset['self_consistency']]['accuracy'].values[0]
                sc_effect = acc_sc - acc_no_sc
                sc_effects.append(sc_effect)
                
                sc_effect_data.append({
                    'Dataset': dataset,
                    'Template': template.capitalize(),
                    'No SC': f"{acc_no_sc:.1f}%",
                    'With SC': f"{acc_sc:.1f}%",
                    'SC Effect': f"{sc_effect:.1f}pp"
                })
    
    # Sort SC effect data by dataset
    sc_effect_df = pd.DataFrame(sc_effect_data)
    sc_effect_df['dataset_rank'] = sc_effect_df['Dataset'].apply(
        lambda ds: dataset_order.index(ds) if ds in dataset_order else len(dataset_order)
    )
    sc_effect_df = sc_effect_df.sort_values(['dataset_rank', 'Template'])
    sc_effect_df = sc_effect_df.drop(columns=['dataset_rank'])
    
    # Add the SC effect table
    sc_table = tabulate(sc_effect_df, headers='keys', tablefmt='grid', showindex=False)
    formatted_text.append(sc_table)
    
    avg_sc_effect = sum(sc_effects) / len(sc_effects) if sc_effects else 0
    formatted_text.append(f"\nAverage self-consistency effect: {avg_sc_effect:.1f} percentage points\n")
    
    # Template effect analysis (vs simple)
    formatted_text.append("## Prompt Template Effect (vs Simple Template)\n")
    
    df_simple = df[df['template'] == 'simple'].copy()
    df_simple.rename(columns={'accuracy': 'simple_accuracy'}, inplace=True)
    df_simple = df_simple[['dataset', 'model_size', 'self_consistency', 'simple_accuracy']]
    
    df_advanced = df[df['template'] != 'simple'].copy()
    
    merged_templates = pd.merge(df_advanced, df_simple, on=['dataset', 'model_size', 'self_consistency'])
    merged_templates['gain'] = merged_templates['accuracy'] - merged_templates['simple_accuracy']
    
    avg_template_gain = merged_templates.groupby(['model_size', 'template', 'self_consistency'])['gain'].mean().reset_index()
    avg_template_gain['method'] = avg_template_gain.apply(lambda row: f"{row['template'].capitalize()}{'+SC' if row['self_consistency'] else ''}", axis=1)
    
    # Create a pivot table for template gains
    template_pivot = avg_template_gain.pivot_table(index='model_size', columns='method', values='gain')
    
    # Add the average row at the bottom
    if not template_pivot.empty:
        average_row = template_pivot.mean().to_frame().T
        average_row.index = ['Average']
        template_pivot = pd.concat([template_pivot, average_row])
        
    template_pivot = template_pivot.applymap(lambda x: f"{x:.1f}pp" if pd.notnull(x) else "-")
    
    if not template_pivot.empty:
        template_gain_table = tabulate(template_pivot, headers='keys', tablefmt='grid')
        formatted_text.append(template_gain_table)
        formatted_text.append("\n")

    # Finetuning effect analysis (vs base 1B model)
    formatted_text.append("## Finetuning Effect (vs 1B Base Model)\n")
    
    df_base = df[df['model_size'] == '1b'].copy()
    df_base.rename(columns={'accuracy': 'base_accuracy'}, inplace=True)
    df_base = df_base[['dataset', 'template', 'self_consistency', 'base_accuracy']]
    
    finetuned_models = sorted([m for m in df['model_size'].unique() if m.startswith('1b-') or m == 'star'])
    df_finetuned = df[df['model_size'].isin(finetuned_models)]
    
    merged_finetune = pd.merge(df_finetuned, df_base, on=['dataset', 'template', 'self_consistency'])
    merged_finetune['gain'] = merged_finetune['accuracy'] - merged_finetune['base_accuracy']
    
    avg_finetune_gain = merged_finetune.groupby('model_size')['gain'].mean().reset_index()
    avg_finetune_gain.rename(columns={'model_size': 'Finetuned Model', 'gain': 'Average Gain vs 1B Base'}, inplace=True)
    avg_finetune_gain['Average Gain vs 1B Base'] = avg_finetune_gain['Average Gain vs 1B Base'].apply(lambda x: f"{x:.1f}pp")

    if not avg_finetune_gain.empty:
        finetune_table = tabulate(avg_finetune_gain.sort_values(by='Finetuned Model'), headers='keys', tablefmt='grid', showindex=False)
        formatted_text.append(finetune_table)
        formatted_text.append("\n")

    # Combine all formatted text
    return "\n".join(formatted_text)

def save_results(table_str, output_dir="results"):
    """Save the formatted tables to files."""
    # Save as text file
    output_file = os.path.join(output_dir, "experiment_results_table.txt")
    with open(output_file, 'w') as f:
        f.write(table_str)
    
    print(f"Results saved to {output_file}")

def main():
    # Ensure results directory exists
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found.")
        return
    
    # Parse results files
    results_data = parse_results_files(results_dir)
    
    # Filter out gsm8k_2 results as they are not needed for the final analysis
    results_data = [row for row in results_data if row.get('dataset') != 'gsm8k_2']
    
    if not results_data:
        print("No results files found or could not parse accuracy information.")
        return
    
    # Generate text tables
    table_str = generate_text_tables(results_data)
    
    # Save results
    save_results(table_str, results_dir)
    
    # Also generate a CSV for further analysis, but filter out unnecessary columns
    df = pd.DataFrame(results_data)[['dataset', 'template', 'model_size', 'self_consistency', 'accuracy']]
    
    # Define dataset order for sorting
    dataset_order = ['gsm8k', 'gsm8k_2', 'arc', 'race', 'mmlu', 'drop']
    df['dataset_rank'] = df['dataset'].apply(
        lambda ds: dataset_order.index(ds) if ds in dataset_order else len(dataset_order)
    )
    
    # Sort by dataset, model_size, template, self_consistency
    df = df.sort_values(['dataset_rank', 'model_size', 'template', 'self_consistency'])
    df = df.drop(columns=['dataset_rank'])
    
    csv_path = os.path.join(results_dir, "experiment_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV data saved to {csv_path}")
    
    print("Results tables generated successfully.")

if __name__ == "__main__":
    main() 