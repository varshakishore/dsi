import pandas as pd
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# Read in the data
df_orig = pd.read_csv('results/nq320k.csv')
df_orig['Metric'] = df_orig['Metric'].apply(lambda x: x.strip())
# Filter out rows with 10 documents
df_orig = df_orig[df_orig['Documents'] >=100]

incdsi_paths = ['nq320k_results/final_mean_new_q/2023-01-25_14-27-55/', "nq320k_results/final_seed42/2023-01-26_09-57-36", "nq320k_results/final_seed43/2023-01-26_10-02-59", "nq320k_results/final_seed44/2023-01-26_10-19-42", "nq320k_results/final_seed45/2023-01-26_10-46-12", "nq320k_results/final_seed46/2023-01-26_10-47-12", 'nq320k_results/final_seed47/2023-01-26_10-03-24',  "nq320k_results/final_seed48/2023-01-26_10-20-08", "nq320k_results/final_seed49/2023-01-26_10-04-16", "nq320k_results/final_seed50/2023-01-26_10-21-12"]


# Create plots for H@1 and H@10 with New and Old documents on the same plot
# Use log scale for number of documents
# Represent old documents with solid lines and new documents with dashed lines
models = ['IncDSI', 'DPR', 'BERT_DPR_10', 'BERT_SCRATCH_10', 'IncDSI_SCRATCH']
model2color = {'IncDSI': 'tab:green', 'IncDSI_SCRATCH': 'tab:olive', 'DPR': 'tab:brown', 'BERT_DPR_10': 'tab:blue', 'BERT_SCRATCH_10': 'tab:purple'}
model2name = {'IncDSI': 'IncDSI', 'IncDSI_SCRATCH': 'IncDSI-Scratch', 'DPR': 'DPR', 'BERT_DPR_10': 'DSI-DPR', 'BERT_SCRATCH_10': 'DSI-Scratch'}

fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[1, 1], figsize=(16, 5))
for ax, metric in zip([ax1, ax2], ['H@1', 'H@10']):
    df_metric = df_orig[df_orig['Metric'] == metric]
    for partition in ['New', 'Old']:
        df_partition = df_metric[df_metric['Doc_Partition'] == partition]
        for model in models:
            if model == 'IncDSI':
                df_incdsis = []
                for path in incdsi_paths:
                    df_incdsi = pd.read_csv(os.path.join('/home/jl3353/dsi/', path, 'results.csv'))
                    df_incdsi = df_incdsi[df_incdsi['Num_Documents'] >=100]
                    df_incdsi = df_incdsi[df_incdsi['Doc_type'] ==partition.lower()]
                    df_incdsis.append(df_incdsi)

                df_incdsi_partition = pd.concat(df_incdsis)
                # groupby Num_Documents,Doc_type,Split and compute the mean
                df_incdsi_mean = df_incdsi_partition.groupby(['Num_Documents', 'Doc_type', 'Split']).mean().reset_index()
                df_incdsi_stdev = df_incdsi_partition.groupby(['Num_Documents', 'Doc_type', 'Split']).std().reset_index()
                # Plot the mean and standard deviation
                ax.plot(df_incdsi_mean['Num_Documents'], df_incdsi_mean[f'{metric}'], label=f'{model} {partition}', linestyle='--' if partition == 'New' else '-', marker='o', color=model2color[model], linewidth=3, markersize=8)
                # ax.fill_between(df_incdsi_mean['Num_Documents'], df_incdsi_mean[f'{metric}'] - df_incdsi_stdev[f'{metric}'], df_incdsi_mean[f'{metric}'] + df_incdsi_stdev[f'{metric}'], alpha=0.2, color=model2color[model])
            # if model == 'IncDSI':
            #     df_incdsi = pd.read_csv(os.path.join('/home/jl3353/dsi/', incdsi_paths[0], 'results.csv'))
            #     df_incdsi = df_incdsi[df_incdsi['Num_Documents'] >=100]
            #     df_incdsi = df_incdsi[df_incdsi['Doc_type'] ==partition.lower()]
            #     ax.plot(df_incdsi['Num_Documents'], df_incdsi[f'{metric}'], label=f'{model} {partition}', linestyle='--' if partition == 'New' else '-', marker='o', color=model2color[model], linewidth=3, markersize=8)
            elif model == 'IncDSI_SCRATCH':
                df_incdsi_scratch = pd.read_csv('/home/jl3353/dsi/nq320k_results/final_dsi/2023-03-14_20-41-48/results.csv')
                df_incdsi_scratch = df_incdsi_scratch[df_incdsi_scratch['Num_Documents'] >=100]
                df_incdsi_scratch = df_incdsi_scratch[df_incdsi_scratch['Doc_type'] ==partition.lower()]
                ax.plot(df_incdsi_scratch['Num_Documents'], df_incdsi_scratch[f'{metric}'], label=f'{model} {partition}', linestyle='--' if partition == 'New' else '-', marker='o', color=model2color[model], linewidth=3, markersize=8)
            else:
                ax.plot(df_partition['Documents'], df_partition[model], label=f'{model} {partition}', linestyle='--' if partition == 'New' else '-', marker='x', color=model2color[model])
    ax.set_title(f'Retrieval Performance ({metric})', fontsize=24)
    # modify font size of x and y labels
    ax.set_xlabel('Number of New Documents Added', fontsize=20)
    ax.set_ylabel(f'{metric}', fontsize=20)
    # modify font size of tick labels
    ax.tick_params(axis='both', which='major', labelsize=16)

    

ax2.legend(handles=[Line2D([0,1],[0,1],linestyle='-', color='k', label='Original Docs'), Line2D([0,1],[0,1],linestyle='--', color='k', label='New Docs')]+[ mpatches.Patch(color=v, label=model2name[k]) for k,v in model2color.items()], fontsize=16, loc='right', bbox_to_anchor=(1.45, 0.5))
fig.tight_layout()
fig.savefig(f'results/rebuttal.png')
fig.clf()

        
    
        
