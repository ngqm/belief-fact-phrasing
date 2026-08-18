import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.style.use('fivethirtyeight')
plt.style.use('fivethirtyeight')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams["font.family"] = "serif"
import os
import numpy as np


# def plot_performance(analysis_base, visualization_base, prompt_template, records):
    
#     if records.empty:
#         print("No data found. Check your 'logs' folder structure.")
#         return
#     MODEL_ORDER = ["gemma-3-4b", "gemma-3-12b", "gemma-3-27b", "llama-3.2-3b", "llama-3.1-8b", "llama-3.3-70b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b"]
#     accuracy_df = records.groupby(["Verb", "Model", "Type"])["Correct"].mean().reset_index()
#     accuracy_df["Accuracy"] = accuracy_df["Correct"] * 100

#     print("\n--- Master Accuracy Summary ---")
#     print(accuracy_df.head())
    
#     summary_dir = os.path.join(analysis_base, prompt_template)
#     if not os.path.exists(summary_dir):
#         os.makedirs(summary_dir)
        
#     summary_path = os.path.join(summary_dir, "master_accuracy_summary.csv")
#     accuracy_df.to_csv(summary_path, index=False)
#     print(f"Saved '{summary_path}'")
    
#     unique_verbs = accuracy_df["Verb"].unique()
#     for verb in unique_verbs:
#         print(f"\nGenerating plot for verb: {verb}...")
#         verb_data = accuracy_df[accuracy_df["Verb"] == verb]
        
#         pivot_data = verb_data.pivot(index="Model", columns="Type", values="Accuracy")
#         existing_models = [m for m in MODEL_ORDER if m in pivot_data.index]
#         pivot_data = pivot_data.reindex(existing_models)
#         fig, ax = plt.subplots(figsize=(8, 6))
#         x = np.arange(len(existing_models))
#         width = 0.35
#         factual_bars = ax.barh(x + width/2, pivot_data.get('factual', 0), width, 
#                               label='factual', color='tab:blue', 
#                               hatch='//')
        
#         false_bars = ax.barh(x - width/2, pivot_data.get('false', 0), width, 
#                             label='false', color='tab:red', 
#                             hatch='\\\\')
#         ax.set_title(f"{verb.replace('_', ' ')} - {prompt_template.replace('_', ' ')}")
#         ax.set_xlabel("Accuracy (%)")
#         ax.set_ylabel("Model")
#         ax.set_xlim(0, 105)
#         ax.set_yticks(x)
#         ax.set_yticklabels(existing_models, rotation=0)
#         ax.legend(title="Query Type", loc='lower left')
#         ax.bar_label(factual_bars, fmt='%.1f', padding=3)
#         ax.bar_label(false_bars, fmt='%.1f', padding=3)
#         plt.tight_layout()
#         for spine in ax.spines.values():
#             spine.set_visible(False)
#         plt.grid(False)
        
#         vis_dir = os.path.join(visualization_base, prompt_template)
#         if not os.path.exists(vis_dir):
#             os.makedirs(vis_dir)
            
#         filename = os.path.join(vis_dir, f"performance_plot_{verb}.png")
#         plt.savefig(filename, dpi=300)
#         print(f"Saved {filename}")
#         plt.close()

def plot_performance(analysis_base, visualization_base, prompt_template, records):
    
    if records.empty:
        print("No data found. Check your 'logs' folder structure.")
        return
        
    MODEL_ORDER = ["gemma-3-4b", "gemma-3-12b", "gemma-3-27b", "llama-3.2-3b", "llama-3.1-8b", "llama-3.3-70b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b"]
    
    # Process data
    accuracy_df = records.groupby(["Verb", "Model", "Type"])["Correct"].mean().reset_index()
    accuracy_df["Accuracy"] = accuracy_df["Correct"] * 100

    print("\n--- Master Accuracy Summary ---")
    print(accuracy_df.head())
    
    summary_dir = os.path.join(analysis_base, prompt_template)
    os.makedirs(summary_dir, exist_ok=True)
        
    summary_path = os.path.join(summary_dir, "master_accuracy_summary.csv")
    accuracy_df.to_csv(summary_path, index=False)
    print(f"Saved '{summary_path}'")
    
    unique_verbs = accuracy_df["Verb"].unique()
    for verb in unique_verbs:
        print(f"\nGenerating plot for verb: {verb}...")
        verb_data = accuracy_df[accuracy_df["Verb"] == verb]
        
        pivot_data = verb_data.pivot(index="Model", columns="Type", values="Accuracy")
        existing_models = [m for m in MODEL_ORDER if m in pivot_data.index]
        pivot_data = pivot_data.reindex(existing_models)
        
        if 'factual' not in pivot_data.columns: pivot_data['factual'] = 0.0
        if 'false' not in pivot_data.columns: pivot_data['false'] = 0.0

        fig, ax = plt.subplots(figsize=(7, 4))
        
        y_positions = np.arange(len(existing_models))
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(existing_models)
        for i, model in enumerate(existing_models):
            val_factual = pivot_data.loc[model, 'factual']
            val_false = pivot_data.loc[model, 'false']
            ax.plot([val_factual, val_false], [i, i], color='black', zorder=1, linewidth=2)
            ax.plot(val_factual, i, marker='o', color='tab:blue', 
                    markeredgecolor='black', zorder=3, markersize=10)
            ax.plot(val_false, i, marker='o', color='tab:red', 
                    markeredgecolor='black',
                    zorder=3, markersize=10)
            if val_factual >= val_false:
                offset_factual, offset_false = (25, 0), (-25, 0)
            else:
                offset_factual, offset_false = (-25, 0), (25, 0)
            # annotate circles with accuracy values
            ax.annotate(f"{val_factual:.1f}", (val_factual, i), textcoords="offset points", xytext=offset_factual, ha='center', zorder=4, fontsize=14)
            ax.annotate(f"{val_false:.1f}", (val_false, i), textcoords="offset points", xytext=offset_false, ha='center', zorder=4, fontsize=14)
        ax.set_xlim(0, 105)
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.grid(False)
        ax.set_title(f"{verb.replace('_', ' ')} - {prompt_template.replace('_', ' ')}")        
        ax.set_xlabel("Accuracy (%)")
        ax.set_ylabel("Model")
        ax.set_xticks([], [])
        ax.scatter([], [], marker='o', color='tab:red', 
                    edgecolor='black', label='false',
                    zorder=3, s=100)
        ax.scatter([], [], marker='o', color='tab:blue', 
                    edgecolor='black', label='factual',
                    zorder=3, s=100)
        ax.legend(title="Query Type", loc='lower left')
        plt.tight_layout()
        vis_dir = os.path.join(visualization_base, prompt_template)
        os.makedirs(vis_dir, exist_ok=True)
        filename = os.path.join(vis_dir, f"performance_plot_{verb}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")
        plt.close()


def plot_answer_distribution(analysis_base, visualization_base, prompt_template, records):

    MODEL_ORDER = ["gemma-3-4b", "gemma-3-12b", "gemma-3-27b", "llama-3.2-3b", "llama-3.1-8b", "llama-3.3-70b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b"]

    if records.empty:
        print("No data found. Check your 'logs' folder structure.")
        return

    # count the rate of Extracted_Answer options
    answer_distribution_df = records.groupby(["Verb", "Model", "Extracted_Answer"]).size().reset_index(name="Count")

    print("\n--- Master Answer Distribution Summary ---")
    print(answer_distribution_df.head())
    summary_dir = os.path.join(analysis_base, prompt_template)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    summary_path = os.path.join(summary_dir, "master_answer_distribution_summary.csv")
    answer_distribution_df.to_csv(summary_path, index=False)
    print(f"Saved '{summary_path}'")

    unique_verbs = answer_distribution_df["Verb"].unique()
    for verb in unique_verbs:
        print(f"\nGenerating answer distribution plot for verb: {verb}...")
        verb_data = answer_distribution_df[answer_distribution_df["Verb"] == verb]
        pivot_data = verb_data.pivot(index="Model", columns="Extracted_Answer", values="Count").fillna(0)
        existing_models = [m for m in MODEL_ORDER if m in pivot_data.index]
        pivot_data = pivot_data.reindex(existing_models)

        fig, ax = plt.subplots(figsize=(8, 8))
        colors = {"(A)": "darkgreen",
                  "(B)": "forestgreen",
                  "(C)": "limegreen",
                  "": "tab:gray"}
        x = np.arange(len(existing_models))
        width = 0.25
        A_bars = ax.barh(x + width, pivot_data.get('(A)', 0), width, label='(A)', color=colors['(A)'], hatch='//')
        B_bars = ax.barh(x, pivot_data.get('(B)', 0), width, label='(B)', color=colors['(B)'], hatch='\\\\')
        C_bars = ax.barh(x - width, pivot_data.get('(C)', 0), width, label='(C)', color=colors['(C)'], hatch='oo')
        ax.set_title(f"'{verb.replace('_', ' ')}' - Answer Distribution")
        ax.set_xlabel("Count")
        ax.set_ylabel("Model")
        ax.set_yticks(x)
        ax.set_yticklabels(existing_models, rotation=0)
        ax.legend(title="Extracted Answer", loc='lower right')
        ax.bar_label(A_bars, fmt='%d', padding=3)
        ax.bar_label(B_bars, fmt='%d', padding=3)
        ax.bar_label(C_bars, fmt='%d', padding=3)
        ax.set_xlim(0, 1050)
        plt.title(f"{verb.replace('_', ' ')} - {prompt_template.replace('_', ' ')}")
        plt.ylabel("Count")
        plt.xlabel("Model")
        plt.xticks(rotation=0)
        plt.legend(title="Extracted Answer", loc='lower center')
        plt.tight_layout()
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.grid(False)
        
        vis_dir = os.path.join(visualization_base, prompt_template)
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
            
        filename = os.path.join(vis_dir, f"answer_distribution_{verb}.png")
        plt.savefig(filename, dpi=300)
        print(f"Saved {filename}")
        plt.close()


def plot_performance_am_x_confident(analysis_base, visualization_base, prompt_template, records):

    if records.empty:
        print("No data found. Check your 'logs' folder structure.")
        return
        
    MODEL_ORDER = ["gemma-3-4b", "gemma-3-12b", "gemma-3-27b", "llama-3.2-3b", "llama-3.1-8b", "llama-3.3-70b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b"]
    CONFIDENCE_ORDER = ["0", "20", "40", "60", "80", "100"]
    
    # Process data
    accuracy_df = records.groupby(["Verb", "Model", "Type"])["Correct"].mean().reset_index()
    accuracy_df["Accuracy"] = accuracy_df["Correct"] * 100

    print("\n--- Master Accuracy Summary ---")
    print(accuracy_df.head())
    
    summary_dir = os.path.join(analysis_base, prompt_template)
    os.makedirs(summary_dir, exist_ok=True)
        
    summary_path = os.path.join(summary_dir, "master_accuracy_summary.csv")
    accuracy_df.to_csv(summary_path, index=False)
    print(f"Saved '{summary_path}'")
    
    unique_verbs = [f"am_{conf}_confident" for conf in CONFIDENCE_ORDER]
    x_positions = np.arange(len(unique_verbs))

    available_confidence_verbs = set(accuracy_df["Verb"].unique()) & set(unique_verbs)
    if not available_confidence_verbs:
        print(f"  (skipping confidence plots — no am_X_confident verbs "
              f"found in '{prompt_template}')")
        return

    for model in MODEL_ORDER:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(CONFIDENCE_ORDER)
        for i, verb in enumerate(unique_verbs):
            verb_data = accuracy_df[accuracy_df["Verb"] == verb]
            pivot_data = verb_data.pivot(index="Model", columns="Type", values="Accuracy")
            existing_models = [m for m in MODEL_ORDER if m in pivot_data.index]
            pivot_data = pivot_data.reindex(existing_models)

            if 'factual' not in pivot_data.columns: pivot_data['factual'] = 0.0
            if 'false' not in pivot_data.columns: pivot_data['false'] = 0.0

            if model not in pivot_data.index:
                plt.close(fig)
                break
            val_factual = pivot_data.loc[model, 'factual']
            val_false = pivot_data.loc[model, 'false']
            ax.plot([i, i], [val_factual, val_false], color='black', zorder=1, linewidth=2)
            ax.plot(i, val_factual, marker='o', color='tab:blue', 
                    markeredgecolor='black', zorder=3, markersize=10)
            ax.plot(i, val_false, marker='o', color='tab:red', 
                    markeredgecolor='black',
                    zorder=3, markersize=10)
            if val_factual >= val_false:
                offset_factual, offset_false = (0, 12), (0, -20)
            else:
                offset_factual, offset_false = (0, -20), (0, 12)
            # annotate circles with accuracy values
            ax.annotate(f"{val_factual:.1f}", (i, val_factual), textcoords="offset points", xytext=offset_factual, ha='center', zorder=4, fontsize=12, color='tab:blue')
            ax.annotate(f"{val_false:.1f}", (i, val_false), textcoords="offset points", xytext=offset_false, ha='center', zorder=4, fontsize=12, color='tab:red')
        ax.set_ylim(0, 105)
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.grid(False)
        ax.set_title(f"Confidence levels of {model} - {prompt_template.replace('_', ' ')}")        
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("I am X% confident")
        ax.set_yticks([], [])
        ax.scatter([], [], marker='o', color='tab:red', 
                    edgecolor='black', label='false',
                    zorder=3, s=100)
        ax.scatter([], [], marker='o', color='tab:blue', 
                    edgecolor='black', label='factual',
                    zorder=3, s=100)
        ax.legend(title="Query Type")
        plt.tight_layout()
        vis_dir = os.path.join(visualization_base, prompt_template)
        os.makedirs(vis_dir, exist_ok=True)
        filename = os.path.join(vis_dir, f"confidence_plot_{model}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")
        plt.close()