# Cell 8: Visualization
def plot_pruning_impact(results_df):
    plt.figure(figsize=(12, 5))
    pruned = results_df[results_df['was_pruned']]
    retained = results_df[~results_df['was_pruned']]

    plt.scatter(pruned['density'], pruned['noise_ratio'], color='red', alpha=0.5, label='Pruned')
    plt.scatter(retained['density'], retained['noise_ratio'], color='green', alpha=0.5, label='Retained')

    plt.xlabel("Element Density")
    plt.ylabel("Noise Ratio")
    plt.title("Cognitive Load Clustering — Pruned vs. Retained Screens")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_pruning_impact(final_results['results'])

# Cell 9: Visualize Original vs. Pruned UI Layouts
import matplotlib.patches as patches

def visualize_screen_comparison(original, pruned, title_prefix=""):
    """Visualize bounding boxes of UI elements before and after pruning"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for ax, screen, title in zip(
        axes, 
        [original, pruned], 
        [f"{title_prefix}Original", f"{title_prefix}Pruned"]
    ):
        ax.set_title(title)
        ax.set_xlim(0, max(e['bounds'][2] for e in screen['elements']) + 50)
        ax.set_ylim(0, max(e['bounds'][3] for e in screen['elements']) + 50)
        ax.invert_yaxis()  # Flip to match mobile top-down layout
        ax.set_aspect('equal')
        
        for e in screen['elements']:
            xmin, ymin, xmax, ymax = e['bounds']
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.4)
            ax.add_patch(rect)
            ax.text(xmin + 2, ymin + 12, e['class'][:8], fontsize=6)
    
    plt.tight_layout()
    plt.show()

# Example: pick 1 screen from pruned results to compare
if final_results['pruned_screens']:
    sample = final_results['pruned_screens'][0]
    original = next(s for s in final_results['metrics'].to_dict('records') 
                    if s['screen_id'] == sample['screen_id'])
    original_screen = next(s for s in final_results['results'].to_dict('records') 
                           if s['screen_id'] == sample['screen_id'])

    # Match original UI from raw screens
    full_original = next(s for s in final_results['metrics'].to_dict('records')
                         if s['screen_id'] == sample['screen_id'])

    original_screen_data = next(s for s in load_rico_data(300, 'raw')
                                if s['screen_id'] == sample['screen_id'])

    visualize_screen_comparison(original=original_screen_data, pruned=sample, 
                                title_prefix=f"Screen {sample['screen_id']} - ")
else:
    print("⚠️ No pruned screens available for visualization.")
