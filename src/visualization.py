# visualization.py
"""Module for visualizing pruning impact and pruned vs raw comparison"""

import matplotlib.patches as patches

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

# Cell 9: Visual Comparison of Original vs Pruned UI Layouts
def visualize_screen_comparison(original, pruned, title_prefix=""):
    """Generates side-by-side comparison of UI layouts before and after pruning
    
    Args:
        original: Dictionary containing original screen elements and metadata
        pruned: Dictionary containing pruned screen elements
        title_prefix: Optional string to prepend to plot titles
    """
    # Create figure with two subplots 
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  
    
    # Plot both original and pruned versions
    for ax, screen, title in zip(
        axes,                         # The two subplot axes
        [original, pruned],           # The screen data to visualize
        [f"{title_prefix}Original",   # Left plot title
         f"{title_prefix}Pruned"]     # Right plot title
    ):
        # Set plot title and boundaries
        ax.set_title(title)
        # Set x-axis limit 
        ax.set_xlim(0, max(e['bounds'][2] for e in screen['elements']) + 50)
        # Set y-axis limit 
        ax.set_ylim(0, max(e['bounds'][3] for e in screen['elements']) + 50)
        ax.invert_yaxis()  # Match mobile coordinate system (origin at top-left)
        ax.set_aspect('equal')  # Prevent distortion of UI elements
        
        # Draw each UI element as a semi-transparent rectangle
        for e in screen['elements']:
            xmin, ymin, xmax, ymax = e['bounds']
            # Create rectangle patch for the element
            rect = patches.Rectangle(
                (xmin, ymin),         
                xmax - xmin,          
                ymax - ymin,          
                linewidth=1,          
                edgecolor='blue',     
                facecolor='lightblue', 
                alpha=0.4             
            )
            ax.add_patch(rect)
            # Add element class name label 
            ax.text(
                xmin + 2,            
                ymin + 12,           
                e['class'][:8],       
                fontsize=6            
            )
    
    # Adjust layout and display
    plt.tight_layout()  # Prevent label overlapping
    plt.show()

# Example usage with pruning results
if final_results['pruned_screens']:
    # Select first pruned screen for demonstration
    sample = final_results['pruned_screens'][0]
    
    # Find matching original screen from different result sets
    original = next(s for s in final_results['metrics'].to_dict('records') 
                    if s['screen_id'] == sample['screen_id'])
    original_screen = next(s for s in final_results['results'].to_dict('records') 
                           if s['screen_id'] == sample['screen_id'])
    
    # Get full original UI data from raw dataset
    full_original = next(s for s in final_results['metrics'].to_dict('records')
                         if s['screen_id'] == sample['screen_id'])
    original_screen_data = next(s for s in load_rico_data(300, 'raw')
                                if s['screen_id'] == sample['screen_id'])
    
    # Generate comparison visualization
    visualize_screen_comparison(
        original=original_screen_data, 
        pruned=sample, 
        title_prefix=f"Screen {sample['screen_id']} - "
    )
else:
    print("⚠️ No pruned screens available for visualization.")
