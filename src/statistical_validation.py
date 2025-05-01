# statistical_validation.py
"""Evaluate screen element pruning with statistical analysis."""

# Cell 11: Statistical Evaluation
# Import required libraries
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind  # For statistical testing
from sklearn.cluster import DBSCAN  # For density-based clustering

def evaluate_pruning_with_stats(screens):
    """
    Args:
        screens (list): List of screen dictionaries containing UI elements
        
    Returns:
        dict: Contains metrics, statistical results, and pruned screens
    """
    
    # Define default optimal thresholds for pruning
    optimal_thresholds = {
        'density_threshold': 0.1,  # Elements per normalized screen area
        'cluster_threshold': 0.3   # Maximum allowed ratio of noise points
    }
    
    # Split data into control and experimental groups
    np.random.seed(42)  # For reproducibility
    shuffled = np.random.permutation(screens)  # Randomize screen order
    control, experimental = np.array_split(shuffled, 2)  # 50/50 split
    
    # Process control group (no pruning applied)
    control_results = []
    for screen in control:
        try:
            elements = screen['screen_elements']
            # Extract all element labels that exist
            labels = [e['label'] for e in elements if 'label' in e]
            if not labels:  
                continue
                
            # Record metrics for control group
            control_results.append({
                'elements': len(elements),  
                'jaccard': len(set(labels))/len(labels),  
                'unique_labels': len(set(labels))  
            })
        except Exception:
            continue  # Skip problematic screens
    
    # Process experimental group (with pruning)
    exp_results = []
    pruned_screens = []
    for screen in experimental:
        try:
            elements = screen['screen_elements']
            if len(elements) < 3:  
                continue
                
            # Calculate center points of all elements
            coords = np.array([[
                (e['xmin']+e['xmax'])/2,  
                (e['ymin']+e['ymax'])/2   
            ] for e in elements])
            
            # Calculate screen element density
            max_x = max(e['xmax'] for e in elements)
            max_y = max(e['ymax'] for e in elements)
            density = len(elements)/(max_x * max_y)  # Elements per pixel area
            
            # Cluster elements and identify noise
            clustering = DBSCAN(eps=0.15, min_samples=2).fit(coords)
            noise_ratio = list(clustering.labels_).count(-1)/len(elements)
            
            # Apply pruning if thresholds are exceeded
            if (density > optimal_thresholds['density_threshold'] or 
                noise_ratio > optimal_thresholds['cluster_threshold']):
                # Keep only non-noise elements (cluster members)
                pruned = [e for i,e in enumerate(elements) if clustering.labels_[i] != -1]
            else:
                pruned = elements  # Keep original if thresholds not exceeded
                
            # Calculate metrics for pruned screen
            labels = [e['label'] for e in elements if 'label' in e]
            pruned_labels = [e['label'] for e in pruned if 'label' in e]
            
            exp_results.append({
                'original_elements': len(elements),  # Before pruning
                'pruned_elements': len(pruned),      # After pruning
                'jaccard': len(set(pruned_labels))/max(1, len(pruned_labels)),  # Prevent div/0
                'unique_labels': len(set(pruned_labels))
            })
            # Store pruned screens with their IDs
            pruned_screens.append({'screen_id': screen['screen_id'], 'elements': pruned})
        except Exception:
            continue  # Skip problematic screens
    
    # Convert results to pandas DataFrames for analysis
    control_df = pd.DataFrame(control_results)
    exp_df = pd.DataFrame(exp_results)
    
    # Calculate percentage reduction in elements
    exp_df['reduction_pct'] = 100 * (1 - exp_df['pruned_elements']/exp_df['original_elements'])
    
    # Perform statistical comparisons between groups
    stats_results = {
        'element_reduction': {
            'mean': exp_df['reduction_pct'].mean(),
            'std': exp_df['reduction_pct'].std(),
            'effect_size': (control_df['elements'].mean() - exp_df['pruned_elements'].mean())/control_df['elements'].std()
        },
        'jaccard_similarity': {  # Measure of element uniqueness
            't_test': ttest_ind(control_df['jaccard'], exp_df['jaccard']),
            'control_mean': control_df['jaccard'].mean(),
            'exp_mean': exp_df['jaccard'].mean(),
            'effect_size': (control_df['jaccard'].mean() - exp_df['jaccard'].mean())/control_df['jaccard'].std()
        },
        'label_diversity': {  # Count of unique labels
            't_test': ttest_ind(control_df['unique_labels'], exp_df['unique_labels']),
            'control_mean': control_df['unique_labels'].mean(),
            'exp_mean': exp_df['unique_labels'].mean(),
            'effect_size': (control_df['unique_labels'].mean() - exp_df['unique_labels'].mean())/control_df['unique_labels'].std()
        }
    }
    
    return {
        'control_metrics': control_df,  # Control group metrics
        'experimental_metrics': exp_df,  # Experimental group metrics
        'pruned_screens': pruned_screens,  # Details of pruned screens
        'statistics': stats_results,  # All statistical tests
        'thresholds_used': optimal_thresholds  # Threshold values used
    }

# Example usage
with open('/kaggle/input/annotated-rico-dataset/rico_semantics-main/data/grouping/train.json') as f:
    screens = json.load(f)

# Run the evaluation
results = evaluate_pruning_with_stats(screens)

# Print comprehensive results
print("=== Experimental Results ===")
print(f"Screens processed: Control={len(results['control_metrics'])}, Experimental={len(results['experimental_metrics'])}")
print(f"Pruned screens: {len(results['pruned_screens'])}")
print(f"Thresholds used: {results['thresholds_used']}")

print("\n=== Statistical Significance ===")
jaccard_test = results['statistics']['jaccard_similarity']['t_test']
print(f"Jaccard Similarity:")
print(f"  Control: {results['statistics']['jaccard_similarity']['control_mean']:.3f} ± {results['control_metrics']['jaccard'].std():.3f}")
print(f"  Experimental: {results['statistics']['jaccard_similarity']['exp_mean']:.3f} ± {results['experimental_metrics']['jaccard'].std():.3f}")
print(f"  t = {jaccard_test.statistic:.2f}, p = {jaccard_test.pvalue:.4f}")
print(f"  Effect Size (Cohen's d): {results['statistics']['jaccard_similarity']['effect_size']:.2f}")

print("\n=== Practical Significance ===")
if jaccard_test.pvalue < 0.05:
    if abs(results['statistics']['jaccard_similarity']['effect_size']) > 0.5:
        print("Large practical significance")
    elif abs(results['statistics']['jaccard_similarity']['effect_size']) > 0.2:
        print("Moderate practical significance")
    else:
        print("Statistically significant but small effect")
else:
    print("No statistically significant difference found")
