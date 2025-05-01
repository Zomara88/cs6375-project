# analysis.py 
"""Cognitive load analysis and UI pruning pipeline"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Cell 3: Cognitive Load Analysis
def analyze_cognitive_load(screens):
    """Calculates UI cognitive load metrics including density and element clustering
    
    Args:
        screens: List of screen dictionaries containing UI elements
        
    Returns:
        DataFrame with metrics for each screen:
        - screen_id: Unique identifier
        - source: Dataset origin ('raw' or 'annotated')
        - original_elements: Total UI elements
        - density: Elements per screen area
        - noise_ratio: Percentage of unclustered elements
        - cluster_count: Number of element clusters
    """
    metrics = []  # Stores calculated metrics for all screens
    
    for screen in screens:
        try:
            # Convert element positions to numpy array for processing
            # Array shape: [n_elements, 2] where columns are x,y coordinates
            coords = np.array([[e['x'], e['y']] for e in screen['elements'])
            
            # Cluster elements using DBSCAN algorithm:
            # - eps=0.15: Maximum distance between points in same cluster
            # - min_samples=2: Minimum points to form a cluster
            clustering = DBSCAN(eps=0.15, min_samples=2).fit(coords)
            
            # Analyze clustering results
            cluster_stats = Counter(clustering.labels_)  # Count points per cluster
            noise_ratio = cluster_stats.get(-1, 0) / len(screen['elements'])  # -1 = noise points
            cluster_count = len(cluster_stats) - (1 if -1 in cluster_stats else 0)  # Exclude noise
            
            # Calculate screen space density
            bounds = np.array([e['bounds'] for e in screen['elements']])  # Get all element boundaries
            max_x, max_y = bounds[:, 2].max(), bounds[:, 3].max()  # Screen dimensions
            screen_area = max(max_x * max_y, 1)  # Prevent division by zero for empty screens
            density = len(screen['elements']) / screen_area  # Elements per unit area
            
            # Store metrics for current screen
            metrics.append({
                'screen_id': screen['screen_id'],
                'source': screen['source'],
                'original_elements': len(screen['elements']),
                'density': density,
                'noise_ratio': noise_ratio,
                'cluster_count': cluster_count
            })
            
        except Exception as e:
            print(f"Skipping screen {screen.get('screen_id', '?')}: {str(e)}")
            continue
    
    # Convert results to DataFrame for easier analysis
    return pd.DataFrame(metrics)


# Cell 4: Visualization of Cognitive Load Metrics
def visualize_load_metrics(metrics_df):
    """Creates scatter plot visualization of cognitive load metrics
    
    Args:
        metrics_df: DataFrame from analyze_cognitive_load()
    """
    plt.figure(figsize=(10, 6))
    
    # Plot each dataset type with different colors/markers
    for source, group in metrics_df.groupby('source'):
        # Create scatter plot for this dataset:
        # - x-axis: Element density
        # - y-axis: Noise ratio 
        # - alpha: Transparency for overlapping points
        # - label: For the legend
        plt.scatter(group['density'], group['noise_ratio'], 
                   alpha=0.5, label=source)
    
    # Configure plot labels and appearance
    plt.xlabel('Element Density (elements/pixel)')  # X-axis label
    plt.ylabel('Noise Ratio')                      # Y-axis label
    plt.title('Cognitive Load Metrics by Dataset')  # Plot title
    plt.legend()                                   # Show dataset legend
    plt.grid(True)                                 # Add background grid
    plt.show()                                     # Display the plot


# Cell 5: Threshold Optimization with Configurable Parameters
def optimize_pruning_thresholds(metrics_df, params=None):
    """Optimizes pruning thresholds based on cognitive load metrics
    
    Args:
        metrics_df: DataFrame containing screen metrics from analyze_cognitive_load()
        params: Optional dictionary overriding default optimization parameters:
               - density_range: (min, max) bounds for density threshold search
               - noise_range: (min, max) bounds for noise threshold search  
               - min_samples: Minimum screens needed to evaluate a threshold pair
               - reduction_weight: Importance of element reduction vs coverage [0-1]
    
    Returns:
        DataFrame with optimal threshold configuration containing:
        - density_threshold: Maximum allowed element density
        - cluster_threshold: Maximum allowed noise ratio
        - screens_retained: Number of screens meeting thresholds
        - avg_reduction: Average element reduction percentage
        - score: Optimization score (higher is better)
    """
    
    # Filter to only raw screens 
    raw_metrics = metrics_df[metrics_df['source'] == 'raw']
    if len(raw_metrics) < 10:
        raise ValueError("Insufficient raw screens (<10) for reliable optimization")
    
    # Set default parameters if none provided
    default_params = {
        'density_range': (0.02, 0.25),  # Reasonable density bounds for mobile UIs
        'noise_range': (0.2, 0.7),      # Noise ratio search range
        'min_samples': 10,              # Need at least 10 screens per threshold
        'reduction_weight': 0.7,         # 70% weight to reduction
    }
    params = params or default_params
    
    # Create threshold search grid
    density_thresholds = np.linspace(*params['density_range'], 10)  
    noise_thresholds = np.linspace(*params['noise_range'], 10)      
    
    # Evaluate all threshold combinations
    results = []
    for density in density_thresholds:
        for noise in noise_thresholds:
            # Find screens meeting both thresholds
            subset = raw_metrics[
                (raw_metrics['density'] <= density) & 
                (raw_metrics['noise_ratio'] <= noise)
            ]
            
            # Evaluate if enough screens meet criteria
            if len(subset) >= params['min_samples']:
                # Calculate element reduction 
                reduction = 1 - subset['original_elements'].mean() / raw_metrics['original_elements'].mean()
                
                # Calculate coverage 
                coverage = len(subset) / len(raw_metrics)
                
                # Combined score (weighted average)
                score = (reduction * params['reduction_weight'] +
                         coverage * (1 - params['reduction_weight']))
                
                results.append({
                    'source': 'raw',
                    'density_threshold': round(density, 4),
                    'cluster_threshold': round(noise, 4),
                    'screens_retained': len(subset),
                    'avg_reduction': round(reduction, 4),
                    'score': round(score, 4)
                })
    
    # Fallback if no thresholds meet criteria
    if not results:
        print("⚠️ No thresholds met min_samples - returning defaults")
        return pd.DataFrame([{
            'source': 'raw',
            'density_threshold': 0.05,    
            'cluster_threshold': 0.4,
            'screens_retained': len(raw_metrics),
            'avg_reduction': 0.3,         
            'score': 0.5                   
        }])
    
    # Select configuration with highest score
    best = max(results, key=lambda x: x['score'])
    return pd.DataFrame([best])  # Return as single-row DataFrame


# Cell 6: Pruning Evaluation
def evaluate_pruning_strategy(screens, metrics_df, optimal_thresholds):
    """Evaluates pruning effectiveness using optimized thresholds
    
    Args:
        screens: List of all screen dictionaries (raw and annotated)
        metrics_df: DataFrame from analyze_cognitive_load()
        optimal_thresholds: DataFrame from optimize_pruning_thresholds()
        
    Returns:
        tuple: (results_df, pruned_screens)
        - results_df: DataFrame with per-screen pruning metrics
        - pruned_screens: List of modified screen dictionaries with elements removed
        
    Raises:
        ValueError: If inputs are invalid or no screens processed
    """
    
    # Input validation
    if not isinstance(optimal_thresholds, pd.DataFrame) or len(optimal_thresholds) == 0:
        raise ValueError("Thresholds must be a non-empty DataFrame")
    
    # Filter to only raw screens (pruning targets)
    raw_screens = [s for s in screens if s['source'] == 'raw']
    if not raw_screens:
        raise ValueError("No raw screens available for pruning")
        
    # Extract optimal thresholds for raw screens
    raw_threshold = optimal_thresholds[optimal_thresholds['source'] == 'raw'].iloc[0]
    
    results = []        # Stores metrics for each screen
    pruned_screens = [] # Stores modified screen data
    
    for screen in raw_screens:
        try:
            # Calculate element centers from bounding boxes
            bounds = np.array([e['bounds'] for e in screen['elements']])
            coords = bounds[:,:2] + (bounds[:,2:] - bounds[:,:2])/2  # (xmin,ymin) + width/2
            
            # Skip screens with too few elements to cluster
            if len(coords) < 3:
                continue
                
            # Calculate current screen metrics
            clustering = DBSCAN(eps=0.15, min_samples=2).fit(coords)
            noise_ratio = list(clustering.labels_).count(-1)/len(screen['elements'])
            density = len(screen['elements'])/(bounds[:,2].max() * bounds[:,3].max())
            
            # Decision to prune based on thresholds
            should_prune = (density > raw_threshold['density_threshold']) or \
                          (noise_ratio > raw_threshold['cluster_threshold'])
            
            # Apply pruning - keep only clustered elements 
            pruned_elements = []
            if should_prune:
                pruned_elements = [e for i,e in enumerate(screen['elements']) 
                                 if clustering.labels_[i] != -1]  # Exclude noise
            
            # Only count as pruned if elements were actually removed
            was_pruned = bool(pruned_elements) and (len(pruned_elements) < len(screen['elements']))
            
            # Record results
            results.append({
                'screen_id': screen['screen_id'],
                'original_elements': len(screen['elements']),
                'pruned_elements': len(pruned_elements) if was_pruned else len(screen['elements']),
                'density': density,
                'noise_ratio': noise_ratio,
                'was_pruned': was_pruned
            })
            
            # Store modified screens if pruned
            if was_pruned:
                pruned_screens.append({
                    **screen,  
                    'elements': pruned_elements,
                    'element_count': len(pruned_elements)
                })
                
        except Exception as e:
            print(f"Skipping screen {screen.get('screen_id', '?')}: {str(e)}")
            continue
    
    # Validation
    if not results:
        raise ValueError("No screens were successfully processed")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df['reduction_pct'] = 100 * (1 - results_df['pruned_elements']/results_df['original_elements'])
    
    # Warning if no pruning occurred
    if not any(results_df['was_pruned']):
        print("⚠️ Warning: Thresholds too lenient - no screens pruned")
        print(f"Current thresholds: density={raw_threshold['density_threshold']}, noise={raw_threshold['cluster_threshold']}")
    
    # Return results and filtered screens 
    return results_df, [s for s in pruned_screens if s['element_count'] > 0]


# Cell 7: Main Pipeline (Configurable)
def run_full_pipeline(limit=300, params=None):
    """Complete pipeline with optional threshold tuning parameters
    
    Args:
        limit: Maximum number of screens to process (divided between raw/annotated)
        params: Optional parameters for threshold optimization
        
    Returns:
        Dictionary containing all pipeline results:
        - metrics: Cognitive load metrics DataFrame
        - thresholds: Optimized thresholds DataFrame
        - results: Pruning evaluation results
        - pruned_screens: List of modified screens
        - raw_screens: Original raw screens
    """
    print("=== Loading Data ===")
    annotated = load_rico_data(limit // 2, 'annotated')
    raw = load_rico_data(limit // 2, 'raw')
    all_screens = annotated + raw

    print("\n=== Analyzing Metrics ===")
    metrics_df = analyze_cognitive_load(all_screens)
    visualize_load_metrics(metrics_df)

    print("\n=== Optimizing Thresholds ===")
    thresholds = optimize_pruning_thresholds(metrics_df, params)
    display(thresholds)

    print("\n=== Evaluating Pruning ===")
    results_df, pruned_screens = evaluate_pruning_strategy(all_screens, metrics_df, thresholds)

    print("\n=== Results Summary ===")
    raw_count = len([s for s in all_screens if s['source'] == 'raw'])
    print(f"Screens processed: {len(all_screens)}")
    print(f"Raw screens pruned: {len(pruned_screens)}/{raw_count}")
    print(f"Average reduction: {results_df['reduction_pct'].mean():.1f}%")

    return {
        'metrics': metrics_df,
        'thresholds': thresholds,
        'results': results_df,
        'pruned_screens': pruned_screens,
        'raw_screens': raw
    }


# Configurable Parameters 
balanced_params = {
    'density_range': (0.03, 0.15),     # Avoid overly strict pruning
    'noise_range': (0.2, 0.5),         # Moderate tolerance to fragmentation
    'target_coverage': 0.6,
    'reduction_weight': 0.75,          # Prioritize reduction more
    'min_samples': 10,
}
