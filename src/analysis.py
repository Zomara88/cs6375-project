def analyze_cognitive_load(screens):
    """Calculate density and clustering metrics"""
    metrics = []
    
    for screen in screens:
        try:
            # Extract element coordinates
            coords = np.array([[e['x'], e['y']] for e in screen['elements']])
            
            # Calculate clustering metrics
            clustering = DBSCAN(eps=0.15, min_samples=2).fit(coords)
            cluster_stats = Counter(clustering.labels_)
            noise_ratio = cluster_stats.get(-1, 0) / len(screen['elements'])
            cluster_count = len(cluster_stats) - (1 if -1 in cluster_stats else 0)
            
            # Calculate screen density
            bounds = np.array([e['bounds'] for e in screen['elements']])
            max_x, max_y = bounds[:, 2].max(), bounds[:, 3].max()
            screen_area = max(max_x * max_y, 1)  # Prevent division by zero
            density = len(screen['elements']) / screen_area
            
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
    
    return pd.DataFrame(metrics)

# Cell 4: Visualization
def visualize_load_metrics(metrics_df):
    """Plot density vs noise ratio"""
    plt.figure(figsize=(10, 6))
    for source, group in metrics_df.groupby('source'):
        plt.scatter(group['density'], group['noise_ratio'], 
                   alpha=0.5, label=source)
    plt.xlabel('Element Density (elements/pixel)')
    plt.ylabel('Noise Ratio')
    plt.title('Cognitive Load Metrics by Dataset')
    plt.legend()
    plt.grid(True)
    plt.show()

# Cell 5: Threshold Optimization with Configurable Parameters
def optimize_pruning_thresholds(metrics_df, params=None):
    raw_metrics = metrics_df[metrics_df['source'] == 'raw']
    if len(raw_metrics) < 10:
        raise ValueError("Insufficient raw screens")

    # Default parameters
    params = params or {
        'density_range': (0.02, 0.25),
        'noise_range': (0.2, 0.7),
        'min_samples': 10,
        'reduction_weight': 0.7,
    }

    density_thresholds = np.linspace(*params['density_range'], 10)
    noise_thresholds = np.linspace(*params['noise_range'], 10)
    
    results = []
    for density in density_thresholds:
        for noise in noise_thresholds:
            subset = raw_metrics[
                (raw_metrics['density'] <= density) &
                (raw_metrics['noise_ratio'] <= noise)
            ]
            if len(subset) >= params['min_samples']:
                reduction = 1 - subset['original_elements'].mean() / raw_metrics['original_elements'].mean()
                coverage = len(subset) / len(raw_metrics)
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

    if not results:
        return pd.DataFrame([{
            'source': 'raw',
            'density_threshold': 0.05,
            'cluster_threshold': 0.4,
            'screens_retained': len(raw_metrics),
            'avg_reduction': 0.3,
            'score': 0.5
        }])
    
    best = max(results, key=lambda x: x['score'])
    return pd.DataFrame([best])

# Cell 6: Pruning Evaluation
def evaluate_pruning_strategy(screens, metrics_df, optimal_thresholds):
    """Fixed version with proper threshold application"""
    if not isinstance(optimal_thresholds, pd.DataFrame) or len(optimal_thresholds) == 0:
        raise ValueError("Invalid thresholds provided")
    
    # Get raw screens and thresholds
    raw_screens = [s for s in screens if s['source'] == 'raw']
    if not raw_screens:
        raise ValueError("No raw screens available")
        
    raw_threshold = optimal_thresholds[optimal_thresholds['source'] == 'raw'].iloc[0]
    
    results = []
    pruned_screens = []
    
    for screen in raw_screens:
        try:
            # Calculate element centers
            bounds = np.array([e['bounds'] for e in screen['elements']])
            coords = bounds[:,:2] + (bounds[:,2:] - bounds[:,:2])/2
            
            # Skip screens with too few elements
            if len(coords) < 3:
                continue
                
            # Calculate metrics
            clustering = DBSCAN(eps=0.15, min_samples=2).fit(coords)
            noise_ratio = list(clustering.labels_).count(-1)/len(screen['elements'])
            density = len(screen['elements'])/(bounds[:,2].max() * bounds[:,3].max())
            
            # Apply pruning only if thresholds are exceeded
            should_prune = (density > raw_threshold['density_threshold']) or \
                          (noise_ratio > raw_threshold['cluster_threshold'])
            
            pruned_elements = []
            if should_prune:
                pruned_elements = [e for i,e in enumerate(screen['elements']) 
                                 if clustering.labels_[i] != -1]
            
            # Only count as pruned if elements were actually removed
            was_pruned = bool(pruned_elements) and (len(pruned_elements) < len(screen['elements']))
            
            results.append({
                'screen_id': screen['screen_id'],
                'original_elements': len(screen['elements']),
                'pruned_elements': len(pruned_elements) if was_pruned else len(screen['elements']),
                'density': density,
                'noise_ratio': noise_ratio,
                'was_pruned': was_pruned
            })
            
            if was_pruned:
                pruned_screens.append({
                    **screen,
                    'elements': pruned_elements,
                    'element_count': len(pruned_elements)
                })
                
        except Exception as e:
            print(f"Skipping screen {screen.get('screen_id', '?')}: {str(e)}")
            continue
    
    if not results:
        raise ValueError("No screens were processed")
    
    results_df = pd.DataFrame(results)
    results_df['reduction_pct'] = 100 * (1 - results_df['pruned_elements']/results_df['original_elements'])
    
    # Verify pruning actually occurred
    if not any(results_df['was_pruned']):
        print("⚠️ Warning: Thresholds too lenient - no screens pruned")
        print("Try lowering density_threshold or cluster_threshold")
    
    return results_df, [s for s in pruned_screens if s['element_count'] > 0]  # Filter empty screens

# Cell 7: Main Pipeline (Configurable)
def run_full_pipeline(limit=300, params=None):
    """Complete pipeline with optional threshold tuning parameters"""
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
        'raw_screens': raw  # <-- ADD THIS LINE
    }

# Balanced pruning config (recommended)
balanced_params = {
    'density_range': (0.03, 0.15),     # Avoid overly strict pruning
    'noise_range': (0.2, 0.5),         # Moderate tolerance to fragmentation
    'target_coverage': 0.6,
    'reduction_weight': 0.75,          # Prioritize reduction more
    'min_samples': 10,
}

# Run pipeline with config
final_results = run_full_pipeline(limit=300, params=balanced_params)
