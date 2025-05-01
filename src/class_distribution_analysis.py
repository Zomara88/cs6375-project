# class_distribution_analysis.py
"""Module for visualizing class preservation statistics"""

# Import required libraries
from sklearn.feature_extraction.text import CountVectorizer  # For text feature extraction
from collections import Counter  # For counting class occurrences
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For visualization

def extract_element_classes(screen):
    """Extracts and formats UI element class names with hierarchy information
    
    Args:
        screen: Dictionary containing UI elements data
        
    Returns: 
        Space-separated string of all class names in the screen
    """
    classes = []
    for element in screen['elements']:
        # Simplify class name by removing package prefix 
        class_name = element['class'].split('.')[-1]
        
        # Include parent-child relationship if available
        if 'parent-class' in element:
            parent_class = element['parent-class'].split('.')[-1]
            classes.append(f"{parent_class}>{class_name}")  
        else:
            classes.append(class_name)
    
    return " ".join(classes)  # Combine all classes into single string

def analyze_ui_classes(raw_screens, pruned_screens):
    """Analyzes how well UI element classes are preserved during pruning
    
    Args:
        raw_screens: List of original screen dictionaries
        pruned_screens: List of pruned screen dictionaries
        
    Returns:
        DataFrame containing preservation metrics for each screen pair
    """
    # Match original and pruned screens by screen_id
    pairs = []
    for raw in raw_screens:
        pruned = next((p for p in pruned_screens if p['screen_id'] == raw['screen_id']), None)
        if pruned:
            pairs.append((
                raw['screen_id'],  
                extract_element_classes(raw),  
                extract_element_classes(pruned),  
                len(raw['elements']),  
                len(pruned['elements'])  
            ))
    
    if not pairs:
        print("⚠️ No matching screen pairs found")
        return pd.DataFrame()

    # Convert class names to numerical features
    vectorizer = CountVectorizer(binary=True, max_features=200)  # Treat classes as binary features
    all_classes = [p[1] for p in pairs] + [p[2] for p in pairs]  # Combine all class strings
    vectorizer.fit(all_classes)  # Learn vocabulary from all classes
    
    # Calculate preservation metrics for each screen pair
    results = []
    for screen_id, raw_classes, pruned_classes, raw_count, pruned_count in pairs:
        raw_set = set(raw_classes.split())  # Unique original classes
        pruned_set = set(pruned_classes.split())  # Unique preserved classes
        
        # Jaccard similarity: intersection over union of class sets
        jaccard = len(raw_set & pruned_set) / max(1, len(raw_set | pruned_set))
        
        # Class preservation ratio: percentage of classes kept
        preservation = len(pruned_set) / max(1, len(raw_set))
        
        results.append({
            'screen_id': screen_id,
            'jaccard_similarity': jaccard,  
            'class_preservation': preservation,  
            'element_reduction': 1 - (pruned_count / raw_count),  
            'raw_classes': raw_classes,  # For debugging
            'pruned_classes': pruned_classes  # For debugging
        })
    
    metrics_df = pd.DataFrame(results)
    
    # Display results summary
    print(f"\n=== UI Class Preservation (n={len(metrics_df)}) ===")
    display(metrics_df[['screen_id', 'jaccard_similarity', 'class_preservation', 'element_reduction']])
    
    # Print average metrics
    print(f"\n📊 Average Metrics:")
    print(f"• Jaccard Similarity: {metrics_df['jaccard_similarity'].mean():.2f}")
    print(f"• Class Preservation: {metrics_df['class_preservation'].mean():.2f}")
    print(f"• Element Reduction: {metrics_df['element_reduction'].mean():.2f}")
    
    # Visualize results if we have multiple screens
    if len(metrics_df) > 1:
        plt.figure(figsize=(12, 5))
        
        # Histogram of Jaccard similarity scores
        plt.subplot(1, 2, 1)
        metrics_df['jaccard_similarity'].plot(kind='hist', bins=10, alpha=0.7)
        plt.title("UI Class Similarity Distribution")
        
        # Scatter plot: reduction vs preservation
        plt.subplot(1, 2, 2)
        plt.scatter(
            metrics_df['element_reduction'], 
            metrics_df['jaccard_similarity'],
            alpha=0.6
        )
        plt.title("Reduction vs. Class Preservation")
        plt.xlabel("Element Reduction")
        plt.ylabel("Jaccard Similarity")
        
        plt.tight_layout()
        plt.show()
    
    # Show detailed examples
    print("\n🔍 Class Distribution Samples:")
    sample_size = min(2, len(metrics_df))
    for _, row in metrics_df.head(sample_size).iterrows():
        print(f"\nScreen {row['screen_id']}:")
        print(f"Original ({len(row['raw_classes'].split())} classes):")
        print("  " + "\n  ".join(row['raw_classes'].split()[:10]) + ("..." if len(row['raw_classes'].split()) > 10 else ""))
        print(f"\nPruned ({len(row['pruned_classes'].split())} classes):")
        print("  " + "\n  ".join(row['pruned_classes'].split()[:10]) + ("..." if len(row['pruned_classes'].split()) > 10 else ""))
        
        # For single screen, show top 5 class preservation details
        if len(metrics_df) == 1:
            original_counts = Counter(row['raw_classes'].split())
            pruned_counts = Counter(row['pruned_classes'].split())
            print("\nClass Preservation Details:")
            for cls in sorted(original_counts, key=original_counts.get, reverse=True)[:5]:
                preserved = "✓" if cls in pruned_counts else "✗"
                print(f"{preserved} {cls}: {pruned_counts.get(cls, 0)}/{original_counts[cls]}")
    
    return metrics_df

# Execute analysis pipeline
print("🔍 Analyzing UI class distributions...")
metrics_df = analyze_ui_classes(
    final_results['raw_screens'],  # Original screens
    final_results['pruned_screens']  # Pruned screens
)
