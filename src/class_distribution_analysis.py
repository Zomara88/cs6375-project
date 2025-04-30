# Cell 10: Final Robust UI Class Analysis
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

def extract_element_classes(screen):
    """Extract class names from UI elements with hierarchy information"""
    classes = []
    for element in screen['elements']:
        # Get simple class name (without package)
        class_name = element['class'].split('.')[-1]
        
        # Add parent class if available
        if 'parent-class' in element:
            parent_class = element['parent-class'].split('.')[-1]
            classes.append(f"{parent_class}>{class_name}")
        else:
            classes.append(class_name)
    
    return " ".join(classes)

def analyze_ui_classes(raw_screens, pruned_screens):
    """Complete UI class analysis with edge case handling"""
    # Prepare data - ensure we only compare matching screens
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

    # Vectorize class names
    vectorizer = CountVectorizer(binary=True, max_features=200)
    all_classes = [p[1] for p in pairs] + [p[2] for p in pairs]
    vectorizer.fit(all_classes)
    
    # Calculate metrics
    results = []
    for screen_id, raw_classes, pruned_classes, raw_count, pruned_count in pairs:
        raw_set = set(raw_classes.split())
        pruned_set = set(pruned_classes.split())
        
        # Similarity metrics
        jaccard = len(raw_set & pruned_set) / max(1, len(raw_set | pruned_set))
        preservation = len(pruned_set) / max(1, len(raw_set))
        
        results.append({
            'screen_id': screen_id,
            'jaccard_similarity': jaccard,
            'class_preservation': preservation,
            'element_reduction': 1 - (pruned_count / raw_count),
            'raw_classes': raw_classes,
            'pruned_classes': pruned_classes
        })
    
    metrics_df = pd.DataFrame(results)
    
    # Display results
    print(f"\n=== UI Class Preservation (n={len(metrics_df)}) ===")
    display(metrics_df[['screen_id', 'jaccard_similarity', 'class_preservation', 'element_reduction']])
    
    # Summary stats
    print(f"\n📊 Average Metrics:")
    print(f"• Jaccard Similarity: {metrics_df['jaccard_similarity'].mean():.2f}")
    print(f"• Class Preservation: {metrics_df['class_preservation'].mean():.2f}")
    print(f"• Element Reduction: {metrics_df['element_reduction'].mean():.2f}")
    
    # Visualization - only if we have enough data
    if len(metrics_df) > 1:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        metrics_df['jaccard_similarity'].plot(kind='hist', bins=10, alpha=0.7)
        plt.title("UI Class Similarity Distribution")
        
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
    
    # Show samples - handle case where n=1
    print("\n🔍 Class Distribution Samples:")
    sample_size = min(2, len(metrics_df))
    for _, row in metrics_df.head(sample_size).iterrows():
        print(f"\nScreen {row['screen_id']}:")
        print(f"Original ({len(row['raw_classes'].split())} classes):")
        print("  " + "\n  ".join(row['raw_classes'].split()[:10]) + ("..." if len(row['raw_classes'].split()) > 10 else ""))
        print(f"\nPruned ({len(row['pruned_classes'].split())} classes):")
        print("  " + "\n  ".join(row['pruned_classes'].split()[:10]) + ("..." if len(row['pruned_classes'].split()) > 10 else ""))
        
        # Show preservation of top classes
        if len(metrics_df) == 1:  # Special formatting for single sample
            original_counts = Counter(row['raw_classes'].split())
            pruned_counts = Counter(row['pruned_classes'].split())
            print("\nClass Preservation Details:")
            for cls in sorted(original_counts, key=original_counts.get, reverse=True)[:5]:
                preserved = "✓" if cls in pruned_counts else "✗"
                print(f"{preserved} {cls}: {pruned_counts.get(cls, 0)}/{original_counts[cls]}")
    
    return metrics_df

# Run analysis
print("🔍 Analyzing UI class distributions...")
metrics_df = analyze_ui_classes(
    final_results['raw_screens'],
    final_results['pruned_screens']
)
