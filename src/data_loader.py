# data_loading.py
"""Module for loading and processing RICO dataset mobile UI screens"""

# Standard Library Imports
import os       # For file path operations and environment variables
import json     # For parsing JSON-formatted UI layout data

# Third-party Imports
import numpy as np   # Numerical operations and array processing
import pandas as pd  # Data manipulation and analysis with DataFrames
from tqdm import tqdm             # Progress bars for data loading loops

# Note: These imports support the data loading workflow:
# 1. File operations (json, os)
# 2. Data processing (np, pd)
# 3. Progress tracking (tqdm)

def load_rico_data(limit=500, dataset_type='raw'):
    """Loads RICO dataset screens with comprehensive error handling
    
    Args:
        limit: Maximum number of screens to load
        dataset_type: 'raw' or 'annotated' dataset version
    
    Returns:
        List of screen dictionaries with elements and metadata
    """
    screens = []  # Stores all loaded screen data
    
    if dataset_type == 'annotated':
        # Path to annotated dataset in Kaggle
        path = '/kaggle/input/annotated-rico-dataset/rico_semantics-main/data/grouping/train.json'
        try:
            with open(path) as f:
                # Load first screens from JSON file
                for screen in json.load(f)[:limit]:
                    elements = []  # Stores UI elements for current screen
                    
                    # Process each UI element in the screen
                    for elem in screen.get('screen_elements', []):
                        try:
                            # Calculate center coordinates and store element data
                            elements.append({
                                'x': (elem['xmin'] + elem['xmax']) / 2,  # X center
                                'y': (elem['ymin'] + elem['ymax']) / 2,  # Y center
                                'class': elem.get('label', 'UNLABELED'),  # Element type
                                'bounds': [elem['xmin'], elem['ymin'], elem['xmax'], elem['ymax']]  # Coordinates
                            })
                        except KeyError:
                            continue  # Skip elements with missing coordinates
                    
                    # Only add screens with valid elements
                    if elements:
                        screens.append({
                            'screen_id': screen.get('screen_id', str(len(screens))),  # Unique ID or fallback
                            'source': 'annotated',
                            'elements': elements,
                            'element_count': len(elements)  # Total elements in screen
                        })
        except Exception as e:
            print(f"Error loading annotated data: {str(e)}")
    
    else:  # Raw data processing
        raw_path = '/kaggle/input/rico-dataset/unique_uis/combined'
        try:
            # Get first JSON files from directory
            files = [f for f in os.listdir(raw_path) if f.endswith('.json')][:limit]
            
            # Process files with progress bar
            for file in tqdm(files, desc="Loading raw screens"):
                try:
                    with open(os.path.join(raw_path, file)) as f:
                        screen = json.load(f)
                        elements = []
                        
                        if 'activity' in screen:
                            # Recursive function to extract nested UI elements
                            def extract_elements(node):
                                if 'bounds' in node:
                                    try:
                                        bounds = node['bounds']
                                        # Store element data with simplified class name
                                        elements.append({
                                            'x': (bounds[0] + bounds[2]) / 2,
                                            'y': (bounds[1] + bounds[3]) / 2,
                                            'class': node.get('class', 'UNKNOWN').split('.')[-1],  # Remove package prefix
                                            'bounds': bounds
                                        })
                                    except (TypeError, IndexError):
                                        pass  # Skip invalid coordinates
                                # Process child elements recursively
                                for child in node.get('children', []):
                                    extract_elements(child)
                            
                            # Start extraction from root node
                            extract_elements(screen['activity']['root'])
                            
                            if elements:
                                screens.append({
                                    'screen_id': file.split('.')[0],  # Use filename as ID
                                    'source': 'raw',
                                    'elements': elements,
                                    'element_count': len(elements)
                                })
                except Exception as e:
                    print(f"Skipping {file}: {str(e)}")  # Log bad files
        except Exception as e:
            print(f"Error accessing raw data: {str(e)}")
    
    # Final status message
    print(f"✅ Loaded {len(screens)} {dataset_type} screens")
    return screens
