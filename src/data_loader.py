import os
import json
from tqdm import tqdm

def load_rico_data(limit=500, dataset_type='raw'):
    """Load RICO screens with robust error handling"""
    screens = []
    
    if dataset_type == 'annotated':
        path = '/kaggle/input/annotated-rico-dataset/rico_semantics-main/data/grouping/train.json'
        try:
            with open(path) as f:
                for screen in json.load(f)[:limit]:
                    elements = []
                    for elem in screen.get('screen_elements', []):
                        try:
                            elements.append({
                                'x': (elem['xmin'] + elem['xmax']) / 2,
                                'y': (elem['ymin'] + elem['ymax']) / 2,
                                'class': elem.get('label', 'UNLABELED'),
                                'bounds': [elem['xmin'], elem['ymin'], elem['xmax'], elem['ymax']]
                            })
                        except KeyError:
                            continue
                    if elements:
                        screens.append({
                            'screen_id': screen.get('screen_id', str(len(screens))),
                            'source': 'annotated',
                            'elements': elements,
                            'element_count': len(elements)
                        })
        except Exception as e:
            print(f"Error loading annotated data: {str(e)}")
    
    else:  # Raw data
        raw_path = '/kaggle/input/rico-dataset/unique_uis/combined'
        try:
            files = [f for f in os.listdir(raw_path) if f.endswith('.json')][:limit]
            for file in tqdm(files, desc="Loading raw screens"):
                try:
                    with open(os.path.join(raw_path, file)) as f:
                        screen = json.load(f)
                        elements = []
                        if 'activity' in screen:
                            def extract_elements(node):
                                if 'bounds' in node:
                                    try:
                                        bounds = node['bounds']
                                        elements.append({
                                            'x': (bounds[0] + bounds[2]) / 2,
                                            'y': (bounds[1] + bounds[3]) / 2,
                                            'class': node.get('class', 'UNKNOWN').split('.')[-1],
                                            'bounds': bounds
                                        })
                                    except (TypeError, IndexError):
                                        pass
                                for child in node.get('children', []):
                                    extract_elements(child)
                            
                            extract_elements(screen['activity']['root'])
                            if elements:
                                screens.append({
                                    'screen_id': file.split('.')[0],
                                    'source': 'raw',
                                    'elements': elements,
                                    'element_count': len(elements)
                                })
                except Exception as e:
                    print(f"Skipping {file}: {str(e)}")
        except Exception as e:
            print(f"Error accessing raw data: {str(e)}")
    
    print(f"✅ Loaded {len(screens)} {dataset_type} screens")
    return screens
