import json

with open('retail_analysis_part_2.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        print(''.join(cell['source']))
        print('\n---\n')
