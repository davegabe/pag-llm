import json

ipynb_file = 'analyze_pag_classification.ipynb'
with open(ipynb_file) as f:
    data = json.load(f)
    
for i in range(len(data['cells'])):
    if 'outputs' in data['cells'][i]:
        print(i)
        data['cells'][i]['outputs'] = [ o for o in data['cells'][i]['outputs'] if 'name' not in o or o['name'] != 'stderr' ]
        
with open(ipynb_file, 'w') as f:
    json.dump(data, f)
    
