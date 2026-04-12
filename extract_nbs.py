import json
import os

for fname in ['gold_official_testset_simulation.ipynb', 'silver_official_testset_simulation.ipynb']:
    print(f"Extracting {fname}...")
    try:
        with open(fname, 'r', encoding='utf-8') as f:
            j = json.load(f)
        
        output_script = fname.replace('.ipynb', '.py')
        with open(output_script, 'w', encoding='utf-8') as f:
            for cell in j.get('cells', []):
                if cell.get('cell_type') == 'code':
                    f.write("".join(cell.get('source', [])) + "\n\n")
        print(f"Extracted to {output_script}")
    except Exception as e:
        print(f"Error extracting {fname}: {e}")
