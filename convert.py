import json
from pathlib import Path

out = Path('outputs')

for c in range(3):
    with open(out / f'task2_checkpoint_chunk{c}.json', encoding='utf-8') as f:
        data = json.load(f)
    if c == 0:
        print("Task 2 checkpoint sample:")
        doc_id = list(data.keys())[0]
        print(f"doc_id: {doc_id} ({type(doc_id)})")
        para_id = list(data[doc_id].keys())[0]
        print(f"para_id: {para_id} ({type(para_id)})")

    with open(out / f'task2_predictions_chunk{c}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    
    print(f'Chunk {c}: {len(data)} documents written')

with open(out / 'task1_predictions_chunk0.json', encoding='utf-8') as f:
    t1_data = json.load(f)
    print("Task 1 prediction sample:")
    print(f"para_id: {t1_data[0]['para_id']} ({type(t1_data[0]['para_id'])})")
