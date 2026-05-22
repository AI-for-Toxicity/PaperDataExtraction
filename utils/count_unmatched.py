FOLDER = 'test_data/labels/scored'

# for files in folder
import json
import os
total = 0
for filename in os.listdir(FOLDER):
    if filename.endswith('.json'):
        with open(os.path.join(FOLDER, filename), encoding='utf-8') as f:
            data = json.load(f)
            # count items in "unmatched_events_chunk"
            count = len(data['unmatched_events_chunk'])
            total += count
            print(f"{filename}: {count}")
print(f"Total unmatched events: {total}")
