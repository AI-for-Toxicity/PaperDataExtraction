FOLDER = 'test_data/labels/scored'

# for files in folder
import json
import os
total = 0
total_any = 0
for filename in os.listdir(FOLDER):
    if filename.endswith('.json'):
        with open(os.path.join(FOLDER, filename), encoding='utf-8') as f:
            data = json.load(f)
            # count items in "unmatched_events_chunk"
            count = len(data['unmatched_events_chunk'])
            total += count
            # count items in "unmatched_events_any"
            count_any = len(data['unmatched_events_any'])
            total_any += count_any
            print(f"{filename}: {count}")
print(f"Total unmatched events (chunks): {total}")
print(f"Total unmatched events (any): {total_any}")