import json
from collections import Counter
labels = Counter()
with open('out/recognition_annotations/recognized_detections.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        for d in rec['detections']:
            labels[d['label']] += 1
print('label counts:', dict(labels))
print('first labeled frame:')
for line in open('out/recognition_annotations/recognized_detections.jsonl'):
    rec = json.loads(line)
    if rec['detections']:
        print('frame', rec['frame'])
        for d in rec['detections']:
            print(' ', d)
        break
