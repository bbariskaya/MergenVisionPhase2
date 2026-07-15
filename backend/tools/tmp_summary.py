import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RECOGNITION_PATH = REPO_ROOT / "out" / "recognition_annotations" / "recognized_detections.jsonl"
labels = Counter()
if not RECOGNITION_PATH.exists():
    print(f"not found: {RECOGNITION_PATH}")
    raise SystemExit(1)
with open(RECOGNITION_PATH) as f:
    for line in f:
        rec = json.loads(line)
        for d in rec['detections']:
            labels[d['label']] += 1
print('label counts:', dict(labels))
print('first labeled frame:')
with open(RECOGNITION_PATH) as f:
    for line in f:
        rec = json.loads(line)
        if rec['detections']:
            print('frame', rec['frame'])
            for d in rec['detections']:
                print(' ', d)
            break
