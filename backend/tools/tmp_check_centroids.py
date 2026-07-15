import json, numpy as np
m = json.load(open('artifacts/gallery/gallery_centroids.json'))
labels = list(m['identities'].keys())
vecs = np.array([m['identities'][l]['centroid'] for l in labels], dtype=np.float32)
print('labels', labels)
print('norms', np.linalg.norm(vecs, axis=1))
mat = vecs @ vecs.T
print('centroid cosine matrix')
for i, l in enumerate(labels):
    row = {labels[j]: round(float(mat[i, j]), 3) for j in range(len(labels))}
    print(l, row)
