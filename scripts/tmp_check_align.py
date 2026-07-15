import sys
sys.path.insert(0, 'scripts')
from generate_gallery_embeddings import _align_face, _similarity_transform
import numpy as np, cv2
from test_engines_and_annotate_phoebe import preprocess_detector, OrtSession, decode_retinaface

img = cv2.imread('DATASET/Chandler/1.jpg', cv2.IMREAD_COLOR)
h, w = img.shape[:2]
ret = OrtSession('models/retinaface_r50_dynamic.onnx')
out = ret.run(preprocess_detector(img))
boxes, scores, landms = decode_retinaface(out['loc'][0], out['conf'][0], out['landms'][0], (w,h))
lms = landms[0]
print('source landmarks', lms)
M = _similarity_transform(lms.astype(np.float32), np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
]))
print('M', M)
aligned = np.hstack([lms, np.ones((5,1))]) @ M.T
print('transformed landmarks', aligned)

aligned_img = _align_face(img, lms)
# Save for visual inspection
cv2.imwrite('out/aligned_chandler_1.jpg', cv2.cvtColor((aligned_img*127.5+127.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
