import cv2, numpy as np, onnxruntime as ort
session = ort.InferenceSession('models/glintr100.onnx', providers=['CPUExecutionProvider'])
inp_name = session.get_inputs()[0].name
print('input shape', session.get_inputs()[0].shape)

# Two random inputs
a = np.random.rand(1, 3, 112, 112).astype(np.float32)
b = np.random.rand(1, 3, 112, 112).astype(np.float32)
emb_a = session.run(None, {inp_name: a})[0]
emb_b = session.run(None, {inp_name: b})[0]
print('random cos', float((emb_a @ emb_b.T) / (np.linalg.norm(emb_a)*np.linalg.norm(emb_b))))
print('random norms', float(np.linalg.norm(emb_a)), float(np.linalg.norm(emb_b)))

# Two real different faces from gallery
def load_and_align(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    # center crop or resize to 112x112 for a rough check
    h, w = img.shape[:2]
    img = cv2.resize(img, (112, 112))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return (rgb - 127.5) / 127.5, np.random.rand(1, 512).astype(np.float32)  # placeholder; not real face

ch = cv2.imread('DATASET/Chandler/1.jpg', cv2.IMREAD_COLOR)
mo = cv2.imread('DATASET/Monica/1.jpg', cv2.IMREAD_COLOR)
ch = cv2.resize(ch, (112, 112))
mo = cv2.resize(mo, (112, 112))
ch_rgb = (cv2.cvtColor(ch, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5
mo_rgb = (cv2.cvtColor(mo, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5
ch_t = ch_rgb.transpose(2,0,1)[np.newaxis]
mo_t = mo_rgb.transpose(2,0,1)[np.newaxis]
emb_ch = session.run(None, {inp_name: ch_t})[0]
emb_mo = session.run(None, {inp_name: mo_t})[0]
print('ch-mo cos', float((emb_ch @ emb_mo.T) / (np.linalg.norm(emb_ch)*np.linalg.norm(emb_mo))))
print('ch norm', float(np.linalg.norm(emb_ch)), 'mo norm', float(np.linalg.norm(emb_mo)))
