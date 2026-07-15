# SPRINT 05 PLAN — CONDITIONAL APPROVAL WITH BINDING IMPLEMENTATION CONTRACT

Plan yön olarak onaylıdır; fakat aşağıdaki maddeler mevcut planı override eder. Bunlar uygulanmadan Sprint 05 PASS sayılmayacak.

## AMAÇ

Bu sprint yalnız şu çalışan vertical slice'ı teslim edecek:

```text
encoded video
→ NVDEC/NVMM
→ temporal frame batch
→ RetinaFace CUDA postprocess
→ landmark-aware GPU face alignment
→ batched TensorRT GlintR100
→ L2-normalized embedding
→ validated gallery top1/top2 matching
→ recognition metadata
→ GPU OSD/NVENC
→ isimli annotated MP4 + structured evidence
```

Bu sprint canonical tracking, firstSeen/lastSeen, appearance reconciliation veya persistent anonymous yaratmayacak. Bunlar bir sonraki sprinttir. Fakat Sprint 06'nın yeniden inference çalıştırmadan kullanabileceği evidence contract'ını üretmelidir.

---

## 0. SPRINT 04 KAPANIŞI

Sprint 05 source değişikliğine başlamadan önce:

- correctness-matrix aggregation bug unit test ile düzeltilecek;
- Sprint 04 split verdict yazılacak:
  - detector batching/render: PASS
  - NVIDIA tracker correctness: KNOWN_BROKEN / DEFERRED
- SPRINT-004-CODE-REVIEW-PACKAGE.md tamamlanacak;
- render için ölçülmüş çalışan buffer-pool değeri korunacak;
- kanıt olmadan pool=128 değeri max(16,batch*2)'ye düşürülmeyecek.

Sprint 04 kapanmadan CURRENT_SPRINT.md Sprint 05 completed gösterilmeyecek.

---

## 1. TEK VE SINIRLI FEASIBILITY GATE

Önce `phase2-sprint-05-feasibility` target'ını oluştur.

Tercih edilen standart DeepStream yolu:

```text
RetinaFace object metadata
→ nvdspreprocess object/SGIE mode
→ custom tensor preparation
→ nvinfer input-tensor-from-meta=1
→ recognition metadata adapter
```

Config intent:

- nvdspreprocess:
  - process-on-frame=0
  - operate-on-gie-id=<RetinaFace component ID>
  - operate-on-class-ids=0
  - process-on-all-objects=1
  - custom-tensor-preparation-function=<landmark align function>
  - CUDA device output tensor
- nvinfer:
  - input-tensor-from-meta=1
  - output-tensor-meta=1
  - classifier-async-mode=0
  - batch-size=<max face batch>

Official DeepStream 9 nvdspreprocess/nvinfer contract'ını Context7 + NVIDIA source üzerinden doğrula.

Feasibility test aynı buffer içinde şunları kanıtlamalı:

- 8 temporal frame;
- bazı framelerde 0 yüz;
- bazı framelerde 1 yüz;
- en az bir framede birden çok yüz;
- actual_face_batch doğru;
- her tensor slot doğru `NvDsFrameMeta` ve `NvDsObjectMeta` ile eşleşiyor;
- final partial batch;
- clean EOS;
- object pointer/ROI mapping kaymıyor;
- yüz sayısı frame batch sayısı sanılmıyor.

Bu gate için yalnız `friendsshort_50f.mp4` üzerinde minimal reproducer kullan.

PASS ise:
- standart nvdspreprocess + nvinfer yolunu kullan;
- ayrıca custom TensorRT engine wrapper yazma;
- yalnız custom tensor preparation ve output metadata adapter yaz.

FAIL ise:
- exact failure/mapping limitation raporla;
- tekrar tekrar config deneme;
- dedicated `gst-nvdsfacerecognizer` fallback'ine geç;
- `run_manifest.json` içinde `recognizer_path=dedicated_plugin` ve fallback reason yaz.

İki production yolunu aynı anda implement etme.

---

## 2. PREPROCESS CONTRACT ÖNCE DONDURULACAK

Kod yazmadan önce aşağıdaki contract source üzerinden kesinleştirilecek:

- model input binding: `input.1`
- model output binding: `1333`
- input shape: `[B,3,112,112]`
- output shape: `[B,512]`
- dtype'lar
- profile min/opt/max
- RGB mi BGR mı
- input pixel range
- mean/std veya `(x-127.5)/127.5`
- normalization graph içinde mi dışında mı
- ArcFace canonical 112×112 landmark template
- landmark order:
  `left_eye, right_eye, nose, left_mouth, right_mouth`
- output engine tarafından normalize mi geliyor
- engine/ONNX SHA-256

Repo manifest'inde BGR yazarken mevcut scripts/kernels RGB kullanıyorsa tahmin etme. Aynı real aligned crop'u:

- frozen CPU oracle,
- ONNX Runtime,
- TensorRT

üzerinden çalıştırarak karar ver.

Sonuç versioned bir contract dosyasına yazılmalı:

`backend/native/configs/glintr100_preprocess_contract.json`

Bu dosya en az şunları içermeli:

- schema_version
- model_sha256
- engine_sha256
- input_name
- output_name
- color_order
- normalization
- landmark_order
- canonical_template
- pixel_center_rule
- border_mode
- output_l2_normalization

Test sonucu görülmeden threshold gevşetme veya RGB/BGR seçme yasak.

---

## 3. FRAME BATCH İLE FACE BATCH AYRILACAK

Kod ve loglarda ayrı metrikler kullanılacak:

- configured_frame_batch
- actual_frame_batch
- actual_face_batch
- recognizer_enqueue_count
- zero_face_buffers
- partial_face_batches
- face_batch_chunks
- alignment_invalid_count
- recognized_count
- unknown_count

Örnek:

8 frame içinde yüz sayıları `[0, 2, 1, 0, 4, 1, 2, 0]` ise:

- frame batch = 8
- face batch = 10

GlintR100 batch'i 8 değil 10'dur.

Face count engine max batch'i aşarsa sessiz truncation yasak. Şu şekilde chunk et:

```cpp
for (size_t offset = 0; offset < faces.size(); offset += max_face_batch) {
    size_t count = std::min(max_face_batch, faces.size() - offset);
    align(count);
    infer(count);
    normalize(count);
    match(count);
}
```

0 yüz varsa TensorRT enqueue çağrılmadan buffer downstream'e aktarılmalı.

### EXPLICIT FACE-TO-SURFACE MAPPING

Custom recognizer kullanılırsa her yüz için explicit work item oluştur:

```cpp
struct FaceWorkItem {
    uint32_t source_id;
    uint64_t frame_num;
    uint64_t pts_ns;
    uint32_t surface_batch_index;

    NvDsFrameMeta* frame_meta;
    NvDsObjectMeta* object_meta;

    float landmarks_original[10];
    float affine_dst_to_src[6];
};
```

Kurallar:

- Tensor mapping için `frame_meta->batch_id` kullanılabilir.
- Temporal sıra için `batch_id` kullanılmaz.
- Raw pointer'lar GstBuffer ömrü dışında saklanmaz.
- Her work item'ın output embedding'i aynı object meta'ya geri bağlanmalı.
- Unit index == frame index varsayımı yapılmamalı.
- Object list iteration order'ı identity mapping olarak kullanılmamalı.

Original-coordinate landmarks recognition surface'e ayrı eksenlerde map edilmeli:

```cpp
x_surface = x_original * surface_width  / original_width
y_surface = y_original * surface_height / original_height
```

Letterbox uygulanıyorsa scale + offset contract'ı kullanılmalı. Squish-resize ile letterbox formülü karıştırılmamalı.

---

## 4. DEDICATED PLUGIN FALLBACK IMPLEMENTATION

Fallback seçilirse pipeline açıkça:

```text
nvdsretinaface
→ queue
→ nvvideoconvert
→ capsfilter video/x-raw(memory:NVMM),format=RGBA
→ mvfacerecognizer
→ queue
→ nvstreamdemux
→ queue
→ nvdsosd
→ nvvideoconvert NV12
→ nvv4l2h264enc
→ h264parse
→ qtmux
→ filesink
```

olmalı.

Element adı NVIDIA official gibi görünmemeli:

- source folder: `gst-mvfacerecognizer`
- element: `mvfacerecognizer`

Plugin runtime'da şunları assert etmeli:

- NvBufSurface mevcut;
- `surfaceList[frame_meta->batch_id]` bounds içinde;
- color format RGBA;
- CUDA-visible device memory;
- supported pitch/layout;
- pitch hiçbir zaman `width*4` varsayılmıyor.

Unsupported surface geldiğinde CPU fallback yasak; explicit GStreamer error ver.

### CUDA WARP ALIGN KERNEL CONTRACT

Kernel bir frame değil, farklı frame surface'lerinden gelen N yüzü desteklemeli.

Input:

- device source pointer array;
- pitch array;
- width/height array;
- face-to-surface index array;
- per-face inverse affine matrix;
- face count;
- color-order/normalization mode;
- CUDA stream.

Output: `[N,3,112,112]` contiguous NCHW tensor.

Kernel davranışı:

- destination pixel → source coordinate inverse warp;
- bilinear interpolation;
- explicit pixel-center convention;
- constant-zero border;
- RGB/BGR frozen contract;
- correct normalization;
- no CPU image access;
- no per-face CUDA stream synchronize.

Grid'in face dimension'ı açık olmalı; bir frame pointer'ının bütün yüzlerde kullanılması yasak.

Testler:

- non-default pitch;
- two faces from same surface;
- faces from different frame batch slots;
- out-of-bounds landmark;
- border face;
- invalid/NaN landmark;
- different original/surface aspect ratios.

### TENSORRT ENGINE WRAPPER — YALNIZ FALLBACK'TE

Custom wrapper gerekiyorsa:

- TensorRT runtime/engine/context plugin start'ta bir kez oluşturulur;
- CUDA stream bir kez oluşturulur;
- max-face-batch input/output buffers bir kez allocate edilir;
- buffer başına cudaMalloc/cudaFree yasak;
- tensor names string ile doğrulanır;
- `setInputShape()` ile actual chunk batch verilir;
- `setTensorAddress()` + `enqueueV3()` kullanılır;
- profile min/max bounds doğrulanır;
- 0 batch enqueue edilmez;
- output order yalnız `list(out.values())[0]` gibi varsayımla alınmaz;
- teardown EOS/finalize'ta exception-safe yapılır.

Tek face için bir enqueue değil, face chunk başına bir enqueue yapılmalı.

### L2 NORMALIZATION

L2 kernel:

- embedding başına 512 dimension;
- NaN/Inf kontrolü;
- gerçek zero norm kontrolü;
- zero norm embedding identity matching'e verilmez;
- status `embedding_invalid` olur;
- sonuç norm tolerance testinden geçer.

Kernel'in bir thread ile bütün 512 float'ı seri dolaşması kabul edilmeden benchmark edilmeli. Gerekirse block reduction kullan.

### COMPACT CPU BOUNDARY

İlk correctness slice'ta aşağıdakine izin var:

- `[face_batch,512]` embedding D2H;
- compact identity metadata;
- affine matrix hesabı için 5 landmark CPU kullanımı.

Yasak:

- full decoded frame D2H;
- OpenCV/Python decode;
- OpenCV production warp;
- NumPy production inference;
- frame başına/per-face synchronize.

D2H:

- chunk başına tek contiguous async copy;
- pinned host buffer;
- bir gerekli completion event/sync;
- per-face cudaMemcpy/synchronize yok.

`run_manifest.json` toplam allowed D2H byte sayısını yazmalı.

---

## 5. GALLERY CONTRACT

Gallery JSON yalnız isim→centroid listesi olmasın. Versioned schema:

```json
{
  "schema_version": "1.0",
  "model_sha256": "...",
  "preprocess_contract_sha256": "...",
  "embedding_dim": 512,
  "identities": [
    {
      "identity_id": "friends_rachel",
      "display_name": "Rachel",
      "sample_count": 12,
      "centroid": []
    }
  ]
}
```

Load sırasında:

- schema version;
- model SHA;
- preprocess contract SHA;
- exact 512 dimension;
- finite values;
- non-empty ID/name;
- duplicate ID/name;
- sample_count > 0;
- unit norm tolerance doğrulansın.

Centroid: `L2(mean(L2(sample_embeddings)))` olarak üretilmeli.

Stale OpenCV gallery embeddings, GPU alignment parity kanıtlanmadan production gallery sayılmasın.

Matching:

- deterministic top1/top2;
- tie-break identity_id asc;
- known yalnız:
  - top1 >= threshold
  - top1 - top2 >= margin
- aksi halde unknown;
- raw cosine probability diye gösterilmesin;
- threshold/margin hardcoded olmasın.

Native plugin new_anonymous yaratmayacak. Bu business/persistence kararı sonraki Python reconciliation sprintine aittir.

---

## 6. RECOGNITION CUSTOM META

Owned custom metadata tanımla:

```cpp
struct FaceRecognitionMeta {
    char identity_id[...];
    char display_name[...];

    float top1_similarity;
    float top2_similarity;
    float margin;
    float alignment_quality;

    bool accepted;
    bool alignment_valid;
    bool embedding_valid;
};
```

- fixed-size veya açık ownership;
- copy callback;
- release callback;
- nvstreamdemux/OSD sonrasında geçerli;
- raw std::string* lifetime hatası yok;
- metadata type versioned.

Detector confidence bu struct'ın recognition similarity alanına yazılmayacak.

---

## 7. JSONL CONTRACT

Ambiguous score/confidence alanları yerine:

```json
{
  "schema_version": "2.0",
  "frame": 42,
  "pts_ns": 1400000000,
  "detections": [
    {
      "detection_id": "f42_d0",
      "track_id": null,
      "bounding_box": {},
      "landmarks": [],
      "detection_confidence": 0.99,
      "alignment_valid": true,
      "alignment_quality": 0.88,
      "identity_id": "friends_rachel",
      "display_name": "Rachel",
      "recognition_status": "known",
      "top1_similarity": 0.73,
      "top2_similarity": 0.41,
      "recognition_margin": 0.32,
      "embedding_ref": 17
    }
  ]
}
```

Tracker kapalı veya sentinel ise `"track_id": null`.

UNTRACKED_OBJECT_ID, UINT64_MAX veya 1 gerçek track gibi yazılamaz.

Sonraki reconciliation sprinti için optional internal sidecar üret:

- `embeddings.f32`
- `embedding_index.jsonl`

Index şunları bağlasın:

- embedding_ref
- frame
- PTS
- detection_id
- model SHA
- preprocess contract SHA

512 float embedding'i ana JSONL içine gömme.

---

## 8. OSD CONTRACT

OSD recognition meta'yı okuyacak:

known:
```
Rachel | sim:0.73 | det:0.99
```
unknown:
```
unknown | sim:0.38 | det:0.97
```
alignment invalid:
```
alignment_invalid | det:0.97
```

Kurallar:

- detector score ile similarity ayrı;
- gallery adı yoksa model kendi kendine isim üretmez;
- color identity_id'den deterministic türetilir;
- unknown gri;
- invalid turuncu;
- tracker off modunda sahte T... yazılmaz;
- offline mode'da queue leaky=no;
- sink sync=false;
- input/output frame sayısı aynı.

---

## 9. TRACKER GATE SPRINT 05 PASS'İ BLOKLAMAYACAK

Mevcut `phase2-sprint-05-tracked-e2e` target'ını şu şekilde değiştir:

`phase2-sprint-05-tracker-diagnostic`

Bu target:

- batch=1;
- NvDCF;
- sentinel/duplicate ID kontrolü;
- sonucu PASS veya KNOWN_BROKEN olarak raporlar.

Fakat `phase2-sprint-05-acceptance` içine dahil edilmez.

Sprint 05'in mandatory production/demo profile'ı:

- batch=8;
- tracker=off;
- recognition=on;
- render=on.

Tracker bug'ı yüzünden kullanıcı isimli videodan mahrum bırakılmayacak.

---

## 10. EKSİK TEST TARGET'LARI

Mevcut target'lara ekle:

- `phase2-sprint-05-feasibility`
- `phase2-sprint-05-surface-mapping`
- `phase2-sprint-05-face-batch`
- `phase2-sprint-05-batch-parity`
- `phase2-sprint-05-output-semantics`
- `phase2-sprint-05-determinism`
- `phase2-sprint-05-tracker-diagnostic` (non-blocking)

Mandatory acceptance chain:

```text
artifacts
→ feasibility/selected architecture evidence
→ native-build/linkcheck
→ unit
→ surface mapping
→ alignment parity
→ engine parity
→ gallery semantics
→ face-batch edge cases
→ batch 1 vs batch 8 parity
→ short E2E
→ long E2E
→ output semantics
→ determinism
→ hotpath
```

48-run benchmark matrisi çalıştırılmayacak.

---

## 11. ALIGNMENT PARITY

CPU oracle production dependency değildir.

Gate en az:

- frozen real frames;
- multiple faces;
- border case;
- non-default pitch;
- original→surface per-axis mapping;
- landmark order test;
- contact sheet;
- pixel MAE/p95/max;
- same aligned crop embedding cosine.

Threshold'lar test output'u görüldükten sonra yükseltilip düşürülmeyecek. Önceden plan/review package içine yazılacak.

---

## 12. ENGINE PARITY

Aynı exact aligned tensor:

- ONNX Runtime FP32;
- TensorRT FP16

üzerinden çalıştırılacak.

Karşılaştır:

- raw embedding finite;
- normalized embedding norm;
- ONNX/TRT cosine;
- batch 1 vs batch N;
- output name 1333;
- partial batch.

ONNX vs TRT testi OpenCV decode veya farklı alignment sonucu karşılaştırmamalı; exact same input tensor kullanılmalı.

---

## 13. GALLERY SEMANTIC GATE

Golden fixture test run başlamadan önce dondurulsun:

- en az bir expected Rachel positive;
- farklı identity negative;
- unknown negative;
- top1 threshold boundary;
- margin rejection;
- duplicate gallery ID rejection;
- wrong model SHA rejection;
- wrong preprocess contract SHA rejection.

Output incelendikten sonra golden label veya threshold değiştirmek yasak.

---

## 14. E2E SEMANTIC PASS

Sadece ffprobe yeterli değildir.

### friendsshort.mp4

- expected frame count;
- expected duration/PTS tolerance;
- clean EOS;
- no duplicate/missing processed frames;
- at least one frozen expected known identity;
- not all detections unknown;
- not all faces same identity;
- detector bbox/landmark count recognition-off baseline ile aynı;
- tracker ID null;
- playable annotated MP4;
- sampled annotated-frame contact sheet.

### Friends.mp4

- input/output frame count eşit;
- PTS monotonik;
- clean EOS;
- avg_frame_batch yaklaşık 8;
- identity distribution raporu;
- known/unknown/alignment-invalid count;
- no silent face-batch truncation;
- final partial batch;
- render output playable.

---

## 15. BATCH PARITY

Aynı video batch=1 ve batch=8:

- same frame coverage;
- same detection count;
- same detection→identity mapping;
- similarity delta frozen tolerance içinde;
- same unknown/known decision;
- same output ordering.

Frame order list order'ına değil `(frame_num, pts)` anahtarına göre karşılaştırılsın.

---

## 16. DETERMINISM

Short fixture en az 3 kez çalıştırılacak:

- same identity decisions;
- same detection ordering;
- similarities tolerance içinde;
- output JSON semantic hash aynı;
- gallery tie-break deterministic.

MP4 binary hash zorunlu değil; decoded frame count/duration ve sampled visual result aynı olmalı.

---

## 17. HOT-PATH GATE

Nsight evidence şunları göstermeli:

- NVDEC/NVMM;
- CUDA alignment;
- TensorRT GlintR100;
- CUDA L2 normalize;
- no full-frame D2H;
- no Python/OpenCV production decode;
- no per-face synchronize;
- no per-buffer cudaMalloc/cudaFree;
- allowed D2H yalnız compact embedding/metadata boyutunda.

"GPU-only" deme; doğru ifade:

"Full decoded frames remain GPU-resident; compact embeddings and metadata cross to CPU for matching/output."

---

## 18. PERFORMANCE EVIDENCE

Ayrı ölç:

- detector-only FPS;
- recognition-on frame FPS;
- faces/sec;
- actual frame batch;
- actual face batch;
- align time;
- TRT time;
- gallery match time;
- render/encode time.

Sprint 05'te keyfi performance threshold uydurma. Ancak:

- long video avg frame batch collapse etmemeli;
- pipeline video süresi kadar real-time beklememeli;
- sink sync=false olmalı;
- face count artınca enqueue count face count kadar olmamalı;
- performance report raw JSON olarak saklanmalı.

---

## 19. CONTROL PLANE

Python yalnız:

- request/path validation;
- command construction;
- process lifecycle;
- manifest/result parsing;
- structured logging

yapacak.

Python yasak:

- video decode;
- face crop;
- alignment;
- embedding inference;
- per-frame recognition loop.

CLI testleri:

- annotate --help;
- exact command construction;
- paths with spaces;
- nonzero native exit;
- timeout;
- missing output;
- malformed manifest;
- clean cancellation.

---

## 20. SPRINT 06 İÇİN EVIDENCE HAZIRLIĞI

Sprint 05 canonical tracking yapmayacak. Ama her detection için şu evidence'i bırakacak:

- frame/PTS;
- detection ID;
- bbox;
- landmark;
- detection confidence;
- alignment quality;
- embedding reference;
- top1/top2/margin;
- model/gallery/preprocess versions.

Bir sonraki sprint:

```text
detection
→ local tracklet
→ quality/top-K best shot
→ offline canonical video person
→ persistent faceId
→ firstSeen/lastSeen/appearances
→ second-pass final render
```

akışını bu evidence üzerinden kuracak.

---

## 21. PASS TANIMI

Sprint 05 PASS yalnızca:

- selected architecture tek ve belgeli;
- mandatory fast profile çalışıyor;
- both annotated MP4s oluşuyor;
- semantic identity gates geçiyor;
- batch parity geçiyor;
- deterministic;
- full-frame D2H yok;
- clean EOS;
- output frame/PTS parity geçiyor;
- tracker bug'ı fast path'e sızmıyor;
- review package tamam

ise verilebilir.

`make phase2-sprint-05-acceptance` exit 0 tek başına yeterli değildir; target yukarıdaki gerçek gate'lerin tamamını çağırmalıdır.

---

## 22. DOKÜMANTASYON

Sprint sonunda:

- CURRENT_SPRINT.md gerçek status;
- IMPLEMENTATION_DETAILS.md;
- REFERENCE_DECISION_LOG.md;
- SPRINT-005-CODE-REVIEW-PACKAGE.md;
- exact selected route;
- rejected route/reason;
- model/engine/gallery/preprocess hashes;
- raw commands/output;
- NOT_RUN/SKIPPED/BLOCKED ayrımı;
- known tracker limitation;
- next sprint recommendation

yazılacak.

No git add/commit/push.

---

## 23. SPRINT 06 EMBEDDING CONTRACT AMENDMENT

Sprint 05 recognition metadata contract must expose an internal normalized 512-D embedding (or stable emb
