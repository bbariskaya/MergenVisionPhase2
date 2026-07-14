# MergenVision Phase 2 Agent Anayasası

Bu dosya `MergenVisionPhase2` repository'sinde çalışan bütün insan ve agent'lar için kalıcı çalışma kurallarını tanımlar. Kullanıcının güncel ve açık talimatı her zaman bu dosyadan üstündür. Sprint'e özgü hedefler ve acceptance komutları `docs/implementation/CURRENT_SPRINT.md` içinde tutulur; bu dosyaya günlük görev listesi eklenmez.

## 1. Ürün amacı ve kesin kapsam

Bu repository, yüklenen video dosyaları üzerinde **offline/asenkron yüz analizi** için doğruluk ve performans zemini oluşturur.

Bu sürümde sistem:

- request ile video kabul eder;
- videoyu doğrular ve retention süresi boyunca saklar;
- frame sampling uygular;
- yüz tespiti, tracking, alignment ve recognition çalıştırır;
- kesintisiz kısa tracker parçalarını video-geneli kimliklerde birleştirir;
- `known`, `anonymous`, `new_anonymous` kararlarını mevcut görüntü davranışıyla uyumlu verir;
- sonuçları frame listesi olarak değil, video içinde görülen kişiler bazında toplar;
- her kişi için original-resolution bbox geçmişini korur;
- job/process, progress, cancellation, history ve appearance sorgularını sağlar;
- Docker Compose ile tekrar üretilebilir biçimde çalışır.

Bu sürümde **kapsam dışı**:

- RTSP, webcam ve canlı yayın ingestion;
- kullanıcı arayüzü;
- browser-side inference;
- object detection, person re-identification veya segmentation'ı ayrı ürün özelliği olarak eklemek;
- Kubernetes, distributed queue platformu veya premature microservice ayrıştırması;
- videoyu render edip üstüne bbox çizilmiş yeni video üretmek (ayrı bir doğrulama aracı olabilir, API zorunluluğu değildir).

Canlı yayın daha sonra gelecektir. Bugünkü source/job abstraction'ları gelecekte live source eklenmesini engellememeli; ancak canlı yayın kodu, endpoint'i veya config'i bu sprintlere eklenmez.

## 2. Source-of-truth sırası

Çelişki halinde aşağıdaki sıra uygulanır:

1. Kullanıcının güncel açık kararı.
2. `requirements/phase2requirements.md`.
3. Onaylanmış `architecture/` belgeleri.
4. `docs/implementation/CURRENT_SPRINT.md`.
5. Birleştirme sırasında Phase 1'in frozen image API, identity ve storage contract'ları.
6. `opensourcereferences/references.md` içindeki resmî kaynaklar.
7. Eski MergenVision repository'leri; yalnız salt-okunur lessons-learned kaynağı.
8. Blog, forum ve model hafızası.

`references.md` gereksinim kaynağı değildir. Bir upstream örneğin bir davranış göstermesi, MergenVision'ın o davranışı istemesi anlamına gelmez.

Requirement veya onaylanmış mimari değişecekse agent kendiliğinden karar vermez. Çelişkiyi, seçenekleri ve etkilerini kullanıcıya sunar; açık karar bekler.

## 3. Her görevde zorunlu başlangıç

Kod veya doküman değiştirmeden önce:

1. Repository root'unu ve `git status --short` çıktısını doğrula.
2. Bu `AGENTS.md` dosyasını tamamen oku.
3. `requirements/phase2requirements.md` dosyasını oku.
4. `docs/implementation/CURRENT_SPRINT.md` varsa oku.
5. Görevle ilgili architecture, contract, source ve testleri oku.
6. `opensourcereferences/references.md` içinden ilgili resmî kaynakları seç ve exact upstream davranışını doğrula.
7. Kullanıcının mevcut değişikliklerini koru; unrelated dosyalara dokunma.

`CURRENT_SPRINT.md` yoksa production implementation başlatılmaz. Önce cohesive bir sprint objective, deliverable, acceptance ve non-goal seti hazırlanır ve kullanıcıya sunulur. Salt-okunur keşif, annotation incelemesi ve planlama yapılabilir.

Context compaction sonrasında AGENTS, requirements, CURRENT_SPRINT, `git status` ve sprint ledger yeniden okunur. Bitmiş iş tekrar yapılmaz.

## 4. Temel domain terimleri

Terimler karıştırılmaz:

- **Video job:** Bir video inputunun doğrulama, analiz, reconciliation ve finalization yaşam döngüsü.
- **Process ID:** Audit/history katmanındaki kalıcı işlem kimliği.
- **Frame:** Decode edilmiş görüntü ve onun original frame index/PTS bilgisi.
- **Detection:** Belirli bir frame'deki yüz bbox, landmark ve detector score'u.
- **Tracklet:** Tracker'ın tek bir kesintisiz sahne veya görünme aralığında ürettiği yerel iz. Sahne kesiminde sona erebilir.
- **Raw tracker ID:** NvDCF/ByteTrack gibi tracker'ın verdiği implementation-local ID. API contract'ı değildir.
- **Canonical video track ID:** Offline reconciliation sonrasında aynı gerçek kişiye ait tracklet'lerin bağlandığı, video boyunca sabit final `trackId`.
- **faceId:** Kalıcı recognition identity ID. Aynı kişi farklı videolarda aynı `faceId` ile dönebilir.
- **Appearance:** Bir canonical kişinin videoda kesintisiz göründüğü zaman/frame aralığı.
- **Known:** Mevcut kalıcı gallery/identity ile kalibre edilmiş eşik ve açık-set kararıyla eşleşen kişi.
- **Anonymous:** Daha önce kalıcı anonim identity olarak bilinen kişi.
- **New anonymous:** Bu job sonunda ilk kez kalıcılaştırılacak anonim identity.

## 5. Offline tracklet reconciliation — dondurulmuş invariant

Offline işleme, bu ürünün bilinçli avantajıdır. Final API sonucu tracker'ın ilk geçişte verdiği ham ID'lerle sınırlı değildir.

Örnek:

```text
shot 01: raw_tracklet=t004 -> Rachel
shot 07: raw_tracklet=t019 -> Rachel
shot 12: raw_tracklet=t031 -> Rachel

offline reconciliation:
t004 + t019 + t031 -> canonicalTrackId=video_person_0002 -> faceId=Rachel
```

Kurallar:

1. Sahne kesiminde raw tracker ID'nin değişmesi normaldir.
2. Raw tracker ID final API `trackId` olarak doğrudan dönülmez.
3. Aynı gerçek kişiye ait güvenilir tracklet'ler offline aşamada tek canonical video track'e bağlanır.
4. Final response'taki `trackId`, video boyunca aynı kişi için sabittir.
5. Her canonical track, audit için kaynak `trackletIds` ve birleşme kanıtlarını iç sistemde korur.
6. Known kişi reconciliation'ı yalnız en yüksek tek frame skoruna dayanmaz; tracklet best-shot'ları, identity-level aggregate score, best-different-identity margin ve kalite kanıtı birlikte değerlendirilir.
7. Aynı frame/zaman aralığında eşzamanlı görülen iki farklı yüz normal şartlarda aynı canonical identity altında birleştirilemez. Ayna/ekran gibi istisnalar annotation veya açık kural gerektirir.
8. Unknown tracklet birleştirme eşiği known recognition eşiğinden bağımsız kalibre edilir. Kanıt yetersizse over-merge yapmak yerine ayrı anonymous identity korunur.
9. Bir tracklet içinde kimlik oyları çelişirse frame-majority tek başına yeterli değildir. Kalite ağırlıklı score aggregation ve best-different margin kullanılır.
10. Reconciliation deterministik olmalıdır: aynı model/config/input aynı canonical graph'ı üretmelidir.
11. Mapping saklanmalıdır: `rawTrackerId -> trackletId -> canonicalTrackId -> faceId`.
12. API requirement'ındaki tek `trackId`, canonical video track ID olarak yorumlanır. İleride contract genişletilirse `trackletIds` teknik/audit alanı olarak eklenebilir; kullanıcı onayı olmadan public contract'a eklenmez.

Bu invariant'ı bozarak “tracker ne verdiyse onu dönmek” acceptance değildir.

## 6. Mantıksal offline pipeline

Pipeline tek fiziksel decode geçişini hedefler; “çok aşamalı” olmak videoyu tekrar tekrar CPU'da decode etmek anlamına gelmez.

### Aşama A — Ingest ve doğrulama

- Job/process ID oluştur.
- Request boyut limitini streaming sırasında uygula.
- Container ve codec'i doğrula; extension/MIME tek başına kanıt değildir.
- Süre, fps, original width/height, total frames veya güvenilir duration bilgilerini çıkar.
- Bozuk, boş, unsupported veya limit dışı videoyu structured error ile reddet.
- Retention owner'a deterministik object key ile yaz.
- Upload, mümkün olduğunda bounded byte stream ile GStreamer source'a paralel beslenebilir.
- MP4 metadata'sı sonda olan seek-required dosyalar için bounded spool/finalized-object fallback tasarla; bunu “GPU yavaş” diye gizleme.

### Aşama B — GPU decode ve frame metadata

- Encoded bitstream GStreamer üzerinden NVIDIA NVDEC'e gider.
- Original frame index, PTS/DTS ve timebase korunur.
- Full frame yalnız inference için gerekli GPU memory formatında kalır.
- CPU'ya full-frame image kopyası production hot path'in parçası değildir.

### Aşama C — Sampling, detection ve yerel tracking

- Sampling request/config üzerinden belirlenir.
- Detector bbox + canonical five-point landmark üretir.
- Coordinate reverse mapping original resolution'a doğrulanır.
- Tracker, kesintisiz yerel tracklet üretir.
- Shot boundary bilgisi tracker reset/reconciliation kanıtı olarak tutulabilir.

### Aşama D — Tracklet best-shot ve recognition

- Her frame'de zorunlu recognition yapılmaz.
- Tracklet içinde face size, pose, blur, occlusion, detector confidence ve landmark geometry ile kaliteli örnekler seçilir.
- Seçilen yüzler CUDA five-point alignment ve TensorRT recognizer'dan batched geçirilir.
- Raw embedding, normalized embedding ve varsa kalite sinyali contract'a göre doğru yerde üretilir.
- Qdrant/sample sonuçları person/face identity seviyesinde gruplanır; aynı identity'nin birden fazla sample'ı rakip kişi gibi gösterilmez.

### Aşama E — Video-geneli reconciliation

- Tracklet embedding özetleri, known identity skorları, temporal constraints ve shot graph birlikte değerlendirilir.
- Known tracklet'ler aynı `faceId` altında birleşir.
- Unknown tracklet'ler yalnız kalibre edilmiş güçlü kanıtla birleştirilir.
- Over-merge ve fragmentation ölçülür.

### Aşama F — Aggregation ve persistence

- Canonical kişi başına firstSeen, lastSeen, totalDuration, appearances ve detections üretilir.
- Bbox'lar original resolution'dadır.
- `faceId` final sonuçta her kişi için doludur.
- `new_anonymous` kayıtları ancak reconciliation tamamlandıktan sonra idempotent biçimde persist edilir.
- No-face sonucu `completed`, `personCount=0` olarak ele alınır.
- Job tamamlanmadan kalıcı sonuç, video retention ve identity/vector lifecycle uyumu doğrulanır.

## 7. GPU hot-path sınırı

Hedef ayrım:

- **Python control plane:** FastAPI, validation orchestration, job state, PostgreSQL/MinIO/Qdrant ports, response contracts.
- **Native data plane:** GStreamer/DeepStream, demux/parser, NVDEC, GPU preprocess, TensorRT detector, tracker integration, CUDA alignment, TensorRT recognizer, compact metadata emission.

Hedef akış:

```text
encoded bytes
  -> appsrc/filesrc adapter
  -> demux + codec parser
  -> NVIDIA NVDEC
  -> NVMM/GPU surface
  -> GPU preprocess
  -> TensorRT face detector
  -> GPU postprocess/NMS/landmarks
  -> tracker
  -> batched best-shot CUDA alignment
  -> TensorRT face embedding
  -> GPU L2 normalization
  -> compact metadata/embedding CPU boundary
```

Production hot path'te yasak:

- `cv2.VideoCapture` ile frame decode;
- PIL/OpenCV ile full-frame decode/resize fallback;
- frame'i JPEG'e encode edip tekrar decode etmek;
- her frame'i Python/NumPy'ye taşımak;
- detector output'unu sırf NMS için topluca CPU'ya almak;
- frame başına zorunlu `cudaDeviceSynchronize`;
- pad-probe veya CUDA critical section içinde dosya/JSON/DB/network I/O;
- aynı videoyu detection, recognition ve rendering için üç kez decode etmek;
- sessiz CPU provider fallback;
- fake inference veya mock output'u gerçek GPU runtime kanıtı saymak.

Compact bbox, landmark, score, track metadata ve seçilmiş embedding CPU sınırını geçebilir. Full frame yalnız explicit debug build/contact-sheet modunda ve bounded sayıda indirilebilir.

## 8. GPU/job topolojisi

- Her GPU worker process/container tam olarak bir fiziksel GPU görür.
- Container içinde CUDA device ID daima `0` olabilir; host GPU UUID ayrıca health/telemetry'de kaydedilir.
- Bir video job varsayılan olarak baştan sona tek GPU worker'a pinlenir.
- Aynı videonun frame'lerini GPU'lar arasında dağıtmak tracker state ve PCIe transfer maliyeti nedeniyle varsayılan değildir.
- Multi-GPU throughput, farklı job'ları farklı GPU'lara vererek sağlanır.
- Tek çok uzun videoyu segmentlere bölmek ayrı benchmark ve track-stitching tasarımı gerektirir; ölçüm/onay olmadan yapılmaz.
- Decode, inference, compact result ve persistence bounded queue'larla ayrılır.
- Persistence yavaşsa backpressure uygulanır; sınırsız host/GPU memory birikmez.
- GPU 0/1/2 rolleri source code'a hardcode edilmez.

## 9. Model seçimi ve artifact kuralları

Hiçbir model yalnız adı, parametre sayısı veya GPU memory bolluğu nedeniyle production seçimi sayılmaz.

İlk adaylar:

- detector: SCRFD-10G-KPS, RetinaFace-R50; exact licensed face checkpoint bulunursa YOLO-face aday olarak;
- recognizer: ArcFace IResNet100 ve MagFace IResNet100;
- tracker: NvDCF ve ByteTrack.

Her model artifact için manifest zorunludur:

```text
model name/version
source URL and upstream commit/tag
weight SHA-256
code license
weight/license terms
training dataset/provenance
commercial-use status
input tensor names/shapes/dtypes/layout
color order
resize/letterbox rules
mean/std or scale/offset
output tensor names/shapes
landmark order
opset/export toolchain
TensorRT/CUDA versions
calibration/precision
```

Kurallar:

- Model/weight indirme açık kullanıcı onayı gerektirir.
- InsightFace model-zoo ağırlıklarının research/non-commercial notu yok sayılamaz.
- Ultralytics/YOLO code ve weight lisansı açıkça çözülmeden proprietary deployment'a alınmaz.
- “YOLO11-face” adı resmî, doğrulanmış face checkpoint garantisi değildir.
- MagFace aynı backbone ile benzer compute maliyetine sahip olabilir; doğruluk veya kalite açısından “free upgrade” kabul edilmez.
- MagFace kalite sinyali embedding normundan okunacaksa pre-L2 norm saklanır; erken normalization ile sinyal kaybedilmez.
- Model swap, alignment ve threshold/calibration yeniden doğrulanmadan yapılamaz.

## 10. Alignment ve coordinate release gate'i

Recognition sonucu kötü olduğunda önce model suçlanmaz. Şunlar ayrı ayrı doğrulanır:

1. Detector input preprocess parity.
2. Detector output tensor/anchor/stride contract'ı.
3. Letterbox/resize padding ve reverse coordinate mapping.
4. Exact five-landmark order.
5. Original-resolution bbox mapping.
6. ArcFace canonical template.
7. Similarity transform yönü.
8. Pixel-center ve interpolation davranışı.
9. GPU aligned crop ile reference crop pixel comparison/contact sheet.
10. Recognizer input tensor parity.
11. Raw embedding parity.
12. L2-normalized embedding parity.
13. Batch-1 ile batch-N parity.
14. Enrollment ile query/video pipeline parity.

Parity requirement sırf test geçsin diye düşürülmez.

## 11. Confidence ve identity kararları

- Raw cosine similarity yüzde olasılık değildir.
- Raw cosine `confidence` diye doğrudan UI/API anlamına çevrilmez.
- Public confidence kullanılacaksa identity-disjoint calibration split üzerinde kalibre edilir.
- Threshold tek bir Friends örneğini geçirmek için seçilmez.
- Candidate sample'lar önce identity/faceId bazında gruplanır.
- Runner-up margin en iyi **farklı identity** ile hesaplanır.
- Aynı kişinin iki sample skoru top1/top2 identity margin sayılmaz.
- Open-set unknown kararı calibration ve untouched held-out set ile doğrulanır.
- Anonymous reconciliation için ayrı threshold ve over-merge ölçümü gerekir.

## 12. Annotation ve golden dataset

`test_videos/` raw medya için local-only dizindir ve Git'e eklenmez. Dataset/model/video artifact'ları repository boyutunu büyütmez; Git yalnız manifest, hash, annotation ve küçük izinli fixture'ları taşır.

### Video ground truth

İlk golden set bütün bölümleri rastgele frame'lere bölmez. Split, episode/video/scene seviyesinde yapılır:

- calibration;
- held-out evaluation;
- stress cases;
- no-face/unknown controls.

Her detection annotation en az:

```text
videoId
frameIndex
ptsMs
shotId
trackletId
canonicalTrackId
identityLabel
bboxXYXY in original pixels
visibility
occluded
ignore
```

Landmark parity için seçilmiş keyframe'lere canonical five landmarks eklenir.

Annotation kuralları:

- CVAT/model-assisted pre-annotation yalnız başlangıçtır; insan review olmadan ground truth değildir.
- Scene cut sonrasında local tracklet değişebilir; canonical identity değişmez.
- Bütün yabancılar tek `unknown` ID altında toplanmaz. Video içindeki farklı kişiler ayrı stable unknown label alır.
- Same-video adjacent frame leakage calibration/evaluation arasında yasaktır.
- Gallery fotoğrafı evaluation videosundan çıkarılmış aynı/komşu frame ise held-out kanıtı sayılmaz.
- Annotator belirsizliği `ignore`/visibility ile ifade edilir; zorla yanlış label verilmez.
- Annotation schema versioned ve converter testli olur.

### Gallery quality

`DATASET/` klasör ismi tek başına doğru hedef yüz kanıtı değildir. Grup fotoğrafları vardır.

Her gallery image için target bbox/landmarks, identity, approval, quality ve split içeren manifest oluşturulur. “En büyük yüz target kişidir” varsayımı production enrollment'ta kullanılamaz.

## 13. Ölçüm ve model bake-off

Aynı annotated input ve aynı split üzerinde ölçülmeyen modeller karşılaştırılmış sayılmaz.

### Detection

- precision/recall ve AP;
- face-size bucket recall;
- profile/occlusion/blur buckets;
- landmark NME veya pixel error;
- original-coordinate parity.

### Tracking

- HOTA;
- IDF1;
- identity switches;
- fragmentation;
- tracklet recall;
- scene-cut behavior.

### Recognition/open set

- top-1 identification rate;
- TAR/FNMR at target FAR;
- known/unknown confusion;
- best-identity vs best-different margin;
- identity-balanced metrics;
- pose/age/quality buckets.

### Offline reconciliation

- canonical identity precision/recall;
- over-merge rate;
- over-split/fragmentation rate;
- known-person continuity across cuts;
- unknown-person false merge rate.

### Performance

- upload/validation latency;
- time to first decoded frame;
- decode, detector, tracker, alignment, recognizer ve reconciliation stage timing;
- end-to-end wall time;
- real-time factor;
- processed/source FPS;
- queue depth/backpressure;
- GPU utilization, memory, power/clocks;
- CPU/RSS/disk/object-store latency;
- cold-start ve steady-state ayrı.

“GPU-only”, “real-time”, “production-ready” veya “3x scaling” gerçek ölçüm olmadan söylenmez.

## 14. API contract kuralları

API-only üründür; UI eklenmez.

Requirement'taki temel davranışlar korunur:

```text
POST   /videos/recognize
GET    /videos/jobs/{jobId}
GET    /videos/jobs/{jobId}/result
DELETE /videos/jobs/{jobId}
GET    /faces/{faceId}/appearances
```

Nihai prefix/versioning architecture sprintinde dondurulur; örnek endpoint'i sessizce değiştirme.

Kurallar:

- POST uzun işlemi beklemez; job/process ID ile `202 Accepted` döner.
- Video request içinde kabul edilir; raw octet-stream veya multipart contract'ı OpenAPI testleriyle netleştirilir.
- Status en az pending/processing/completed/failed anlamlarını taşır; cancellation açıkça modellenir.
- Progress source frames, processed frames ve pipeline stage üzerinden dürüst hesaplanır.
- No-face başarılı sonuçtur.
- Failed job sanitized stable error code taşır; raw stack/SQL/path/secret dönmez.
- Result yalnız finalization sonrası kalıcı ve tutarlı okunur.
- Her person result: faceId, canonical trackId, status, name, metadata, firstSeen, lastSeen, totalDuration, appearances, detections ve calibrated confidence taşır.
- Detection frame index, timestamp ve original-resolution bbox taşır.
- Büyük detection listeleri için response size/pagination/compression kararı ölçülerek verilir; requirement sessizce atılmaz.
- Cancel idempotent olur; GStreamer EOS/teardown, temporary upload ve job state temizlenir.

## 15. Video validation ve retention

- Extension ve client MIME güvenilir değildir; container/codec probe gerekir.
- Supported container ve codec ayrı config alanlarıdır.
- Max bytes, max duration, timeout, sampling, retention ve concurrency environment üzerinden gelir.
- Video blob owner, object key, checksum, content length ve retention expiry açıkça tanımlanır.
- Retention cleanup idempotent ve audit edilebilir olur.
- Job/process ID ile video yeniden işlenebilir.
- Raw local filesystem path client response/log/Qdrant payload'a yazılmaz.
- Video upload başarısı ile inference başarısı ayrı lifecycle'dır.

## 16. Persistence ve cross-store consistency

Phase 1 ile birleştiğinde:

- PostgreSQL relational/job/identity source of truth;
- MinIO video ve binary object owner;
- Qdrant rebuildable embedding index olur.

Bu bağımsız lab repository'si bu sistemlerin contract'larını port arkasında tutar; Phase 1'i taklit eden ikinci ve uyumsuz identity modeli üretmez.

Multi-store workflow'lar:

- deterministic IDs/object keys;
- idempotent retry;
- explicit state;
- bounded batch/concurrency;
- failure event;
- compensation/reconciliation;
- partial-failure integration tests

içermelidir.

`new_anonymous` için aynı video retry'ı duplicate identity/sample/vector oluşturmamalıdır. Reconciliation tamamlanmadan geçici tracklet anonymous person olarak kalıcılaştırılmaz.

## 17. Katman sınırları

Hedef paket ayrımı:

- `domain`: job, tracklet, canonical identity, appearance ve result invariant'ları;
- `application`: ingest, process, reconcile, aggregate use-case'leri;
- `ports`: video store, job repository, gallery/vector search, GPU worker contract'ları;
- `infrastructure`: PostgreSQL/MinIO/Qdrant/GStreamer implementations;
- `api`: validation ve application çağrıları;
- `native`: GStreamer/DeepStream/CUDA/TensorRT data plane;
- `tools`: annotation converters, gallery QA ve benchmark;
- `tests`: unit, contract, integration, GPU runtime, accuracy ve performance.

API router içinde SQL, MinIO, Qdrant veya GPU business logic olmaz. Native callback içinde DB/network I/O olmaz. Domain outer layer import etmez.

## 18. Docker ve runtime discovery

- Docker Compose tek komutla API, worker ve gerekli persistence servislerini kaldırabilmelidir.
- Production worker source bind-mount'a bağımlı olmaz.
- Dataset/video/model/engine image içine kopyalanmaz; read-only/configured mount veya object store kullanılır.
- GPU mapping Compose/runtime config ile yapılır.
- Başlangıçta exact driver, CUDA, TensorRT, DeepStream, GStreamer ve plugin envanteri kaydedilir.
- `gst-inspect-1.0` ile seçilen her NVIDIA/GStreamer elementinin gerçekten kurulu olduğu doğrulanır.
- Health `live` ile `ready` ayrılır; ready model/engine/decoder ve dependency health'ini doğrular.
- Worker SIGTERM/cancel/EOS sonrası CUDA/GStreamer resource'larını temiz kapatır.

## 19. Test ve doğrulama disiplini

Normal sıra:

1. Failing unit/contract test veya reproducer.
2. Minimum implementation.
3. Targeted unit tests.
4. Integration/contract tests.
5. Gerçek GStreamer/plugin smoke.
6. Gerçek TensorRT/GPU video run.
7. Golden annotation accuracy run.
8. Performance telemetry run.
9. Lint/type/build.
10. `git diff --check`, scope ve privacy review.

Mock test kanıtlamaz:

- gerçek NVDEC/GPU memory path;
- gerçek tracker metadata lifetime;
- gerçek model/alignment doğruluğu;
- gerçek PostgreSQL/MinIO/Qdrant consistency;
- gerçek video retention;
- gerçek throughput.

GStreamer ERROR bus message başarı exit code'u veremez. Pipeline error/timeout/cancel/EOS yolları ayrı test edilir.

## 20. Security, privacy ve veri hijyeni

- Raw video, face crop, embedding, kişi adı ve filesystem path'i gereksiz loglanmaz.
- Public loglar dataset folder/person label taşımaz; benchmark raporları anonymized identity ID kullanır.
- Qdrant payload'a geniş PII veya raw media path eklenmez.
- MinIO key'leri sistem UUID'lerinden oluşur.
- Secret/key hardcode edilmez.
- Raw exception client'a dönmez.
- Friends videoları ve internetten toplanmış gallery fotoğrafları public Git'e eklenmez; mevcut tracked media'nın lisans durumu production kullanımdan önce çözülür.
- Test verisi müşteri doğruluk iddiası veya ticari model lisansı yerine geçmez.

## 21. Reference-first engineering

Implementation öncesi:

1. İlgili reference ID'lerini `opensourcereferences/references.md` içinden seç.
2. Exact installed version'ın official docs'unu oku.
3. Upstream source/sample/test dosyasını pinned tag/commit ile doğrula.
4. License ve model-weight terms'i kaydet.
5. Eski `NVDIAgstreamer` davranışı varsa salt-okunur karşılaştır.
6. Reuse/adapt/reject kararını `docs/implementation/REFERENCE_DECISION_LOG.md` içine yaz.
7. Failing test/reproducer ile başla.

Blog veya LLM önerisi tek başına teknik karar kanıtı değildir.

## 22. Agent araç ve skill disiplini

Mevcut olduğunda:

- `using-superpowers`: skill-first workflow governance;
- `brainstorming`: yeni architecture/model/runtime kararı;
- `writing-plans`: multi-file sprint planı;
- `executing-plans`: onaylanmış plan uygulaması;
- `test-driven-development`: production behavior/bug fix;
- `systematic-debugging`: runtime/performance/root-cause;
- `verification-before-completion`: completion claim;
- `codebase-memory-mcp`: multi-file discovery/compaction recovery;
- `context7`: version-sensitive API doğrulaması;
- `deepwiki`: approved upstream repo architecture incelemesi;
- `exa`: primary/current source discovery;
- `postman`: gerçek API acceptance;
- `playwright`: UI olmadığı için kullanılmaz;
- `21st`: `FORBIDDEN_NOT_USED`.

Subagent/parallel çalışma varsayılan değildir; yalnız kullanıcı açıkça onaylarsa bağımsız read-only işler için kullanılır. Ruflo kullanılmaz.

Her final raporda gerçekten kullanılan/skipped araçlar dürüstçe belirtilir.

## 23. Git ve destructive işlem sınırı

Kullanıcı açıkça onaylamadan:

- `git add`, commit, push, merge, rebase veya history rewrite yapma;
- tracked dosya/model/engine/video silme;
- volume veya persistence verisi silme;
- model/dataset indirme;
- system CUDA/driver/package değiştirme;
- architecture/requirement contract'ını güncelleme.

Dirty worktree kullanıcıya aittir. Unrelated değişiklikleri koru.

## 24. Sprint dokümantasyonu ve completion

Her sprintte `docs/implementation/CURRENT_SPRINT.md` şunları içerir:

- objective;
- exact deliverables;
- acceptance commands;
- non-goals;
- blockers/hard stops.

Sprint sonunda kısa implementation ledger ve review evidence oluşturulur. Rapor yazmak ürün ilerlemesinin yerine geçmez.

Completion verdict:

- `PASS`: bütün zorunlu acceptance kanıtları geçti.
- `PARTIAL`: çalışan değer var fakat zorunlu gate eksik.
- `BLOCKED`: kullanıcı kararı/yetkisi gerekiyor.
- `NOT_TESTED`: implementasyon var, gerçek runtime kanıtı yok.

Kanıt olmadan `production-ready`, `GPU-only`, `real-time`, `accuracy verified`, `commercially licensed` veya `fully optimized` deme.

## 25. İlk önerilen geliştirme sırası

1. Repository governance, sprint planı ve runtime inventory.
2. Video manifest/probe ve local-only test media contract'ı.
3. Gallery target-face QA manifesti.
4. CVAT annotation schema/converter ve golden clips.
5. Raw upload/job API contract.
6. Bounded upload -> GStreamer `appsrc` -> demux/parser.
7. NVDEC/NVMM ve no-full-frame-CPU-copy gate'i.
8. Detector reference parity ve model bake-off.
9. Local tracker/tracklet output.
10. CUDA alignment + recognizer parity.
11. Best-shot ve identity-level aggregation.
12. Offline canonical reconciliation.
13. Known/anonymous/new_anonymous persistence.
14. Person-level result/history/appearance APIs.
15. Cancellation, retention ve failure recovery.
16. Docker Compose end-to-end.
17. Accuracy/performance report.
18. Phase 1 integration contract ve merge planı.

Bu sıra, blocker olmayan küçük bulgular için tekrar tekrar foundation sprinti açmak amacıyla kullanılmaz. Her sprint çalışan dikey sonuç üretmelidir.
