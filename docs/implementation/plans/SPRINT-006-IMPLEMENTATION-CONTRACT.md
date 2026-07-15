# SPRINT 06 — NATIVE FACE TRACKLETS + OFFLINE CANONICAL IDENTITY RECONCILIATION
BINDING IMPLEMENTATION CONTRACT

IMPORTANT

Bu contract Sprint 05 recognition tamamlandıktan sonra uygulanacaktır. Sprint 05 sırasında tracker implement etmeye çalışma. Sprint 05 yalnız bu sprintin ihtiyaç duyduğu embedding/quality/evidence contract’ını üretmelidir.

Amaç yalnız ByteTrack yazmak değildir. Hedef zincir:

```text
per-frame detection
→ short-term local tracklet
→ tracklet recognition evidence
→ cross-scene canonical video person
→ persistent faceId candidate
→ firstSeen / lastSeen / appearances
→ best shot
→ final second-pass annotated video
```

Aynı kişinin uzak sahnelerinde raw tracker ID’yi yapay biçimde korumaya çalışma.

Örnek doğru ilişki:

```text
frame 1:
  detection=f1_d0
  tracklet=tl_0007
  videoPerson=vp_0001
  faceId=face_rachel

frame 7345:
  detection=f7345_d1
  tracklet=tl_0191
  videoPerson=vp_0001
  faceId=face_rachel
```

`tl_0007 != tl_0191` normaldir.
Canonical video person ve faceId aynıdır.

## 1. REFERENCE-FIRST ZORUNLULUĞU

ByteTrack algoritmasını hafızadan yeniden icat etme.

Primary references:

1. Paper:
   https://arxiv.org/abs/2110.06864

2. Official repository:
   https://github.com/FoundationVision/ByteTrack

3. Exact pinned reference commit:
   d1bf0191adff59bc8fcfeaa0b33d3d1642552a99

4. Relevant upstream C++ source:
   https://github.com/FoundationVision/ByteTrack/tree/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/deploy/TensorRT/cpp

5. License:
   https://github.com/FoundationVision/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/LICENSE

6. NVIDIA tracker contract:
   https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_plugin_gst-nvtracker.html

7. DeepStream custom metadata:
   https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_plugin_metadata.html

8. GStreamer BaseTransform:
   https://gstreamer.freedesktop.org/documentation/base/gstbasetransform.html

ByteTrack MIT license notice ve adapted-source provenance:

- `THIRD_PARTY_NOTICES.md`
- `REFERENCE_DECISION_LOG.md`

içinde tutulmalı.

Official C++ kodunu körlemesine kopyalama. Upstream C++ örneğinde bizim production contract’ımıza uymayan noktalar var:

- OpenCV `cv::Rect` bağımlılığı;
- fixed `dt=1`;
- update çağrısını frame sayan internal counter;
- process-global static track ID generator;
- hardcoded thresholds;
- bbox IoU hesabında `+1`;
- `exit()` / `system("pause")` tarzı error handling;
- single-source demo varsayımları;
- PTS/sampling gap bilgisi yok;
- deterministic multi-source ID contract’ı yok.

Algoritmik state machine adapte edilecek; demo application kodu taşınmayacak.

## 2. ID KATMANLARI

Aşağıdaki kimlikler kesinlikle ayrılmalı:

```text
detection_id
    Tek frame içindeki bir bbox observation.

raw_object_id
    DeepStream NvDsObjectMeta.object_id.
    Internal transport alanı. API identity değildir.

tracklet_id
    Kesintisiz veya kısa süreli kayıp toleranslı motion/appearance track’i.

video_person_id
    Aynı video içinde farklı tracklet’lerin offline birleştirilmiş canonical kişisi.

face_id
    Veritabanı/gallery tarafındaki kalıcı known veya anonymous identity. Public API’de legacy trackId, canonical video_person_id olacaktır.
```

Yeni evidence alanı:

```json
"tracklets": [
  {"trackletId": "s0_tl_0007", "...": "..."},
  {"trackletId": "s0_tl_0191", "...": "..."}
]
```

Raw DeepStream object ID public API’ye sızmayacak.

## FINAL TOPOLOGY

Sprint 05 recognition tamamlandıktan sonraki analysis topology:

```text
filesrc
→ qtdemux
→ h264parse
→ nvv4l2decoder
→ nvstreammux temporal batch
→ RetinaFace
→ GPU face alignment
→ TensorRT GlintR100
→ compact embedding/recognition metadata
→ mvfacetracker
→ evidence writer
→ fakesink or provisional render
```

Tracker recognition’dan sonra yerleştirilecek. Bunun nedeni:

IoU/motion association temel yol olarak kalır;
geçerli embedding varsa face appearance gating kullanılabilir;
scene cut veya crossing durumunda yanlış ID continuation azaltılır;
embedding yoksa tracker IoU-only çalışabilir.

Final annotated output tek-pass provisional render olmayacak. Canonical reconciliation sonrası:

```text
original encoded video
→ NVDEC
→ final metadata join by frame/PTS
→ nvdsosd
→ NVENC
→ final annotated MP4
```

Inference ikinci kez çalıştırılmayacak.

## SOURCE LAYOUT

Önerilen source yapısı:

```text
backend/native/tracking/
  tracker_types.h
  continuous_iou.h
  continuous_iou.cpp
  kalman_filter.h
  kalman_filter.cpp
  linear_assignment.h
  linear_assignment.cpp
  byte_tracker.h
  byte_tracker.cpp
  multi_source_tracker.h
  multi_source_tracker.cpp

backend/native/plugins/gst-mvfacetracker/
  gstmvfacetracker.h
  gstmvfacetracker.cpp
  tracklet_meta.h
  tracklet_meta.cpp
  CMakeLists.txt

backend/app/domain/video_tracking.py
backend/app/application/services/reconcile_video_identities.py
backend/app/application/services/select_best_shots.py

backend/tests/unit/tracking/
backend/tests/native/tracking/
backend/tests/integration/tracking/
backend/tests/fixtures/tracking/
```

Element adı mvfacetracker olacak. NVIDIA official elementi gibi nvdsbytetrack adı kullanılmayacak.

CLI:

```text
--tracker off|nvdcf|mvfacetracker
```

Production default ancak acceptance geçince mvfacetracker olabilir.

## CORE C++ TYPES

Core tracker DeepStream veya GStreamer type’larına bağımlı olmayacak.

Yaklaşık public contract:

```cpp
namespace mv::tracking {

using SourceId = std::uint32_t;
using FrameNumber = std::uint64_t;
using TimestampNs = std::uint64_t;
using TrackletId = std::uint64_t;
using DetectionId = std::uint64_t;

struct RectF {
    float x1;
    float y1;
    float x2;
    float y2;
};

struct FrameKey {
    SourceId source_id;
    FrameNumber frame_number;
    TimestampNs pts_ns;
};

struct Detection {
    DetectionId detection_id;
    RectF bbox;
    float detector_score;

    bool tracking_only;

    bool embedding_valid;
    float embedding_quality;
    std::array<float, 512> embedding;
};

enum class TrackState {
    Tentative,
    Tracked,
    Lost,
    Removed
};

enum class TerminationReason {
    LostTimeout,
    SourceEos,
    StreamReset,
    TimestampGap,
    IdentityConflict,
    PipelineShutdown
};

struct Assignment {
    DetectionId detection_id;
    TrackletId tracklet_id;
    TrackState state;
    bool newly_created;
};

struct EndedTracklet {
    TrackletId tracklet_id;
    SourceId source_id;

    FrameNumber start_frame;
    FrameNumber end_frame;

    TimestampNs start_pts_ns;
    TimestampNs end_pts_ns;

    std::size_t detection_count;
    TerminationReason reason;
};

struct UpdateResult {
    std::vector<Assignment> assignments;
    std::vector<EndedTracklet> ended_tracklets;
};

struct TrackerConfig {
    float detector_emit_threshold;
    float track_low_threshold;
    float track_high_threshold;
    float new_track_threshold;

    float first_match_cost_threshold;
    float second_match_cost_threshold;

    float min_iou_gate;
    float min_embedding_gate;

    float iou_weight;
    float embedding_weight;

    float min_embedding_quality;

    TimestampNs lost_timeout_ns;
    TimestampNs maximum_timestamp_gap_ns;

    std::size_t max_active_tracks;
};

class ByteTracker {
public:
    explicit ByteTracker(SourceId source_id, TrackerConfig config);

    UpdateResult update(
        const FrameKey& frame,
        std::span<const Detection> detections);

    std::vector<EndedTracklet> flush(TerminationReason reason);

private:
    // No GStreamer or DeepStream types here.
};

class MultiSourceTracker {
public:
    UpdateResult update(
        const FrameKey& frame,
        std::span<const Detection> detections);

    std::vector<EndedTracklet> flush_source(
        SourceId source_id,
        TerminationReason reason);

    std::vector<EndedTracklet> flush_all(
        TerminationReason reason);

private:
    std::unordered_map<SourceId, ByteTracker> trackers_;
};

}
```

Core unit testleri GPU veya DeepStream olmadan çalışmalı.

## BOUNDING BOX VE IoU CONTRACT

Bütün bbox’lar continuous xyxy:

```text
x1, y1 inclusive geometric boundary
x2, y2 continuous right/bottom boundary
width  = x2 - x1
height = y2 - y1
```

IoU:

```cpp
float intersection_width =
    std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));

float intersection_height =
    std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
```

`+1` kullanılmayacak.

Invalid/zero-area/NaN bbox match’e girmeyecek.

Sprint 01 NMS ile aynı coordinate convention kullanılmalı.

## BYTE STATE MACHINE

Her processed frame, detection olmasa bile tracker’a verilmelidir.

Algoritma:

A. Input detections:
   high = score >= track_high_threshold
   low  = track_low_threshold <= score < track_high_threshold
   discard = score < track_low_threshold

B. Predict:
   active Tracked + Lost track states Kalman ile current PTS’ye predict edilir.

C. First association:
   tracked/lost pool ↔ high detections
   cost = IoU/motion + optional appearance
   LAP/Hungarian/JV assignment

D. Second association:
   unmatched currently-tracked tracks ↔ low detections
   low detections yeni track başlatamaz

E. Tentative association:
   tentative/unconfirmed tracks ↔ remaining high detections

F. New track:
   unmatched high detection ancak score >= new_track_threshold ise track başlatır

G. Lost:
   unmatched tracked → Lost

H. Removed:
   current_pts - last_matched_pts > lost_timeout_ns ise Removed

I. Output:
   only activated assignments object metadata’ya yazılır

Track ID hiçbir zaman reuse edilmez.

Track state:

```text
Tentative → Tracked → Lost → Tracked
Tentative → Removed
Tracked → Lost → Removed
```

Removed state geri açılamaz.

## LOW-CONFIDENCE DETECTOR CONTRACT

Mevcut RetinaFace yaklaşık 0.5 altında detection siliyorsa gerçek ByteTrack uygulanamaz.

Ayrı threshold’lar:

```text
detector_emit_threshold = örn. 0.10
track_low_threshold
track_high_threshold
public_detection_threshold
recognition_threshold
```

hardcoded olmayacak.

RetinaFace low candidates için custom metadata:

```cpp
enum class DetectionRole {
    Public,
    TrackingOnly
};
```

Low-confidence detection:

yeni track başlatamaz;
mevcut track’i güncelleyebilir;
public JSON/OSD/recognition’a otomatik çıkmaz;
tracking_only=true taşır.

Eğer low candidate downstream’de obj_meta olarak tutuluyorsa MvDetectionRoleMeta ile açıkça işaretlenmeli. misc_obj_info[3]=1 gibi undocumented magic integer kullanılmamalı.

## PTS-AWARE KALMAN

Upstream fixed dt=1 doğrudan taşınmayacak.

Her source için:

```cpp
delta_pts = current_pts - previous_pts
nominal_frame_period = video metadata’dan
dt = delta_pts / nominal_frame_period
```

Kurallar:

PTS monotonic olmalı;
duplicate veya backward PTS hard error;
VFR videoda frame/fps primary timestamp değildir;
PTS yoksa açık fallback raporlanır;
sampling gap dt içine yansıtılır;
aşırı gap maximum_timestamp_gap_ns aşarsa aktif tracklet’ler kapanır;
lost age update-count ile değil gerçek elapsed PTS ile ölçülür.

Kalman state örneği upstream ByteTrack ile uyumlu:

```text
[cx, cy, aspect_ratio, height,
 vx, vy, va, vh]
```

Motion matrix her update’ta actual dt ile oluşturulmalı.

## TEMPORAL BATCH ADAPTER

Aynı batch içindeki framelerin ardışık olduğu varsayılmayacak.

Plugin transform:

```cpp
GstFlowReturn transform_ip(GstBuffer* buffer) {
    NvDsBatchMeta* batch = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch) return GST_FLOW_ERROR;

    std::vector<NvDsFrameMeta*> frames = collect_frames(batch);

    std::stable_sort(
        frames.begin(),
        frames.end(),
        [](auto* a, auto* b) {
            if (a->source_id != b->source_id)
                return a->source_id < b->source_id;
            if (a->buf_pts != b->buf_pts)
                return a->buf_pts < b->buf_pts;
            return a->frame_num < b->frame_num;
        });

    for (NvDsFrameMeta* frame : frames) {
        validate_monotonic_frame(frame);

        auto adapter_detections =
            collect_and_stably_sort_detections(frame);

        auto result = tracker.update(
            to_frame_key(frame),
            to_core_detections(adapter_detections));

        attach_tracklet_metadata(
            batch,
            frame,
            adapter_detections,
            result.assignments);

        post_ended_tracklet_messages(result.ended_tracklets);
    }

    return GST_FLOW_OK;
}
```

batch_id yalnız tensor/surface slot mapping için kullanılabilir. Temporal sıra için kullanılamaz.

Cross-buffer ledger:

```cpp
struct SourceLedger {
    bool initialized;
    uint64_t last_frame_number;
    uint64_t last_pts_ns;
    uint64_t processed_frame_count;
};
```

Hard invariants:

frame_number > last_frame_number
pts_ns > last_pts_ns, duplicate PTS policy açıkça tanımlanmadıysa
her frame bir kez update edilir
empty detection frame de update edilir
final partial batch kaybolmaz

Offline correctness modunda out-of-order frame geldiğinde reorder window uydurma. Explicit pipeline error ver ve ledger raporla.

## DETERMINISTIC DETECTION ORDER

NvDsObjectMeta linked-list order’ı deterministic ID kaynağı değildir.

Detection’ları match öncesi stable sırala:

class_id
x1
y1
x2
y2
detector_score descending
original object meta sequence index

Track rows da tracklet_id ascending sıralanmalı.

LAP input row/column sırası deterministik olmalı.

Equal-cost tie testinde sonuç her run aynı olmalı.

Global static next_id() yasak.

Per-source counter:

```cpp
uint64_t next_local_track_id_ = 1;
```

NvDs object ID encode gerekiyorsa:

```text
upper bits = source ID
lower bits = local monotonically increasing ID
```

Bounds assert edilmeli.

Reserved values:

```text
0
UNTRACKED_OBJECT_ID
UINT64_MAX
```

gerçek track ID olarak kullanılmaz.

## ASSOCIATION COST

Base cost:

```cpp
iou_cost = 1 - continuous_iou(predicted_box, detection_box)
```

Embedding geçerliyse:

```cpp
appearance_cost = 1 - cosine(track_embedding, detection_embedding)
```

High-stage combined cost:

```cpp
cost =
    iou_weight * iou_cost +
    embedding_weight * appearance_cost
```

Hard gates:

```text
if IoU < min_iou_gate and embedding evidence is unavailable:
    cost = INF

if both track and detection have high-quality embeddings
and cosine < min_embedding_gate:
    cost = INF
```

Low-confidence second association başlangıçta IoU ağırlıklı olabilir.

Gallery label association cost olarak doğrudan kullanılmamalı. Raw normalized face embedding kullanılmalı. Yanlış bir top-1 label bütün track’i zorla merge etmemeli.

Track online appearance prototype:

```text
prototype = L2(
    weighted running mean of valid normalized embeddings
)
```

Yalnız:

```text
embedding_valid
and embedding_quality >= min_embedding_quality
```

ise güncellenir.

Online EMA yalnız motion association içindir. Final canonical identity offline top-K evidence üzerinden tekrar hesaplanır.

## TRACKLET META

Her assigned public object’a owned user metadata ekle:

```cpp
struct MvTrackletMeta {
    uint32_t schema_version;

    uint64_t raw_object_id;
    uint64_t tracklet_id;
    uint32_t source_id;

    uint64_t frame_number;
    uint64_t pts_ns;

    uint32_t track_age;
    uint32_t hit_count;
    uint32_t lost_count;

    bool tentative;
    bool confirmed;
};
```
copy callback;
release callback;
no dangling pointer;
nvstreamdemux sonrasında geçerli;
metadata type versioned.

NvDsObjectMeta.object_id aynı gerçek tracklet ID ile güncellenebilir fakat public API bunu doğrudan kullanmaz.

Tracklet sonlandığında plugin bus’a application message atsın:

```text
message name: mv-tracklet-ended
fields:
  source_id
  tracklet_id
  start_frame
  end_frame
  start_pts_ns
  end_pts_ns
  detection_count
  termination_reason
```

EOS event’ı downstream’e gönderilmeden önce tüm active/lost tracklet’ler flush edilmeli.

## THREADING VE LIFETIME

Tracker core CPU/metadata-only çalışır.
Full frame CPU’ya taşınmaz.
Plugin buffer streaming thread’inde çalışabilir; face sayısı küçük olduğundan önce benchmark edilir.
Bir pipeline içinde global singleton tracker yok.
Her element instance kendi context’ine sahip.
Multiple source state ayrı map’te.
Raw NvDsObjectMeta* core tracker içinde saklanmaz.
Plugin transform döndükten sonra object pointer saklanmaz.
State değişimi için process-global static kullanılmaz.
EOS/READY/NULL transition leak testi yapılır.

## BEST-SHOT EVIDENCE

Tracker best shot seçmez; yalnız tracklet sınırı üretir.

Recognition her detection için şu scalars’ı üretmelidir:

```text
detection_confidence
face_width/height/area
border_clip_fraction
landmark/alignment validity
alignment residual
pose/frontal score
sharpness score
top1 similarity
top2 similarity
margin
embedding_ref
```

Sharpness için CPU full-frame kullanılmaz. Aligned 112×112 crop üzerinde GPU Tenengrad/Sobel veya eşdeğer compact scalar üretilebilir.

Hard quality gate:

```text
alignment valid
embedding valid
minimum face size
maximum border clipping
finite quality fields
```

Tracklet içinde top-K candidate:

quality descending;
minimum temporal separation;
duplicate adjacent frames elenir;
deterministic tie: earlier PTS/frame.

Tek bir “en yüksek detector score” best shot değildir.

## OFFLINE TRACKLET EVIDENCE

Native analysis output:

```text
detections.jsonl
tracklets.jsonl
embeddings.f32
embedding_index.jsonl
run_manifest.json
```

tracklets.jsonl örneği:

```json
{
  "schema_version": "2.0",
  "source_id": 0,
  "tracklet_id": "s0_tl_0007",
  "start_frame": 31,
  "end_frame": 184,
  "start_pts_ns": 1292958333,
  "end_pts_ns": 7674333333,
  "termination_reason": "lost_timeout",
  "detection_count": 143,
  "embedding_refs": [12, 18, 25],
  "best_shot_candidates": [
    {
      "frame": 120,
      "pts_ns": 5005000000,
      "detection_id": "f120_d0",
      "quality": 0.91,
      "embedding_ref": 18
    }
  ]
}
```

Embedding sidecar public log değildir. Biyometrik internal artifact olarak retention/security policy’ye tabidir.

## PYTHON OFFLINE RECONCILER

Python burada production video hot path değildir. Yalnız compact tracklet/embedding evidence işler.

Domain types:

```python
@dataclass(frozen=True)
class TrackletEvidence:
    tracklet_id: str
    source_id: int
    start_frame: int
    end_frame: int
    start_pts_ns: int
    end_pts_ns: int
    observations: tuple[RecognitionObservation, ...]


@dataclass(frozen=True)
class CanonicalVideoPerson:
    video_person_id: str
    face_id: str | None
    status: str
    name: str | None
    tracklet_ids: tuple[str, ...]
    appearances: tuple[AppearanceInterval, ...]
    first_seen: float
    last_seen: float
    total_duration: float
    final_confidence: float
    best_shot: BestShotEvidence
```

Reconciliation aşamaları:

1. Load and validate evidence
2. Build tracklet-level quality-filtered prototype
3. Resolve strong known-gallery candidates
4. Match existing anonymous gallery candidates
5. Build cannot-link constraints
6. Cluster unresolved tracklets conservatively
7. Create canonical video persons
8. Compute appearances
9. Select best shot
10. Build final per-frame label map

## TRACKLET PROTOTYPE

Her tracklet için:

Geçersiz quality observations elenir.
Zamansal olarak çeşitli top-K observation seçilir.
Her embedding L2 normalized olmalı.
Prototype:
`L2(weighted mean(selected embeddings))`
Aynı tracklet içindeki bütün frameler eşit oy kullanmaz.
Bin tane benzer adjacent frame, bir tracklet’e bin bağımsız oy sağlamaz.

Tracklet identity acceptance:

minimum qualified observation count;
prototype top1 threshold;
top1-top2 margin;
qualified observations içinde candidate consistency;
conflicting strong identity evidence yok.

Threshold’lar gallery/model version’ıyla birlikte kaydedilir.

## KNOWN PERSON MERGE

İki tracklet aynı known face_id ile güçlü biçimde eşleşiyorsa canonical merge adayıdır.

Fakat hard constraints önce kontrol edilir.

Cannot-link:

aynı source ve zaman aralıkları overlap ediyorsa;
aynı frame’de ayrı bbox olarak görünüyorsa;
strong different known identity evidence varsa.

Cannot-link ihlalinde:

merge etme;
conflict evidence raporla;
confidence düşürüp gizlice birini seçme.

Rachel örneği:

```text
tl_0007 prototype → Rachel 0.74, margin 0.22
tl_0191 prototype → Rachel 0.76, margin 0.24
no temporal overlap
=> vp_0001 / face_rachel
```

## UNKNOWN / ANONYMOUS RECONCILIATION

Önce persistent anonymous gallery aranır.

Eşleşme varsa mevcut faceId kullanılır.

Eşleşme yoksa aynı video içindeki unresolved tracklet prototype’ları cluster edilir.

Basit transitive union-find yasak:

```text
A similar B
B similar C
A not similar C
```

durumunda A-B-C zincir merge olmamalı.

Conservative seçenek:

agglomerative complete-link clustering;
bütün pair similarities cluster threshold üzerinde;
cannot-link constraint;
deterministic node order;
known/unknown threshold’ları ayrı.

Yeni anonymous cluster için yalnız reconciliation tamamlanınca bir face oluşturulur.

Idempotency key:

```text
(job_id, canonical_cluster_signature)
```

Aynı job retry:

```text
duplicate face;
duplicate sample;
duplicate appearance
```

oluşturmamalı.

## APPEARANCE SEMANTICS

Canonical person’a ait bütün accepted detections PTS’ye göre sıralanır.

Yeni appearance interval:

source değişirse;
explicit scene/tracklet boundary varsa;
gap configured threshold’u aşarsa.

Hesap:

```text
firstSeen = min(appearance.start)
lastSeen = max(appearance.end)
totalDuration = sum(interval end - start)
```

lastSeen - firstSeen kullanılmaz.

Her detection tam olarak bir appearance interval’a ait olmalı.

Intervals:

sorted;
non-overlapping;
bounds valid;
en az bir detection içermeli.

VFR timestamp primary source PTS’dir.

## API SCHEMA

Backward-compatible final person:

```json
{
  "faceId": "face_rachel",
  "trackId": "vp_0001",
  "tracklets": [
    {
      "trackletId": "s0_tl_0007",
      "startFrame": 31,
      "endFrame": 184
    },
    {
      "trackletId": "s0_tl_0191",
      "startFrame": 7345,
      "endFrame": 7490
    }
  ],
  "status": "known",
  "name": "Rachel",
  "firstSeen": 1.29,
  "lastSeen": 312.42,
  "totalDuration": 12.87,
  "appearances": [],
  "bestShot": {},
  "confidence": 0.75,
  "detections": []
}
```

Confidence anlamları ayrı:

```text
detectionConfidence
recognitionSimilarity
recognitionMargin
finalConfidence
```

Raw cosine probability olarak gösterilmez.

## TWO-PASS FINAL RENDER

Canonical decisions EOS’tan sonra oluştuğundan final video ikinci pass ile render edilir.

Pass 1:

analysis + evidence

Reconciliation:

tracklet → canonical person map

Pass 2:

```text
NVDEC original
→ join final labels by frame/PTS+detection ID
→ nvdsosd
→ NVENC
```

Pass 2:

detector/recognizer tekrar çalıştırmaz;
OpenCV decode yapmaz;
full frame CPU’ya taşımaz;
input/output frame count aynı;
PTS monotonik;
clean EOS.

Final overlay:

```text
P01 | Rachel | sim:0.75 | det:0.99
```

Debug mode:

```text
P01/TL0191 | Rachel | ...
```

## CORE UNIT TESTS

Minimum C++ tests:

continuous IoU basics
invalid/zero-area bbox
single detection creates track
tentative activation
high-confidence first association
low-confidence second association
low detection cannot create track
lost then reactivated
lost timeout removal
ID never reused
deterministic equal-cost tie
two simultaneous faces never same ID
crossing trajectories
short occlusion
empty frame ages tracks
PTS gap changes prediction
excessive PTS gap terminates track
source states isolated
EOS flush
batch boundary does not alter result

CPU tracker tests ASan/UBSan ile de çalıştırılmalı.

## TEMPORAL BATCH TESTS

Aynı frozen detection input:

```text
batch=1
batch=2
batch=4
batch=8
```

için exact comparison:

same per-frame detection coverage;
same tracklet assignment;
same track lifecycle;
same ended tracklets;
same canonical reconciliation input.

Batch list’i kasıtlı shuffled fixture ile verilsin; adapter PTS/frame sort ile doğru sonucu üretmeli.

Duplicate/backward frame fixture açıkça fail etmeli.

## RECONCILIATION UNIT TESTS

Minimum:

same known person, distant non-overlapping tracklets → one video person
same known person at frame 1 and frame 7345 → same faceId/videoPersonId
two simultaneous different faces → cannot merge
strong conflicting known identities → conflict
unknown tracklets above complete-link threshold → merge
A-B, B-C, A-not-C chain → no three-way merge
existing anonymous match → existing faceId
new anonymous retry → one face only
multiple appearance intervals
totalDuration excludes gaps
firstSeen/lastSeen exact
best shot belongs to accepted detection
low-quality side face does not override frontal evidence
deterministic cluster ordering
model/gallery version mismatch fails

## REAL GOLDEN FIXTURE

Before running final test, freeze manual ground truth fixture containing:

Rachel early appearance;
Rachel distant later appearance;
Rachel + another cast member simultaneously;
side/profile Rachel;
short occlusion;
scene cut;
unknown/background face if available.

Expected evidence file must be committed without private dataset names in public logs where prohibited.

Metrics:

```text
duplicate track IDs per frame = 0
sentinel track IDs = 0
ID switches on labeled continuous tracklets
tracklet fragmentation
canonical person precision
erroneous merge count
firstSeen error
lastSeen error
appearance interval overlap
best-shot identity correctness
```

Full 6665-frame video için yalnız “unique track count” başarı metriği değildir.

## ACCEPTANCE TARGETS

```text
phase2-sprint-06-reference-check
phase2-sprint-06-native-build
phase2-sprint-06-core-unit
phase2-sprint-06-sanitizers
phase2-sprint-06-plugin-integration
phase2-sprint-06-temporal-order
phase2-sprint-06-batch-parity
phase2-sprint-06-tracklet-correctness
phase2-sprint-06-reconciliation-unit
phase2-sprint-06-known-reentry
phase2-sprint-06-unknown-clustering
phase2-sprint-06-appearance-aggregation
phase2-sprint-06-best-shot
phase2-sprint-06-second-pass-render
phase2-sprint-06-determinism
phase2-sprint-06-performance
phase2-sprint-06-acceptance
```

Acceptance 48-run performance matrisi değildir. Targeted correctness gates + bir short ve bir long E2E yeterlidir.

## HARD FAIL CONDITIONS

Aşağıdakilerden biri varsa PASS verme:

UNTRACKED_OBJECT_ID gerçek track olarak sayılıyor;
aynı frame’de duplicate tracklet ID;
batch 1/8 farklı assignments;
out-of-order frame sessizce kabul ediliyor;
low-confidence detection yeni track başlatıyor;
raw tracker ID public face identity olarak kullanılıyor;
scene-cut tracklet’leri yalnız zaman yakınlığıyla merge ediliyor;
different known identities merge ediliyor;
overlapping faces aynı canonical person oluyor;
totalDuration last-first hesaplanıyor;
best shot geçersiz detection’a referans veriyor;
Python/OpenCV production video decode kullanılıyor;
full frame D2H;
thresholds test sonucu görüldükten sonra gerekçesiz gevşetiliyor;
tracker/reconciler output’u deterministic değil.

## NON-GOALS

Bu sprintte:

```text
live RTSP;
multi-camera cross-camera tracking;
distributed tracker;
GPU Hungarian optimization;
neural scene-cut model;
final threshold calibration for every domain;
UI/API expansion
```

yapılmayacak.

## TOOL/MCP ACCOUNTABILITY

Zorunlu:

```text
codebase-memory-mcp:
existing caller/callee ve source insertion point keşfi
context7:
DeepStream/GStreamer metadata ve plugin lifecycle
deepwiki:
ByteTrack repository architecture; sonra actual pinned source doğrulaması
exa:
yalnız official/upstream eksikse
ASan/UBSan:
CPU tracker core
ffprobe:
final video integrity
Nsight:
full-frame D2H olmadığını doğrulama
```

21st ve Ruflo kullanılmayacak.

## IMPLEMENTATION ORDER

TDD sırası:

```text
Detection/tracklet schemas
Continuous IoU tests
Kalman tests
LAP deterministic assignment tests
ByteTrack state machine tests
Multi-source + PTS tests
Batch adapter tests
Plugin metadata tests
Recognition embedding adapter
Tracklet evidence writer
Python reconciler tests
Known merge
Unknown clustering
Appearance aggregation
Best shot
Second-pass renderer
Real short video
Real long video
Batch parity
Determinism/performance/hotpath
Review package
```

No git add/commit/push.
