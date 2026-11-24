# STT 엔진 ↔ audiostackd 약속 (엔진 쪽)

작성: 2026-11-09 · AI 파트

이 문서는 엔진(`board_stream.py --astk`, 실행은 `run_engine.sh`)이 audio-stack의 ASTK 프로토콜 v1(`audio-stack/docs/PROTOCOL.md`) 위에서 **실제로 하는 일**을 적습니다. 데몬과 웹 백엔드는 이 문서만 보고 엔진 결과를 해석할 수 있어야 합니다.

## 1. 연결

- 엔진은 클라이언트입니다. 모델을 다 불러온 뒤 `engine.sock`(기본 `/run/audiostack/engine.sock`, 환경변수 `ASTK_ENGINE_SOCKET`)에 접속하고, 접속하자마자 `ENGINE_STATUS {"event":"ready"}`를 보냅니다.
- 연결이 끊기면 모델을 그대로 둔 채 다시 접속을 시도합니다. 0.5초 간격으로, 데몬이 다시 뜰 때까지 계속 시도합니다. `--once`를 주면 처음 연결이 끝날 때 종료합니다(WAV 시험용).
- 끊긴 동안 보내지 못한 RESULT와 STATUS는 순서대로 최대 512개까지 보관했다가 다시 접속하면 보냅니다.
- 인식이 실패하면(예: NPU 실행기 `whisper_npu`가 죽음) `ENGINE_STATUS {"event":"error"}`를 보내고 종료 코드 2로 끝납니다. 다시 띄우는 일은 systemd(`whisper-engine.service`, `Restart=always`)가 맡습니다.

## 2. 데몬이 보내는 것을 엔진이 처리하는 방법

| 프레임 | 엔진 처리 |
| --- | --- |
| `HELLO` | `format.rate`가 16000이 아니면 종료 |
| `AUDIO` | 링버퍼에 넣고 구간을 나눕니다. **`cached` 표시가 있고, 이미 받은 오디오보다 `t_us`가 앞서면 버립니다**(다시 접속할 때 데몬이 같은 오디오를 다시 보내므로). `silence` 표시는 쓰지 않습니다(엔진이 자체 에너지 VAD를 씀) |
| `SESSION_BEGIN` | 새 세션을 시작합니다. 페이로드에 `t_us`(세션 첫 샘플의 녹음 시각, audio-stack `32124c7`부터)가 있으면 그 시각을 세션의 0으로 씁니다. 이미 열려 있는 세션과 번호가 같으면 무시합니다(데몬은 접속할 때마다 보냄). 앞 세션이 `SESSION_END` 없이 남아 있으면(데몬 재시작) 앞 세션을 먼저 닫습니다 |
| `SESSION_END` | 말하던 구간을 final로 내고, 그 세션의 마지막 final 뒤에 `session_done`을 보냅니다 |
| `GOODBYE` | 기록만 합니다. `--once`일 때는 남은 결과를 보낸 뒤 종료합니다 |
| `PING` | `PONG`으로 자동 응답합니다 |

## 3. 엔진이 보내는 것

### RESULT (0x20)

```json
{"type": "final", "text": "제가 문을 열고 들어가서 할머니가 누워계시는", "lang": "ko",
 "session_id": "sess-20261109-021500-4f2a", "seg_id": 3,
 "t_start_ms": 16770, "t_end_ms": 20340, "t_start_us": 792999382656, "t_end_us": 793002952656}
```

| 필드 | 뜻 |
| --- | --- |
| `type` | `partial`(중간 결과) 또는 `final`(확정 결과) |
| `text` | 구간의 텍스트. 띄어쓰기와 문장부호(`.`, `?`)가 들어가고, 숫자는 아라비아 숫자로 나올 때가 많습니다(`1839년 8월 19일`) |
| `session_id` | **이 결과가 속한 세션**. 엔진이 받은 오디오 기준이라 데몬이 결과를 받은 시점의 세션과 다를 수 있습니다(5절) |
| `seg_id` | 세션마다 0부터 1씩 늘어납니다. 같은 구간의 partial과 final은 같은 번호를 씁니다 |
| `t_start_ms`, `t_end_ms` | 그 세션의 첫 오디오 샘플을 0으로 둔 구간 시작·끝(ms). 데몬의 녹음 시각(`AUDIO.t_us`)으로 계산하므로 다시 접속해도 어긋나지 않습니다. `AUDIO.t_us`는 `SESSION_BEGIN`에 `t_us`가 오면 조각의 첫 샘플 시각으로, 없으면(예전 데몬) 조각 끝 시각으로 읽습니다 |
| `t_start_us`, `t_end_us` | 같은 구간의 녹음 시각(데몬의 `CLOCK_MONOTONIC` µs). 지연 측정에 씁니다 |
| `empty` | final에만 붙고 값은 `true`입니다. 구간에서 글자가 나오지 않았다는 뜻입니다(지어낸 말을 걸러냈거나, 겹침을 지우고 나니 남은 글자가 없음). `text`는 `""`이고, **이 구간의 partial을 화면에서 지우라는 신호**입니다 |
| `stable_chars` | partial에만 붙습니다(안정화를 켰을 때). `text` 앞에서 이 글자 수만큼은 확정이라 같은 구간의 뒤 partial과 final에서 바뀌지 않습니다. 화면에서 확정 부분과 나머지를 다르게 보여 줄 때 씁니다 |
| `lang` | 항상 `ko` |

규칙:
- 구간마다 final은 정확히 한 번 나갑니다(`empty` 포함). final 뒤에는 그 구간의 partial이 오지 않습니다.
- partial은 구간이 1.5초 이상 길어진 뒤 2초마다(`PARTIAL`) **구간 처음부터 다시 받아쓴 전체 텍스트**를 보냅니다. 앞서 나온 단어가 뒤에서 바뀔 수 있습니다(`stable_chars` 앞부분은 바뀌지 않음). 안정화를 켜면 마지막 단어는 다음 partial이나 final에 나옵니다. 인식이 밀리면 오래된 partial은 건너뜁니다. 글자가 없는 partial은 보내지 않습니다.
- 구간은 말이 멈춘 곳에서 끊습니다. 쉬지 않고 6초를 넘기면 그 뒤 처음 나오는 단어 사이에서 끊고(`--cut early`), 길어도 10초입니다.
- 열린 세션이 없을 때(`SESSION_END` 뒤, 다음 `SESSION_BEGIN` 전) 나온 결과는 보내지 않습니다.

### ENGINE_STATUS (0x21)

| `event` | 언제 | 다른 필드 |
| --- | --- | --- |
| `ready` | 접속할 때마다(모델은 이미 불러온 상태) | `model`, `backend`, `decoder`, `window_s`, `partial_s`, `pid` |
| `session_done` | `SESSION_END`를 받고, 그 세션의 마지막 final을 보낸 뒤 | `session_id`, `finals`(그 세션에서 보낸 final 개수) |
| `error` | 인식이 실패해 엔진을 종료하기 직전 | `message`, `session_id` |

## 4. 확인한 것

| 확인 | 환경 | 결과 |
| --- | --- | --- |
| 연결이 끊긴 뒤 다시 접속, cached 오디오 다시 받기, 세션 교체, 세션 종료 (`astk_selftest.py`, 검사 17개) | Mac, ONNX(`fake_runner.py`) | 모두 통과. 연결을 끊지 않은 실행과 final 텍스트·구간이 같음(편집 거리 1, 세션 교체로 끊긴 곳) |
| audio-stack `tools/feed_wav.py` → 엔진 `--once` | Mac | final 6개와 `session_done`까지 전달됨 |
| 실행기가 죽었을 때 | Mac | `error` 상태를 보낸 뒤 종료 코드 2 |
| 말이 없는 소리(`--lp-min -0.5`) | Mac | 빈 final 10개, 모두 `empty: true` |
| 같은 자체 시험 (talk04 70초, 실시간) | 보드 NPU | 모두 통과. final 12개, 지연 평균 0.91 s / 최대 2.13 s, `session_done`은 `SESSION_END` 뒤 0.70 s |
| `feed_wav.py` → 엔진 `--once` (talk04 60초) | 보드 NPU | final 11개와 `session_done (finals 11)` 전달, 종료 코드 0 |
| 실제 `audiostackd` (ALSA snd-aloop로 talk04 162초 + 데몬 재시작 후 talk05 30초, 세션 교체 4번) | 보드 NPU | 녹음 조각 8914개 중 버림 0. 세션 6개, final 34개, `seg_id` 세션마다 0..n. CER 3.5%, 지연 평균 1.07 s / 최대 1.77 s. 데몬 재시작 뒤 엔진이 다시 접속. **세션 교체 직후 앞 세션 결과 3건이 새 세션으로 들어가고, final 2건이 `/end` 뒤에 도착함** (5절 요청 1, 2) |

Mac에서 확인한 것은 프로토콜 동작뿐이고, 정확도와 속도는 보드 값만 결과로 씁니다. 보드 측정의 방법과 원본: AI 파트 `results/board_2026-11-09.md` 6절.

## 5. 다른 파트에 요청할 것

### audio-stack(하드웨어)

1. **결과에 붙이는 세션 번호** (`src/daemon.cpp:483`): 지금은 RESULT를 받은 순간 열려 있는 세션 번호를 URL과 `session_id`에 붙입니다. `session rotate` 직후에 도착한 앞 세션의 마지막 final이 새 세션으로 들어가고, 종료 뒤에 도착한 final은 빈 번호로 갑니다. 보드에서 세션 교체 4번 중 2번에서 실제로 일어났습니다(partial 1, final 2). 엔진은 `transcript.session_id`에 올바른 세션을 넣고 있으니, 이 값이 있으면 그것을 쓰도록 바꿔 주세요.
2. **세션 종료 순서**: `SESSION_END`를 보낸 뒤 엔진의 `session_done`을 받으면(늦어도 5초) 그때 `/end`를 POST해 주세요. 지금은 곧바로 `/end`를 보내서, 마지막 final이 `/end` 뒤에 도착합니다(보드에서 +0.20 s, +0.40 s). 보드에서 `session_done`은 `SESSION_END` 뒤 0.7 s 안에 왔습니다.
3. **`ENGINE_STATUS` 전달**: 지금은 로그에만 남습니다. `ready`와 `error`를 백엔드에 전달하거나 `astk-ctl status`에 넣어 주세요(웹 질문 B5, F1).
4. **문서**: `PROTOCOL.md`에서 `t_start_ms`, `t_end_ms`가 "segment-relative"로 적혀 있습니다. 엔진은 세션 시작 기준으로 보냅니다.
5. **엔진을 새로 띄운 경우의 시각**: 반영됨(audio-stack `32124c7`의 `SESSION_BEGIN.t_us`). 엔진은 이 값을 세션 시작으로 쓰고, 없으면 예전처럼 받은 첫 오디오 샘플을 씁니다.

### web-app(앱)

- `segmentId`는 서버가 붙이는 안(T19)과 엔진의 `seg_id` 중 하나를 고르면 됩니다. `seg_id`는 세션마다 0부터 시작하고 partial과 final이 같은 번호를 쓰므로 그대로 써도 됩니다.
- `empty: true`인 final은 버리지 말고 "이 구간의 partial 지움"으로 처리해 주세요.
- 시각은 `t_start_ms`, `t_end_ms`를 그대로 `startMs`, `endMs`로 쓰면 됩니다(세션 시작 기준 ms).

## 6. T19 '박지윤(AI)' 질문에 대한 답

| 질문 | 답 |
| --- | --- |
| 모델 파일 형식, 엔진 코드, CPU 실행 | 보드 NPU용 QNN context binary(`enc_10s.bin`, `dec_10s.bin`). 엔진 코드는 이 저장소에 있습니다. CPU 실행은 Mac 시험용 `fake_runner.py`(ONNX Runtime)만 있고, 정확도와 속도는 보드 값과 다릅니다 |
| 실제 결과 예시(H2) | 3절 예시. 보드에서 실제 데몬을 거쳐 백엔드가 받은 POST는 아래 7절 |
| partial이 누적인지(E2) | 누적입니다. 구간 처음부터 다시 받아쓴 전체 텍스트이고, 앞 단어가 바뀔 수 있습니다 |
| final이 구간마다 한 번이고, 그 뒤 partial이 없는지(E6, E9) | 네 |
| 세션 기준 시각(E7) | 네, ms입니다. 4절의 한계(5번)가 있습니다 |
| 빈 `text`(E8) | final에서만 `empty: true`와 함께 옵니다 |
| WAV를 넣는 도구(H1) | audio-stack `tools/feed_wav.py`가 그대로 동작합니다. 엔진은 `--once`로 실행합니다 |
| `SESSION_END` 뒤 끝 신호(C3) | `session_done` |
| 다시 접속할 때 중복 문장(F2) | 엔진이 살아 있는 동안은 이미 받은 오디오를 버리므로 중복이 없습니다. 엔진 프로세스를 새로 띄우면 최근 10초를 다시 받아쓰므로, 죽기 직전에 보낸 문장과 겹칠 수 있습니다 |
| 엔진을 다시 켰을 때 시각 기준 | 5절의 audio-stack 요청 5번 |

## 7. 보드에서 실제 데몬을 거쳐 백엔드가 받은 POST (2026-11-09)

```json
{"session_id": "sess-20261108-212659-119d", "seq": 2, "received_at": "2026-11-08T21:27:04Z", "audio_t_us": 20134112571,
 "transcript": {"type": "final", "text": "지금도 여러분들이 카메라를 가지고 계시고 사진을 찍고 계시죠.", "lang": "ko",
  "t_start_ms": 0, "t_end_ms": 3620, "session_id": "sess-20261108-212659-119d", "seg_id": 0,
  "t_start_us": 20129541470, "t_end_us": 20133171262}}
```
