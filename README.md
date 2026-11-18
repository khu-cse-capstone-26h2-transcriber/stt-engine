# stt-engine

Rubik Pi 3(QCS6490) NPU에서 한국어 음성을 실시간으로 받아쓰는 엔진입니다. `audiostackd`(audio-stack 저장소)에서 16 kHz 모노 오디오를 받아 partial/final 결과를 돌려보냅니다. 인터넷 연결 없이 동작합니다.

```text
mic → audiostackd ─AUDIO/SESSION_*→ stt-engine (NPU) ─RESULT/ENGINE_STATUS→ audiostackd ─HTTP→ web-app
```

- 모델: 한국어 Whisper-small [`seastar105/whisper-small-komixv2`](https://huggingface.co/seastar105/whisper-small-komixv2)을 QNN으로 변환함. 입력은 10초, 인코더 8bit 가중치/16bit 활성값, 디코더는 대칭 int8.
- 결과 형식과 세션 처리 규칙: [`ENGINE_CONTRACT.md`](ENGINE_CONTRACT.md). 데몬·웹 파트는 이 문서만 보면 됩니다.

## 보드 성능 (Ubuntu 24.04, QNN 2.50, CPU 클럭 최고 고정)

| 평가 (2026-11-02, 11-09) | CER | RTF 평균 / 최대 | final 지연 평균 / 최대 |
| --- | --- | --- | --- |
| TEDx 강연 41개, 실시간 연속 입력, 중간 결과 2초, `dec_10s.bin` (기본) | 5.5% | 0.115 / 0.131 | 0.98 / 2.96 s |
| 같은 조건, `dec_10s_pv.bin` | 5.5% | 0.105 / 0.121 | 0.90 / 2.89 s |
| 실제 `audiostackd`를 거침 (snd-aloop, talk04 162초) | 3.5% | - | 1.07 / 1.77 s |

잡음: 합성 잡음 10 dB에서 CER이 13.7~25.4%까지 오르지만 `--adapt`를 켜면 9.6~14.7%로 내려갑니다. 말이 없는 먼 말소리에서 지어낸 글자는 `--lp-min -0.5`로 94% 줄어듭니다. `run_engine.sh`는 둘 다 기본으로 켜고, `NOISE=0`이면 끕니다.

partial 안정화: 연속한 두 partial이 같게 낸 앞부분(단어 단위)을 확정해 다음 partial과 final의 앞에 고정하고, partial의 마지막 단어는 보내지 않습니다(`--stable final --hold-last`). 보드에서 강연 10개 기준 partial이 앞서 낸 글자를 바꾸는 비율이 53%→24%, 마지막 partial과 final이 다른 비율이 40%→19%로 줄었고, CER은 7.15%→6.94%, 지연은 그대로였습니다. `run_engine.sh`는 기본으로 켜고, `STABLE=0`이면 끕니다.

측정 방법과 원본 수치는 AI 파트 결과 문서(`results/board_*.md`)에 있습니다.

## 폴더 구성 (보드)

```text
~/rubikpi/
├─ runner/   ← 이 저장소
├─ bins/     ← 모델 (아래 '모델 받기')
└─ sdk250/   ← QNN 2.50 런타임 (gcc11.2/lib, dsp)
```

`run_stream.sh`가 `~/rubikpi/sdk250`을 라이브러리 경로로 쓰고, `board_stream.py`의 기본 `--bins`는 `~/rubikpi/bins`입니다.

## 저장소에 없는 것

| 항목 | 이유 | 얻는 방법 |
| --- | --- | --- |
| QNN 헤더 `include/*.h` | Qualcomm "Confidential and Proprietary" | QNN SDK 2.50의 `include/QNN/*.h`를 `include/`에 복사 |
| QNN 런타임 `sdk250/` | 같은 이유 | QNN SDK 2.50의 `lib/aarch64-ubuntu-gcc11.2`, `lib/hexagon-v68/unsigned` |
| 모델 `.bin` | 파일 하나가 107~223 MB라 GitHub 100 MB 제한을 넘음 | 이 저장소의 Release `v0.1`에 첨부 (아래) |

모델 파일은 Release `v0.1`에서 받아 `~/rubikpi/bins/`에 둡니다. 저장소가 비공개라 GitHub 로그인이 필요합니다(`gh auth login` 한 번).

```bash
mkdir -p ~/rubikpi/bins && cd ~/rubikpi/bins
gh release download v0.1 -R khu-cse-capstone-26h2-transcriber/stt-engine -p '*.bin' -p '*.vocab' -p SHA256SUMS
sha256sum -c --ignore-missing SHA256SUMS      # ONNX(Mac 시험용)는 보드에 필요 없음
```


| 파일 | 크기 | sha256 |
| --- | --- | --- |
| `enc_10s.bin` | 107 MB | `c2f1212136b51563439324e732883a774c2831b0cea9538fdfc2f32a2cfc9cec` |
| `dec_10s.bin` (전체 단어, 기본) | 223 MB | `cfb3f781820cdb6be6ef4f3c68b29b496fb1b380ab809176bdb6c779cc69577a` |
| `dec_10s_pv.bin` (출력 단어 5,557개) | 187 MB | `7a56f788fbc0f7775871cad8ad9959210bc2a35a9ab2b85fb46c432e2fe7dfbb` |
| `dec_10s_pv.vocab` (`dec_10s_pv.bin`과 함께) | 22 KB | `4451b1651b1514cbd606bf9eead5d594301769fef6d3cef877ec91a63e8045d8` |

## 설치 (보드에서)

```bash
sudo apt install -y g++ python3-numpy
sh build.sh                                  # whisper_npu, asr_cli, bin_info, libwhisper_asr.a
sudo sh set_governor.sh performance          # CPU 클럭 최고 고정 (재부팅하면 풀림, 측정값은 이 상태 기준)
```

엔진을 실행하는 사용자는 `audio` 그룹에 있어야 합니다. `audiostackd`가 `engine.sock`을 `root:audio 0660`으로 만들기 때문입니다.

## 실행

```bash
sh run_engine.sh                              # /run/audiostack/engine.sock 에 접속, 끊기면 다시 접속
ASTK_ENGINE_SOCKET=/tmp/astk/engine.sock DEC=dec_10s_pv.bin sh run_engine.sh
```

| 환경변수 | 기본값 | 뜻 |
| --- | --- | --- |
| `ASTK_ENGINE_SOCKET` | `/run/audiostack/engine.sock` | 데몬 소켓 |
| `DEC` | `dec_10s.bin` | 디코더 |
| `PARTIAL` | `2` | 중간 결과 간격(초), 0이면 끔 |
| `CUT` | `early` | 긴 말을 자르는 방식 |
| `NOISE` | `1` | 잡음 옵션 `--adapt --lp-min -0.5` (`0`이면 끔) |
| `STABLE` | `1` | partial 안정화 `--stable final --hold-last` (`0`이면 끔) |
| `EXTRA` | (없음) | 그 밖의 `board_stream.py` 옵션 |

서비스로 등록하는 방법: [`whisper-engine.service`](whisper-engine.service) 맨 위 주석.

## 시험

마이크와 데몬 없이 (WAV 파일):

```bash
python3 ~/audio-stack/tools/feed_wav.py some.wav --socket /tmp/astk/engine.sock &
ASTK_ENGINE_SOCKET=/tmp/astk/engine.sock sh run_engine.sh --once
```

프로토콜 자체 시험(가짜 데몬: 연결 끊기, 세션 교체, 세션 종료를 일부러 일으키고 검사 17개를 PASS/FAIL로 출력):

```bash
python3 tools/astk_selftest.py some_70s.wav --socket /tmp/astk-st/engine.sock --seconds 70 --drop-at 25 --rotate-at 45 &
ASTK_ENGINE_SOCKET=/tmp/astk-st/engine.sock sh run_engine.sh
```

`--idle-gap 15`를 더하면 세션 교체 때 15초 동안 AUDIO를 보내지 않습니다(`audiostackd --idle`에서 `session stop` 뒤 `session start`까지와 같음). 둘째 세션의 시각이 쉰 시간만큼 밀리지 않는지 봅니다.

Mac에서는 NPU 대신 ONNX로 같은 흐름을 돌릴 수 있습니다(동작 확인용이며, 정확도와 속도는 보드 값이 아님). ONNX 모델(양자화 전, 약 1.1 GB)도 Release `v0.1`에 있습니다:

```bash
pip install numpy onnxruntime
mkdir -p ~/onnx && gh release download v0.1 -R khu-cse-capstone-26h2-transcriber/stt-engine -p '*.onnx' -D ~/onnx
python3 ~/audio-stack/tools/feed_wav.py samples/fleurs_ko_30s.wav --socket /tmp/astk/engine.sock &
WHISPER_RUNNER="python3 fake_runner.py" python3 board_stream.py --astk /tmp/astk/engine.sock --once --bins ~/onnx
```

시험용 음성 `samples/fleurs_ko_30s.wav`(26초, 5문장, 정답 `.txt`)는 FLEURS(CC BY 4.0)에서 만들었습니다. 웹 화면까지 보는 방법은 web-app의 `docs/device-ingest.md`에 있습니다.

실제 `audiostackd` 연결 시험(ALSA snd-aloop + 백엔드 대역): `tools/astk_backend_stub.py`가 데몬의 HTTP POST를 기록하고, `tools/astk_loopback_check.py`가 세션별로 결과가 맞게 들어갔는지 확인합니다.

## 파일

| 파일 | 내용 |
| --- | --- |
| `board_stream.py` | 엔진 본체: 입력 → 링버퍼 → 구간 나누기 → NPU 인식 → partial/final |
| `segmenter.py` | 구간 나누기 (에너지 VAD, 단어 사이 자르기) |
| `board_asr.py` | NPU 실행기와 통신, 멜 계산, 토큰→글자 |
| `whisper_npu.cpp`, `whisper_asr.cpp/.h` | C++ NPU 실행기와 라이브러리 (`libwhisper_asr.a`, C++ 앱에서 직접 쓸 수 있음, 예제 `asr_cli.cpp`) |
| `astk_client.py` | audio-stack `client/astk_client.py`를 그대로 복사함 |
| `run_engine.sh`, `run_stream.sh` | 실행 스크립트 (빠른 코어 고정, 라이브러리 경로) |
| `mel_filters.*`, `vocab.bin`, `vocab_bytes.json` | 멜 필터와 토큰 표 (Whisper에서 가져옴) |
| `silero_vad.onnx` | 선택 사항인 Silero VAD (`--vad silero`, MIT) |
| `fake_runner.py` | Mac에서 쓰는 ONNX 대역 실행기 |
| `tools/` | 프로토콜 시험 도구 |
| `samples/` | 시험용 음성과 정답 (FLEURS) |
| `THIRD_PARTY.md` | 다른 프로젝트에서 가져온 파일과 라이선스 |

## 모델 출처와 라이선스 (2026-11-09 확인)

모델 파일(`.bin`)은 아래 모델을 QNN으로 변환하고 양자화한 것입니다. 변환한 모델을 배포할 때는 아래 출처를 함께 적습니다.

| 항목 | 라이선스·조건 | 근거 |
| --- | --- | --- |
| 기반 모델 `openai/whisper-small` | Apache-2.0 (Hugging Face 모델 카드), 코드 저장소 `openai/whisper`는 MIT | 각 모델 카드와 LICENSE |
| 파인튜닝 `seastar105/whisper-small-komixv2` | 모델 카드에 라이선스 표기 없음. 작성자 답변: "기반 모델과 AI Hub 학습 데이터셋의 제약 외에 추가로 부여한 라이선스는 없음" | Hugging Face 토론 [whisper-base-komixv2 #1](https://huggingface.co/seastar105/whisper-base-komixv2/discussions/1)(2026-10-22), 작성자가 small에도 같다고 답함 [whisper-small-komixv2 #1](https://huggingface.co/seastar105/whisper-small-komixv2/discussions/1)(2026-11-05) |
| 학습 데이터 (AI Hub 5종) | 학습해서 만든 모델은 영리·비영리로 자유롭게 쓰고 배포할 수 있음. 데이터셋 정식 명칭과 AI 허브(aihub.or.kr)를 출처로 표기해야 함. 원본 데이터 재배포는 금지 (이 저장소에는 원본 데이터가 없음) | [AI 허브 FAQ](https://www.aihub.or.kr/aihubnews/faq/list.do?currMenu=146&topMenu=104) '영리적(상업적) 목적으로 사용할 수 있나요?' |

출처 표기 (모델 학습 데이터, 모델 카드 기준):
이 모델은 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을 받아 구축된 AI 허브(aihub.or.kr) 데이터 "한국어 음성", "주소 음성 데이터", "주요 영역별 회의 음성인식 데이터", "저음질 전화망 음성인식 데이터", "방송 콘텐츠 대화체 음성인식 데이터"로 학습한 `seastar105/whisper-small-komixv2`(기반 `openai/whisper-small`)를 변환한 것입니다.

주의: 작성자의 답변은 모델 카드가 아니라 토론 댓글입니다. 작성자가 나중에 모델 카드에 라이선스를 넣으면 그것을 따릅니다. 이 저장소의 엔진 코드 자체의 라이선스(`LICENSE` 파일)는 아직 정하지 않았습니다.
