// Korean Whisper (komixv2) speech-to-text on the Rubik Pi 3 NPU (QCS6490, QNN HTP context binaries).
//
// Usage:
//   wasr::Engine asr("bins/enc_10s.bin", "bins/dec_10s.bin", "runner");   // loads once (~0.3 s)
//   std::string text = asr.transcribe(pcm, n);   // pcm: mono 16 kHz float in [-1, 1], up to 10 s
//
// Assets in assets_dir: mel_filters.bin (201x80 float32), vocab.bin (token id -> UTF-8 bytes).
// Needs libQnnHtp.so / libQnnSystem.so on LD_LIBRARY_PATH and ADSP_LIBRARY_PATH set (see run.sh).
// One Engine per process (the QNN backend is process-wide). transcribe() is not thread-safe: call it from one thread.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace wasr {

struct Stats {
  double mel_ms = 0, enc_ms = 0, dec_ms = 0, npu_ms = 0, total_ms = 0;
  float avg_logprob = 0;  // average log probability of the chosen tokens: very low = probably not real speech
  int steps = 0;  // decoder runs (prefix + new tokens + end; the prefix counts 1 with Options::prefix_bin)
};

struct Options {
  bool zero_copy = false;      // shared rpcmem buffers: no per-step KV cache copies
  std::string vocab_map;       // pruned-vocab decoder: int32 table logits index -> token id (e.g. dec_10s_pv.vocab)
  int max_new_tokens = 120;
  bool pin_fast_cores = true;  // keep the calling thread on the fastest CPU cores (board: encoder stage 156 -> 88 ms)
  // Prefix decoder (scripts/export_prefix_decoder.py, e.g. dec_10s_pv_pre.bin): runs the 4 forced start tokens in one
  // NPU call instead of 4 (each call ~18 ms on the board, mostly reading the weights). Same text; empty = 4 step calls.
  std::string prefix_bin;
};

class Engine {
 public:
  Engine(const std::string& enc_bin, const std::string& dec_bin, const std::string& assets_dir, const Options& opt = {});
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Audio -> text. Audio longer than window_seconds() is cut.
  std::string transcribe(const float* pcm, size_t n, Stats* st = nullptr);

  int window_seconds() const;  // encoder input length (10 for enc_10s.bin)

  // Lower level steps (used by the stdin server whisper_npu).
  std::vector<float> mel(const float* pcm, size_t n) const;           // (80, window*100) row-major
  double encode(const float* mel);                                     // returns ms
  // audio_seconds > 0 turns on the repeat stop: once more than 5 tokens per second of audio have come out and the
  // last 5 tokens repeat an earlier run, decoding stops and that repeat is dropped (the model sometimes loops).
  // abort: if given and set (e.g. by a signal handler) decoding stops before the next step and returns what it has.
  std::vector<int> decode(const std::vector<int>& prefix, int max_new, double* total_ms, double* npu_ms, double audio_seconds = 0,
                          const volatile int* abort = nullptr);
  std::string detok(const std::vector<int>& ids) const;
  float last_avg_logprob() const;                                      // of the last decode()

  static const std::vector<int>& korean_prefix();  // <|startoftranscript|><|ko|><|transcribe|><|notimestamps|>

  struct Impl;
 private:
  Impl* p_;
};

}  // namespace wasr
