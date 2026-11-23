// Persistent runner for the Python tools (board_asr.py, board_stream.py): wraps wasr::Engine (whisper_asr.h)
// in a stdin/stdout protocol so the model loads once and the KV cache stays in memory.
//
// Run:  ./whisper_npu enc_10s.bin dec_10s.bin        (libraries via run.sh; assets next to this executable)
// Env:  WHISPER_PIN=0 -> do not pin to the fast cores; WHISPER_ZC=1 -> zero-copy buffers; WHISPER_VOCAB=file -> logits index -> token id table (pruned decoder)
//       WHISPER_PRE=file -> prefix decoder .bin (4 start tokens in one call, e.g. dec_10s_pv_pre.bin)
// Protocol (little-endian, binary):
//   'E' + float32[80 * T]                        -> float32 encoder_ms
//   'D' + int32 n + int32[n] prefix + int32 max_new -> int32 count + int32[count] tokens + float32 total_ms + float32 npu_ms + float32 avg_logprob
//   'G' + float32 seconds -> (no reply) audio length for the following decodes: turns on the repeat stop
//   'A' + uint8 on -> (no reply) following decodes may be cut short by SIGUSR1 (on=1) or not (on=0); clears a pending
//                     SIGUSR1. board_stream.py sends A1 before a partial and A0 before a final: when a final arrives
//                     while a partial is decoding, it sends SIGUSR1 and the partial stops after its current step.
//   'Q' -> quit
// Diagnostics go to stderr.
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <csignal>
#include <unistd.h>
#include "whisper_asr.h"

static volatile sig_atomic_t g_abort = 0;   // set by SIGUSR1
static void on_usr1(int) { g_abort = 1; }
static bool rd(void* p, size_t n) {   // fread that resumes after a signal interrupted the read
  size_t got = 0;
  while (got < n) {
    size_t r = fread((char*)p + got, 1, n - got, stdin); got += r;
    if (got < n) { if (ferror(stdin) && errno == EINTR) { clearerr(stdin); continue; } return false; }
  }
  return true;
}
static FILE* g_out;   // the protocol pipe (see main)
static void wr(const void* p, size_t n) { fwrite(p, 1, n, g_out); }

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s encoder.bin decoder.bin\n", argv[0]); return 1; }
  // Replies go to a private copy of stdout; fd 1 itself is pointed at stderr so that anything a library prints
  // to stdout cannot corrupt the binary protocol (it shows up in the log instead).
  g_out = fdopen(dup(1), "wb"); dup2(2, 1);
  const bool trace = getenv("WHISPER_TRACE") != nullptr;
  std::string exe = argv[0], dir = exe.find('/') == std::string::npos ? "." : exe.substr(0, exe.rfind('/'));
  wasr::Options opt;
  opt.zero_copy = getenv("WHISPER_ZC") && atoi(getenv("WHISPER_ZC"));
  if (getenv("WHISPER_VOCAB")) opt.vocab_map = getenv("WHISPER_VOCAB");
  if (getenv("WHISPER_PRE") && *getenv("WHISPER_PRE")) opt.prefix_bin = getenv("WHISPER_PRE");
  if (getenv("WHISPER_PIN")) opt.pin_fast_cores = atoi(getenv("WHISPER_PIN"));
  try {
    wasr::Engine asr(argv[1], argv[2], dir, opt);
    const size_t mel_n = 80 * (size_t)asr.window_seconds() * 100;
    fflush(stderr);
    struct sigaction sa {}; sa.sa_handler = on_usr1; sa.sa_flags = SA_RESTART; sigaction(SIGUSR1, &sa, nullptr);
    char cmd; float audio_s = 0; bool abortable = false;
    while (rd(&cmd, 1)) {
      if (trace) fprintf(stderr, "[cmd] %c\n", cmd);
      if (cmd == 'Q') break;
      if (cmd == 'G') { if (!rd(&audio_s, 4)) break; continue; }
      if (cmd == 'A') { uint8_t on; if (!rd(&on, 1)) break; abortable = on; g_abort = 0; continue; }
      if (cmd == 'E') {
        std::vector<float> mel(mel_n);
        if (!rd(mel.data(), mel.size() * 4)) break;
        float ms = (float)asr.encode(mel.data()); wr(&ms, 4); fflush(g_out);
        if (trace) fprintf(stderr, "[done] E %.0f ms\n", ms);
      } else if (cmd == 'D') {
        int32_t n, max_new; if (!rd(&n, 4)) break;
        std::vector<int32_t> pre(n); if (!rd(pre.data(), 4 * n) || !rd(&max_new, 4)) break;
        double tot, npu;
        auto out = asr.decode(std::vector<int>(pre.begin(), pre.end()), max_new, &tot, &npu, audio_s,
                              abortable ? (const volatile int*)&g_abort : nullptr);
        int32_t c = (int32_t)out.size(); float t = (float)tot, np = (float)npu, lp = asr.last_avg_logprob();
        std::vector<int32_t> o(out.begin(), out.end());
        wr(&c, 4); wr(o.data(), 4 * c); wr(&t, 4); wr(&np, 4); wr(&lp, 4); fflush(g_out);
        if (trace) fprintf(stderr, "[done] D %d tokens, %.0f ms\n", c, t);
      } else { fprintf(stderr, "error: unknown command byte 0x%02x (protocol out of sync)\n", (unsigned char)cmd); return 1; }
    }
  } catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
  return 0;
}
