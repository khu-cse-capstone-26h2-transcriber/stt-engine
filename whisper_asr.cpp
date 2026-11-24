// Implementation of whisper_asr.h: log-mel (same as the HF Whisper feature extractor), QNN HTP encoder/decoder
// execution with the self KV cache kept on the device side, greedy decoding, token bytes -> UTF-8 text.
#include "whisper_asr.h"

#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#ifdef __linux__
#include <sched.h>
#endif

#include "QnnInterface.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "System/QnnSystemInterface.h"

namespace wasr {
namespace {

#define CHECK(x, msg) do { if ((x) != QNN_SUCCESS) throw std::runtime_error(std::string("QNN error: ") + (msg) + " (" + std::to_string((unsigned long)(x)) + ")"); } while (0)
using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

constexpr int SR = 16000, HOP = 160, NFFT = 400, NBIN = NFFT / 2 + 1, NMEL = 80, EOT = 50257;

QNN_INTERFACE_VER_TYPE qnn;
QNN_SYSTEM_INTERFACE_VER_TYPE qsys;
Qnn_BackendHandle_t backend;
Qnn_DeviceHandle_t device;
bool zc = false;
void* (*rpcmem_alloc_fn)(int, uint32_t, int);
void (*rpcmem_free_fn)(void*);
int (*rpcmem_to_fd_fn)(void*);

std::vector<uint8_t> read_file(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb"); if (!f) throw std::runtime_error("cannot open " + p);
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> v(n); size_t got = fread(v.data(), 1, n, f); fclose(f);
  if (got != (size_t)n) throw std::runtime_error("cannot read " + p);
  return v;
}

// One graph input/output and its buffer.
struct Buf {
  Qnn_Tensor_t t;
  std::string name;
  std::vector<uint8_t> mem;      // own storage when not zero-copy
  uint8_t* own = nullptr;        // own buffer (mem.data() or rpcmem)
  uint8_t* cur = nullptr;        // buffer currently bound to the tensor
  size_t bytes = 0, elems = 1, esize = 2;
  float scale = 1.f; int32_t offset = 0;  // real = scale * (q + offset)
  bool quant = false;
  uint8_t* data() { return own; }
};

size_t dtype_size(Qnn_DataType_t d) {
  switch (d) {
    case QNN_DATATYPE_UFIXED_POINT_8: case QNN_DATATYPE_SFIXED_POINT_8: case QNN_DATATYPE_UINT_8: case QNN_DATATYPE_INT_8: case QNN_DATATYPE_BOOL_8: return 1;
    case QNN_DATATYPE_UFIXED_POINT_16: case QNN_DATATYPE_SFIXED_POINT_16: case QNN_DATATYPE_UINT_16: case QNN_DATATYPE_INT_16: case QNN_DATATYPE_FLOAT_16: return 2;
    default: return 4;
  }
}

Buf make_buf(const Qnn_Tensor_t& src) {
  Buf b;
  b.t = src;                                   // name/dims pointers stay valid (system context is kept)
  auto& v = b.t.v1;                            // v2 starts with the same fields
  b.name = v.name;
  for (uint32_t i = 0; i < v.rank; i++) b.elems *= v.dimensions[i];
  b.esize = dtype_size(v.dataType);
  b.bytes = b.elems * b.esize;
  if (zc) {
    b.own = (uint8_t*)rpcmem_alloc_fn(25 /*RPCMEM_HEAP_ID_SYSTEM*/, 1 /*RPCMEM_DEFAULT_FLAGS*/, (int)b.bytes);
    if (!b.own) throw std::runtime_error("rpcmem_alloc failed for " + b.name);
    memset(b.own, 0, b.bytes);
  } else { b.mem.assign(b.bytes, 0); b.own = b.mem.data(); }
  b.cur = b.own;
  if (v.quantizeParams.encodingDefinition == QNN_DEFINITION_DEFINED &&
      v.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    b.quant = true;
    b.scale = v.quantizeParams.scaleOffsetEncoding.scale;
    b.offset = v.quantizeParams.scaleOffsetEncoding.offset;
  }
  v.memType = QNN_TENSORMEMTYPE_RAW;
  v.clientBuf.data = b.own;
  v.clientBuf.dataSize = (uint32_t)b.bytes;
  return b;
}

uint32_t quantize(const Buf& b, float x) {
  double q = std::round(x / b.scale - b.offset);
  double hi = b.esize == 1 ? 255 : 65535;
  return (uint32_t)std::min(std::max(q, 0.0), hi);
}

void put(Buf& b, size_t i, uint32_t q) {   // writes the tensor's own buffer
  if (b.esize == 1) b.own[i] = (uint8_t)q; else if (b.esize == 2) ((uint16_t*)b.own)[i] = (uint16_t)q; else ((uint32_t*)b.own)[i] = q;
}

// Same quantization (scales equal up to float rounding, e.g. self cache in/out calibrated to the same range).
bool same_enc(const Buf& a, const Buf& b) {
  return a.esize == b.esize && a.offset == b.offset && std::fabs(a.scale - b.scale) <= 1e-5f * std::fabs(b.scale);
}

// Copy src -> dst, requantizing if the encodings differ (lookup table per src/dst pair: 9 M values per encode).
// off: start element in src (a decoder fed per head takes one head's slice of the encoder's cross cache).
void requant(const Buf& src, Buf& dst, size_t off = 0) {
  if (same_enc(src, dst)) { memcpy(dst.own, src.cur + off * src.esize, dst.bytes); return; }
  static std::map<std::pair<const Buf*, const Buf*>, std::vector<uint32_t>> luts;
  auto& lut = luts[{&src, &dst}];
  if (lut.empty()) {
    lut.resize(src.esize == 1 ? 256 : 65536);
    for (size_t q = 0; q < lut.size(); q++) lut[q] = quantize(dst, src.scale * ((float)q + src.offset));
  }
  const uint8_t* s = src.cur + off * src.esize;
  if (src.esize == 2 && dst.esize == 2) {
    const uint16_t* a = (const uint16_t*)s; uint16_t* o = (uint16_t*)dst.own;
    for (size_t i = 0; i < dst.elems; i++) o[i] = (uint16_t)lut[a[i]];
  } else
    for (size_t i = 0; i < dst.elems; i++) put(dst, i, lut[src.esize == 1 ? s[i] : ((const uint16_t*)s)[i]]);
}

struct Graph {
  Qnn_ContextHandle_t ctx = nullptr;
  Qnn_GraphHandle_t g = nullptr;
  std::vector<Buf> in, out;
  std::map<std::string, int> in_idx, out_idx;
  std::map<uint8_t*, Qnn_MemHandle_t> mh;      // zero-copy: buffer -> handle registered in this context
  Buf& I(const std::string& n) { return in.at(in_idx.at(n)); }
  Buf& O(const std::string& n) { return out.at(out_idx.at(n)); }
  // Bind buffer p (own or borrowed, same shape) to tensor b without copying.
  void point(Buf& b, uint8_t* p) {
    b.cur = p;
    if (!zc) { b.t.v1.memType = QNN_TENSORMEMTYPE_RAW; b.t.v1.clientBuf.data = p; b.t.v1.clientBuf.dataSize = (uint32_t)b.bytes; return; }
    auto it = mh.find(p);
    if (it == mh.end()) {
      Qnn_MemDescriptor_t d; memset(&d, 0, sizeof d);
      d.memShape.numDim = b.t.v1.rank; d.memShape.dimSize = b.t.v1.dimensions; d.memShape.shapeConfig = nullptr;
      d.dataType = b.t.v1.dataType; d.memType = QNN_MEM_TYPE_ION; d.ionInfo.fd = rpcmem_to_fd_fn(p);
      Qnn_MemHandle_t h = nullptr;
      CHECK(qnn.memRegister(ctx, &d, 1, &h), ("memRegister " + b.name).c_str());
      it = mh.emplace(p, h).first;
    }
    b.t.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE; b.t.v1.memHandle = it->second;
  }
  void exec() {
    std::vector<Qnn_Tensor_t> ti, to;
    for (auto& b : in) ti.push_back(b.t);
    for (auto& b : out) to.push_back(b.t);
    CHECK(qnn.graphExecute(g, ti.data(), ti.size(), to.data(), to.size(), nullptr, nullptr), "graphExecute");
  }
};

void load(Graph& G, const std::string& path) {
  auto blob = read_file(path);
  auto t0 = clk::now();
  QnnSystemContext_Handle_t sc; CHECK(qsys.systemContextCreate(&sc), "systemContextCreate");
  const QnnSystemContext_BinaryInfo_t* info; Qnn_ContextBinarySize_t isz;
  CHECK(qsys.systemContextGetBinaryInfo(sc, blob.data(), blob.size(), &info, &isz), "getBinaryInfo");
  uint32_t ng; QnnSystemContext_GraphInfo_t* gs;
  switch (info->version) {
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: ng = info->contextBinaryInfoV1.numGraphs; gs = info->contextBinaryInfoV1.graphs; break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: ng = info->contextBinaryInfoV2.numGraphs; gs = info->contextBinaryInfoV2.graphs; break;
    default: ng = info->contextBinaryInfoV3.numGraphs; gs = info->contextBinaryInfoV3.graphs; break;
  }
  if (ng != 1) throw std::runtime_error(path + ": expected 1 graph");
  auto& gi = gs[0].graphInfoV1;               // V2/V3 start with the same fields
  for (uint32_t i = 0; i < gi.numGraphInputs; i++) { G.in.push_back(make_buf(gi.graphInputs[i])); G.in_idx[G.in.back().name] = i; }
  for (uint32_t i = 0; i < gi.numGraphOutputs; i++) { G.out.push_back(make_buf(gi.graphOutputs[i])); G.out_idx[G.out.back().name] = i; }
  CHECK(qnn.contextCreateFromBinary(backend, device, nullptr, blob.data(), blob.size(), &G.ctx, nullptr), "contextCreateFromBinary");
  CHECK(qnn.graphRetrieve(G.ctx, gi.graphName, &G.g), "graphRetrieve");
  for (auto* v : {&G.in, &G.out}) for (auto& b : *v) G.point(b, b.own);   // zero-copy: register own buffers
  fprintf(stderr, "[load] %s: %.0f ms, %zu inputs, %zu outputs\n", path.c_str(), ms_since(t0), G.in.size(), G.out.size());
}

// AI Hub context binaries name outputs output_0, output_1, ... in the source ONNX output order.
// Give them the source names (encoder: k/v_cache_cross_i; decoder and prefix decoder: logits, k/v_cache_self_i_out or _new)
// and check shapes.
void name_outputs(Graph& G, bool encoder) {
  for (size_t i = 0; i < G.out.size(); i++) {
    Buf& b = G.out[i];
    if (b.name.rfind("output_", 0) != 0) continue;           // already has a real name
    int k = std::stoi(b.name.substr(7));
    std::string n;
    if (encoder) n = std::string(k % 2 ? "v" : "k") + "_cache_cross_" + std::to_string(k / 2);
    auto* d = b.t.v1.dimensions; uint32_t r = b.t.v1.rank;
    // new-token-only self cache (scripts/self_new_only.py): one slot, K [H,1,D,1] / V [H,1,1,D] -> "<x>_new"
    if (!encoder) n = k == 0 ? "logits" : std::string((k - 1) % 2 ? "v" : "k") + "_cache_self_" + std::to_string((k - 1) / 2) +
                      (r == 4 && ((k - 1) % 2 ? d[2] : d[3]) == 1 ? "_new" : "_out");
    bool ok = n == "logits" ? (r == 4 && d[1] > 1000) : r == 4 && (n[0] == 'k' ? d[2] == 64 : d[3] == 64);
    if (!ok) throw std::runtime_error("output " + b.name + " -> " + n + ": unexpected shape");
    G.out_idx.erase(b.name); b.name = n; G.out_idx[n] = (int)i;
  }
}

void set_burst() {
  QnnDevice_Infrastructure_t inf = nullptr;
  if (qnn.deviceGetInfrastructure(&inf) != QNN_SUCCESS || !inf) { fprintf(stderr, "no perf infrastructure\n"); return; }
  auto* htp = (QnnHtpDevice_Infrastructure_t*)inf;
  auto& perf = htp->perfInfra;
  uint32_t id; if (perf.createPowerConfigId(0, 0, &id) != QNN_SUCCESS) { fprintf(stderr, "createPowerConfigId failed\n"); return; }
  QnnHtpPerfInfrastructure_PowerConfig_t c; memset(&c, 0, sizeof c);
  c.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  auto& d = c.dcvsV3Config;
  d.contextId = id; d.setDcvsEnable = 1; d.dcvsEnable = 0;
  d.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  d.setSleepLatency = 1; d.sleepLatency = 40; d.setSleepDisable = 1; d.sleepDisable = 1;
  d.setBusParams = 1; d.busVoltageCornerMin = d.busVoltageCornerTarget = d.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  d.setCoreParams = 1; d.coreVoltageCornerMin = d.coreVoltageCornerTarget = d.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  const QnnHtpPerfInfrastructure_PowerConfig_t* cs[] = {&c, nullptr};
  fprintf(stderr, "[perf] burst %s\n", perf.setPowerConfig(id, cs) == QNN_SUCCESS ? "on" : "FAILED");
}

template <class T> T get_iface(void* lib, const char* sym, bool system) {
  T res{};
  if (!system) {
    auto fn = (Qnn_ErrorHandle_t(*)(const QnnInterface_t***, uint32_t*))dlsym(lib, sym);
    const QnnInterface_t** p; uint32_t n; if (!fn || fn(&p, &n) != QNN_SUCCESS || !n) throw std::runtime_error(std::string("no ") + sym);
    memcpy(&res, &p[0]->QNN_INTERFACE_VER_NAME, sizeof res);
  } else {
    auto fn = (Qnn_ErrorHandle_t(*)(const QnnSystemInterface_t***, uint32_t*))dlsym(lib, sym);
    const QnnSystemInterface_t** p; uint32_t n; if (!fn || fn(&p, &n) != QNN_SUCCESS || !n) throw std::runtime_error(std::string("no ") + sym);
    memcpy(&res, &p[0]->QNN_SYSTEM_INTERFACE_VER_NAME, sizeof res);
  }
  return res;
}

}  // namespace

// ---------------------------------------------------------------- log-mel (no NPU)

struct MelTables {
  std::vector<float> filters;          // (201, 80)
  std::vector<double> win;             // Hann window (400)
  std::vector<std::complex<double>> tw;  // FFT twiddles exp(-2 pi i k / 400)
};

// Mixed-radix recursive FFT for n = 400 = 2^4 * 5^2 (out-of-place, stride access).
static void fft(const std::complex<double>* x, std::complex<double>* y, int n, int stride, const std::vector<std::complex<double>>& tw) {
  if (n == 1) { y[0] = x[0]; return; }
  int r = n % 5 == 0 ? 5 : n % 4 == 0 ? 4 : 2, m = n / r, tstep = NFFT / n;
  for (int j = 0; j < r; j++) fft(x + j * stride, y + j * m, m, stride * r, tw);   // sub-FFTs of x[j], x[j+r], ...
  std::complex<double> t[5];
  for (int k = 0; k < m; k++) {
    for (int j = 0; j < r; j++) t[j] = y[j * m + k] * tw[(j * k * tstep) % NFFT];
    for (int q = 0; q < r; q++) {
      std::complex<double> acc = 0;
      for (int j = 0; j < r; j++) acc += t[j] * tw[((j * q * m) % n) * tstep];
      y[q * m + k] = acc;
    }
  }
}

static MelTables make_mel_tables(const std::string& assets_dir) {
  MelTables m;
  auto f = read_file(assets_dir + "/mel_filters.bin");
  if (f.size() != NBIN * NMEL * 4) throw std::runtime_error("mel_filters.bin: wrong size");
  m.filters.resize(NBIN * NMEL); memcpy(m.filters.data(), f.data(), f.size());
  m.win.resize(NFFT);
  for (int k = 0; k < NFFT; k++) m.win[k] = (float)(0.5 - 0.5 * std::cos(2 * M_PI * k / NFFT));   // np.hanning(401)[:-1] as float32
  m.tw.resize(NFFT);
  for (int k = 0; k < NFFT; k++) m.tw[k] = std::polar(1.0, -2 * M_PI * k / NFFT);
  return m;
}

// Same as board_asr.log_mel: pad/cut to `seconds`, reflect pad 200, 400-pt Hann frames every 160, power, mel, log10,
// clamp to max-8, (x+4)/4.
static std::vector<float> log_mel(const MelTables& M, const float* pcm, size_t n, int seconds) {
  const int N = seconds * SR, T = N / HOP;
  std::vector<float> x(N + NFFT, 0.f);
  for (int i = 0; i < N; i++) x[NFFT / 2 + i] = i < (int)n ? pcm[i] : 0.f;
  for (int k = 1; k <= NFFT / 2; k++) { x[NFFT / 2 - k] = x[NFFT / 2 + k]; x[NFFT / 2 + N - 1 + k] = x[NFFT / 2 + N - 1 - k]; }
  std::vector<double> mel((size_t)NMEL * T);
  std::vector<std::complex<double>> fr(NFFT), sp(NFFT); std::vector<double> pw(NBIN);
  double mx = -1e30;
  for (int t = 0; t < T; t++) {
    const float* s = &x[(size_t)t * HOP];
    for (int k = 0; k < NFFT; k++) fr[k] = (double)(float)(s[k] * (float)M.win[k]);   // float32 product like numpy
    fft(fr.data(), sp.data(), NFFT, 1, M.tw);
    for (int b = 0; b < NBIN; b++) pw[b] = std::norm(sp[b]);
    for (int m = 0; m < NMEL; m++) {
      double v = 0;
      for (int b = 0; b < NBIN; b++) v += pw[b] * M.filters[b * NMEL + m];
      v = std::log10(std::max(v, 1e-10));
      mel[(size_t)m * T + t] = v; mx = std::max(mx, v);
    }
  }
  std::vector<float> out(mel.size());
  for (size_t i = 0; i < mel.size(); i++) out[i] = (float)((std::max(mel[i], mx - 8.0) + 4.0) / 4.0);
  return out;
}

// ---------------------------------------------------------------- engine

static std::vector<std::string> load_vocab(const std::string& path);
static std::string detok_bytes(const std::vector<std::string>& vocab, const std::vector<int>& ids);

// Pin the calling thread to the cores with the highest maximum clock (QCS6490: cpu4-7). The CPU-side work
// (mel, cache conversion, argmax) otherwise lands on the small cores between NPU calls. Once per thread.
static void pin_fast_cores() {
#ifdef __linux__
  static thread_local bool done = false;
  if (done) return;
  done = true;
  long f[64], lo = 0, hi = 0; int n = 0;
  for (; n < 64; n++) {
    FILE* fp = fopen(("/sys/devices/system/cpu/cpu" + std::to_string(n) + "/cpufreq/cpuinfo_max_freq").c_str(), "r");
    if (!fp) break;
    if (fscanf(fp, "%ld", &f[n]) != 1) f[n] = 0;
    fclose(fp);
    if (n == 0 || f[n] < lo) lo = f[n];
    if (f[n] > hi) hi = f[n];
  }
  if (n == 0 || hi == lo) return;   // no clock info or all cores equal
  cpu_set_t set; CPU_ZERO(&set);
  for (int i = 0; i < n; i++) if (f[i] > lo) CPU_SET(i, &set);
  if (sched_setaffinity(0, sizeof set, &set) != 0) fprintf(stderr, "[pin] sched_setaffinity failed\n");
#endif
}

struct Engine::Impl {
  MelTables mel;
  std::vector<std::string> vocab;   // token id -> bytes
  std::vector<int32_t> vmap;        // pruned decoder: logits index -> token id
  Graph E, D, Pre;                  // Pre: prefix decoder (optional, Pre.g == nullptr when not loaded)
  std::vector<std::pair<int, int>> self;  // self cache (out idx, in idx)
  std::vector<std::pair<int, int>> pre_self;  // prefix decoder self cache out idx -> step decoder in idx
  // Decoder with new-token-only self cache outputs (scripts/self_new_only.py): output "<x>_new" [H,1,D,1] (K) or
  // [H,1,1,D] (V) is written into slot (slots - 1 - step) of input "<x>_in" [H,1,D,slots] / [H,1,slots,D].
  // Attention does not depend on slot order, so this fills the same slots the shifting cache used, in reverse.
  struct NewKV { int out, in; size_t H, A, B; bool k; };
  std::vector<NewKV> self_new;
  struct Cross { Graph* g; int src, dst; size_t off; };
  std::vector<Cross> cross;  // encoder cross cache output (from element off) -> decoder input, copied after each encode
  int window = 10, max_new = 120, last_steps = 0;
  float last_lp = 0;   // average log probability of the chosen tokens in the last decode
  bool pin = true;
};

const std::vector<int>& Engine::korean_prefix() { static const std::vector<int> p = {50258, 50264, 50359, 50363}; return p; }

Engine::Engine(const std::string& enc_bin, const std::string& dec_bin, const std::string& assets_dir, const Options& opt) : p_(new Impl) {
  auto& P = *p_;
  P.max_new = opt.max_new_tokens; P.pin = opt.pin_fast_cores;
  P.mel = make_mel_tables(assets_dir);
  P.vocab = load_vocab(assets_dir + "/vocab.bin");
  if (!opt.vocab_map.empty()) { auto v = read_file(opt.vocab_map); P.vmap.resize(v.size() / 4); memcpy(P.vmap.data(), v.data(), v.size()); }

  void* h = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
  void* hs = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
  if (!h || !hs) throw std::runtime_error(std::string("dlopen QNN: ") + dlerror());
  if (opt.zero_copy) {
    void* rl = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (!rl) throw std::runtime_error(std::string("dlopen libcdsprpc.so: ") + dlerror());
    rpcmem_alloc_fn = (void* (*)(int, uint32_t, int))dlsym(rl, "rpcmem_alloc");
    rpcmem_free_fn = (void (*)(void*))dlsym(rl, "rpcmem_free");
    rpcmem_to_fd_fn = (int (*)(void*))dlsym(rl, "rpcmem_to_fd");
    auto init = (void (*)())dlsym(rl, "rpcmem_init"); if (init) init();
    if (!rpcmem_alloc_fn || !rpcmem_to_fd_fn) throw std::runtime_error("rpcmem symbols missing");
    zc = true;
  }
  fprintf(stderr, "[mode] %s\n", zc ? "zero-copy (rpcmem)" : "client buffers (copied each execute)");
  qnn = get_iface<QNN_INTERFACE_VER_TYPE>(h, "QnnInterface_getProviders", false);
  qsys = get_iface<QNN_SYSTEM_INTERFACE_VER_TYPE>(hs, "QnnSystemInterface_getProviders", true);
  CHECK(qnn.backendCreate(nullptr, nullptr, &backend), "backendCreate");
  CHECK(qnn.deviceCreate(nullptr, nullptr, &device), "deviceCreate");
  set_burst();
  load(P.E, enc_bin); load(P.D, dec_bin);
  name_outputs(P.E, true); name_outputs(P.D, false);
  if (!opt.prefix_bin.empty()) { load(P.Pre, opt.prefix_bin); name_outputs(P.Pre, false); }
  P.window = (int)(P.E.in[0].elems / NMEL / 100);

  // Encoder outputs (cross KV cache) -> decoder inputs; share memory when encodings match.
  // A decoder may take the cross cache per head (inputs <name>_h0.._h{H-1}, scripts/split_cross_heads.py):
  // each head is a contiguous slice of the encoder output (heads are the first axis).
  for (Graph* G : {&P.D, &P.Pre}) {
    if (!G->g) continue;
    for (size_t k = 0; k < P.E.out.size(); k++) {
      Buf& o = P.E.out[k];
      if (G->in_idx.count(o.name)) {
        Buf& di = G->I(o.name);
        if (same_enc(o, di) && o.bytes == di.bytes) { G->point(di, o.data()); continue; }
        fprintf(stderr, "  requant %s (enc %g/%d -> %s %g/%d)\n", o.name.c_str(), o.scale, o.offset,
                G == &P.D ? "dec" : "prefix", di.scale, di.offset);
        P.cross.push_back({G, (int)k, G->in_idx.at(o.name), 0});
        continue;
      }
      int H = 0; while (G->in_idx.count(o.name + "_h" + std::to_string(H))) H++;
      if (!H) throw std::runtime_error("encoder output " + o.name + " has no decoder input");
      if (o.elems % H) throw std::runtime_error(o.name + ": size not divisible by " + std::to_string(H) + " heads");
      for (int h = 0; h < H; h++) {
        int di = G->in_idx.at(o.name + "_h" + std::to_string(h));
        if (G->in[di].elems != o.elems / H) throw std::runtime_error(o.name + "_h: head size mismatch");
        P.cross.push_back({G, (int)k, di, (size_t)h * (o.elems / H)});
      }
    }
  }
  fprintf(stderr, "[wire] %zu cross cache copies per encode\n", P.cross.size());
  // Self cache: output "<x>_out" feeds input "<x>_in" on the next step.
  for (auto& o : P.D.out) {
    if (o.name.size() < 4 || o.name.compare(o.name.size() - 4, 4, "_out")) continue;
    P.self.push_back({P.D.out_idx.at(o.name), P.D.in_idx.at(o.name.substr(0, o.name.size() - 4) + "_in")});
  }
  for (auto& o : P.D.out) {
    if (o.name.size() < 4 || o.name.compare(o.name.size() - 4, 4, "_new")) continue;
    int ii = P.D.in_idx.at(o.name.substr(0, o.name.size() - 4) + "_in"); Buf& in = P.D.in[ii];
    auto* id = in.t.v1.dimensions; auto* od = o.t.v1.dimensions;
    if (in.t.v1.rank != 4 || o.t.v1.rank != 4 || id[0] != od[0]) throw std::runtime_error(o.name + ": unexpected shape");
    bool k = od[3] == 1;                       // K: [H,1,D,1] into [H,1,D,slots]; V: [H,1,1,D] into [H,1,slots,D]
    if (k ? od[2] != id[2] : (od[2] != 1 || od[3] != id[3])) throw std::runtime_error(o.name + ": shape does not match " + in.name);
    P.self_new.push_back({P.D.out_idx.at(o.name), ii, id[0], id[2], id[3], k});
  }
  if (!P.self_new.empty()) fprintf(stderr, "[wire] %zu self cache outputs are new-token only\n", P.self_new.size());
  int n_rq = 0; for (auto& s : P.self) n_rq += !same_enc(P.D.out[s.first], P.D.in[s.second]);
  fprintf(stderr, "[wire] %zu self cache pairs, %d need requant per step\n", P.self.size(), n_rq);
  if (P.Pre.g) {   // prefix decoder: its self cache outputs (199 slots = after 4 steps) feed the step decoder's inputs
    if (P.Pre.I("input_ids").elems != korean_prefix().size()) throw std::runtime_error("prefix decoder: input_ids must hold the 4 prefix tokens");
    if (P.Pre.O("logits").elems != P.D.O("logits").elems) throw std::runtime_error("prefix decoder: logits size != step decoder");
    for (auto& o : P.Pre.out) {
      if (o.name.size() < 4 || o.name.compare(o.name.size() - 4, 4, "_out")) continue;
      int di = P.D.in_idx.at(o.name.substr(0, o.name.size() - 4) + "_in");
      if (o.bytes != P.D.in[di].bytes) throw std::runtime_error("prefix decoder: " + o.name + " size != step decoder input");
      P.pre_self.push_back({P.Pre.out_idx.at(o.name), di});
    }
    for (auto& b : P.Pre.in) {   // fixed inputs: zero self caches (196 slots), the prefix tokens, positions, causal mask
      if (b.name.find("_cache_self_") != std::string::npos) { uint32_t z = quantize(b, 0.f); for (size_t i = 0; i < b.elems; i++) put(b, i, z); }
    }
    Buf& ids = P.Pre.I("input_ids"); Buf& pos = P.Pre.I("position_ids"); Buf& m = P.Pre.I("attention_mask");
    const size_t np = korean_prefix().size(), ctx = m.elems / np;
    for (size_t i = 0; i < np; i++) { ((int32_t*)ids.data())[i] = korean_prefix()[i]; ((int32_t*)pos.data())[i] = (int32_t)i; }
    uint32_t qneg = quantize(m, -1e9f), qzero = quantize(m, 0.f);
    for (size_t i = 0; i < np; i++)      // token i sees the slots of tokens 0..i, which sit at the end of the window
      for (size_t j = 0; j < ctx; j++) put(m, i * ctx + j, j >= ctx - np && j <= ctx - np + i ? qzero : qneg);
    int n_prq = 0; for (auto& s : P.pre_self) n_prq += !same_enc(P.Pre.out[s.first], P.D.in[s.second]);
    fprintf(stderr, "[wire] prefix decoder: %zu self cache pairs, %d need requant\n", P.pre_self.size(), n_prq);
  }
  Buf& logits = P.D.O("logits");
  if (!P.vmap.empty() && P.vmap.size() != logits.elems) throw std::runtime_error("vocab map size != decoder logits");
  if (P.vmap.empty() && logits.elems != P.vocab.size()) throw std::runtime_error("pruned decoder needs Options::vocab_map");
  fprintf(stderr, "[ready] window %d s, %zu self cache pairs, logits %zu\n", P.window, P.self.size(), logits.elems);
}

Engine::~Engine() {
  if (p_->E.ctx) qnn.contextFree(p_->E.ctx, nullptr);
  if (p_->D.ctx) qnn.contextFree(p_->D.ctx, nullptr);
  if (p_->Pre.ctx) qnn.contextFree(p_->Pre.ctx, nullptr);
  delete p_;
}

int Engine::window_seconds() const { return p_->window; }

std::vector<float> Engine::mel(const float* pcm, size_t n) const { if (p_->pin) pin_fast_cores(); return log_mel(p_->mel, pcm, n, p_->window); }

double Engine::encode(const float* mel) {
  auto& P = *p_;
  if (P.pin) pin_fast_cores();
  auto t = clk::now();
  Buf& x = P.E.in[0];
  for (size_t i = 0; i < x.elems; i++) put(x, i, x.quant ? quantize(x, mel[i]) : 0);
  P.E.exec();
  for (auto& c : P.cross) requant(P.E.out[c.src], c.g->in[c.dst], c.off);
  return ms_since(t);
}

std::vector<int> Engine::decode(const std::vector<int>& pre, int max_new, double* total_ms, double* npu_ms, double audio_s,
                                 const volatile int* abort) {
  auto& P = *p_; auto& D = P.D;
  if (P.pin) pin_fast_cores();
  auto t = clk::now(); double npu = 0;
  Buf& ids = D.I("input_ids"); Buf& mask = D.I("attention_mask"); Buf& pos = D.I("position_ids"); Buf& logits = D.O("logits");
  const size_t ctx = mask.elems;
  for (auto& s : P.self) {   // self cache to quantized zero
    Buf& in = D.in[s.second]; D.point(in, in.data());
    uint32_t z = quantize(in, 0.f); for (size_t i = 0; i < in.elems; i++) put(in, i, z);
  }
  for (auto& s : P.self_new) {   // same for caches filled slot by slot
    Buf& in = D.in[s.in]; D.point(in, in.data());
    uint32_t z = quantize(in, 0.f); for (size_t i = 0; i < in.elems; i++) put(in, i, z);
  }
  uint32_t qneg = quantize(mask, -1e9f), qzero = quantize(mask, 0.f);   // "masked" clamps to the lowest value
  for (size_t i = 0; i < ctx; i++) put(mask, i, qneg);
  std::vector<int> outv; int tok = pre[0], step = 0, runs = 0; bool go = true;
  auto argmax = [](const Buf& L) {   // over quantized logits (monotonic when scale > 0)
    size_t best = 0;
    if (L.esize == 2) { auto* v = (const uint16_t*)L.cur; for (size_t i = 1; i < L.elems; i++) if (v[i] > v[best]) best = i; }
    else if (L.esize == 1) { auto* v = L.cur; for (size_t i = 1; i < L.elems; i++) if (v[i] > v[best]) best = i; }
    else { auto* v = (const float*)L.cur; for (size_t i = 1; i < L.elems; i++) if (v[i] > v[best]) best = i; }
    return best;
  };
  double lp_sum = 0; int lp_n = 0;
  auto add_lp = [&](const Buf& L, size_t best) {   // log P(best) = -log(sum exp(z_i - z_best)); values over 16 nats lower add nothing
    double sum = 1;
    if (L.esize == 4) { auto* v = (const float*)L.cur; const float b = v[best]; for (size_t i = 0; i < L.elems; i++) { float d = b - v[i]; if (d < 16.f && i != best) sum += std::exp(-(double)d); } }
    else if (L.esize == 2) { auto* v = (const uint16_t*)L.cur; const int b = v[best], w = (int)(16.f / L.scale); for (size_t i = 0; i < L.elems; i++) { int d = b - v[i]; if (d < w && i != best) sum += std::exp(-(double)d * L.scale); } }
    else { auto* v = L.cur; const int b = v[best], w = (int)(16.f / L.scale); for (size_t i = 0; i < L.elems; i++) { int d = b - v[i]; if (d < w && i != best) sum += std::exp(-(double)d * L.scale); } }
    lp_sum -= std::log(sum); lp_n++;
  };
  auto take = [&](size_t best) {   // next token from the logits; false = stop
    tok = P.vmap.empty() ? (int)best : P.vmap[best];
    if (tok == EOT || (int)outv.size() >= max_new) return false;
    outv.push_back(tok);
    const size_t RN = 5, n = outv.size();   // repeat stop (see whisper_asr.h)
    if (audio_s > 0 && n >= 2 * RN && n > audio_s * 5.0) {
      bool rep = false;
      for (size_t i = 0; i + 2 * RN <= n && !rep; i++) rep = std::equal(outv.end() - RN, outv.end(), outv.begin() + i);
      if (rep) { outv.resize(n - RN); return false; }
    }
    return true;
  };
  const auto& kp = korean_prefix();
  if (P.Pre.g && pre.size() >= kp.size() && std::equal(kp.begin(), kp.end(), pre.begin())) {
    // the 4 start tokens in one call (inputs were set at load); further forced tokens (e.g. committed partial text)
    // follow as normal steps below
    auto te = clk::now(); P.Pre.exec(); npu += ms_since(te); runs++;
    for (auto& s : P.pre_self) requant(P.Pre.out[s.first], D.in[s.second]);   // memcpy when encodings match
    for (step = 0; step < (int)kp.size(); step++) put(mask, ctx - step - 1, qzero);
    if (pre.size() > kp.size()) tok = pre[kp.size()];   // next forced token
    else { size_t b0 = argmax(P.Pre.O("logits")); add_lp(P.Pre.O("logits"), b0); go = take(b0); }
  }
  for (; go && step < (int)ctx - 1; step++) {
    if (abort && *abort) break;   // the caller no longer needs this result (e.g. a partial overtaken by a final)
    put(mask, ctx - step - 1, qzero);
    ((int32_t*)ids.data())[0] = tok;
    ((int32_t*)pos.data())[0] = step;
    auto te = clk::now(); D.exec(); npu += ms_since(te); runs++;
    size_t best = argmax(logits);
    for (auto& s : P.self_new) {   // new token's K/V into a free slot of the input cache
      const Buf& o = D.out[s.out]; Buf& in = D.in[s.in];
      const size_t slots = s.k ? s.B : s.A, D_ = s.k ? s.A : s.B, slot = slots - 1 - (size_t)step;
      const bool same = same_enc(o, in);
      for (size_t h = 0; h < s.H; h++)
        for (size_t d = 0; d < D_; d++) {
          size_t src = h * D_ + d, dst = h * s.A * s.B + (s.k ? d * s.B + slot : slot * s.B + d);
          uint32_t q = o.esize == 1 ? o.cur[src] : o.esize == 2 ? ((const uint16_t*)o.cur)[src] : ((const uint32_t*)o.cur)[src];
          put(in, dst, same ? q : quantize(in, o.scale * ((float)q + o.offset)));
        }
    }
    for (auto& s : P.self) {   // this step's self cache outputs become next step's inputs
      Buf& o = D.out[s.first]; Buf& in = D.in[s.second];
      if (same_enc(o, in)) { uint8_t* co = o.cur; uint8_t* ci = in.cur; D.point(in, co); D.point(o, ci); }   // swap
      else requant(o, in);
    }
    if (step + 1 < (int)pre.size()) { tok = pre[step + 1]; continue; }   // forced prefix
    add_lp(logits, best);
    go = take(best);
  }
  P.last_steps = runs; P.last_lp = lp_n ? (float)(lp_sum / lp_n) : 0.f;
  for (auto& s : P.self) { Buf& o = D.out[s.first]; D.point(o, o.data()); }   // restore own output buffers
  if (total_ms) *total_ms = ms_since(t);
  if (npu_ms) *npu_ms = npu;
  return outv;
}

// Token bytes -> UTF-8 (bytes that do not form a complete character are dropped like Python errors="ignore", see
// board_asr.detok), then strip spaces.
static std::string detok_bytes(const std::vector<std::string>& vocab, const std::vector<int>& ids) {
  std::string b;
  for (int i : ids) if (i >= 0 && i < EOT && i < (int)vocab.size()) b += vocab[i];
  // Strict UTF-8 like Python: no overlong forms (C0/C1, E0 80-9F, F0 80-8F), no surrogates (ED A0-BF), nothing past
  // U+10FFFF (F4 90+, F5-FF). An invalid sequence is dropped as its lead byte plus the continuation bytes that were
  // still valid (Python's "maximal subpart"), so the next byte is read again as a possible lead.
  std::string o; size_t i = 0;
  while (i < b.size()) {
    unsigned char c = b[i];
    int n = c < 0x80 ? 1 : c >= 0xC2 && c <= 0xDF ? 2 : c >= 0xE0 && c <= 0xEF ? 3 : c >= 0xF0 && c <= 0xF4 ? 4 : 0;
    unsigned char lo = c == 0xE0 ? 0xA0 : c == 0xF0 ? 0x90 : 0x80, hi = c == 0xED ? 0x9F : c == 0xF4 ? 0x8F : 0xBF;
    int k = 1;   // bytes of the sequence that are valid so far
    for (; n > 1 && k < n && i + k < b.size(); k++) {
      unsigned char d = b[i + k];
      if (d < (k == 1 ? lo : 0x80) || d > (k == 1 ? hi : 0xBF)) break;
    }
    if (n > 0 && k == n) o.append(b, i, n);
    i += k;
  }
  size_t s = o.find_first_not_of(" \t\n\r"), e = o.find_last_not_of(" \t\n\r");
  return s == std::string::npos ? "" : o.substr(s, e - s + 1);
}

float Engine::last_avg_logprob() const { return p_->last_lp; }

std::string Engine::detok(const std::vector<int>& ids) const { return detok_bytes(p_->vocab, ids); }

static std::vector<std::string> load_vocab(const std::string& path) {
  auto v = read_file(path);
  int32_t n; memcpy(&n, v.data(), 4); size_t o = 4; std::vector<std::string> out(n);
  for (int i = 0; i < n; i++) { int32_t l; memcpy(&l, &v[o], 4); o += 4; out[i].assign((const char*)&v[o], l); o += l; }
  return out;
}

std::string Engine::transcribe(const float* pcm, size_t n, Stats* st) {
  auto t0 = clk::now();
  auto m = mel(pcm, n);
  double mel_ms = ms_since(t0);
  double enc_ms = encode(m.data());
  double dec_ms, npu_ms;
  auto toks = decode(korean_prefix(), p_->max_new, &dec_ms, &npu_ms, std::min((double)n / 16000.0, (double)p_->window));
  std::string text = detok(toks);
  if (st) { st->mel_ms = mel_ms; st->enc_ms = enc_ms; st->dec_ms = dec_ms; st->npu_ms = npu_ms;
            st->steps = p_->last_steps; st->avg_logprob = p_->last_lp; st->total_ms = ms_since(t0); }
  return text;
}

}  // namespace wasr
