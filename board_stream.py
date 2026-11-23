"""Continuous-input transcription on the Rubik Pi NPU (board version of scripts/stream_asr.py).
[capture thread] -> ring buffer -> [segmenter thread (VAD)] -> segment queue -> [inference thread (whisper_npu)] -> merge
  - Capture only writes to the ring buffer, so slow inference never stalls it; overwritten samples are counted.
  - Segments end at pauses; speech longer than the encoder window is cut at a word gap (--cut dip) or, if none,
    at the quietest frame with overlap (segmenter.py).
  - --astk [path]: audio comes from the audio-stack daemon (ASTK protocol, astk_client.py) instead of a wav/mic.
    Engine contract (ENGINE_CONTRACT.md):
      RESULT {type, text, lang, session_id, seg_id, t_start_ms, t_end_ms, t_start_us, t_end_us[, empty]}
        seg_id counts from 0 in each session; a segment's partials and its one final share it; no partial after its final.
        t_*_ms = ms from the session's first audio sample, from the capture clock (AUDIO.t_us), so audio missed
        while the engine was away and reconnects do not shift them. t_*_us = the capture clock itself.
        partial text = the whole segment so far (re-transcribed, earlier words may change).
        empty final ("text": "", "empty": true) = the segment produced no text: drop its partial.
      ENGINE_STATUS {event: ready | session_done | error, ...}; session_done comes after the session's last final.
    SESSION_END flushes the open segment as a final, then session_done. The connection is retried forever (the model
    stays loaded); replayed cached audio already received is skipped. --once = exit when the daemon closes.
  - --partial N: while a segment is still growing, transcribe what is there every N s and print it as a provisional
    result ([partial] lines). Finals always run first; a partial that is outdated before it runs is skipped.
Usage (on the board, via run_stream.sh for library paths):
  sh run_stream.sh --fleurs 15 --gap 0.6            # stream FLEURS sentences (fleurs70/) in real time, prints CER
  sh run_stream.sh --wav read/take1.wav --ref read/script1.txt   # continuous read speech, prints CER
  sh run_stream.sh --mic --seconds 60               # USB mic via arecord
  sh run_stream.sh --astk --partial 2               # live input from audiostackd (engine.sock), results sent back
  add --vad silero (needs: pip install onnxruntime) ; --speed 2 to stream faster than real time
  --cut last = the 10/13 board runs (quietest frame in the last 2 s, overlap); --partial 2 = provisional results
Error location (when sentence times are known: FLEURS, or tedx/talkNN.seg next to the wav): each reference char gets
an estimated time (spread evenly over its sentence); deleted/substituted chars are classified as inside a segment,
near a segment cut (<1.5 s), or in no segment (VAD dropped it); worst segments are printed with ref/hyp text."""
import argparse, collections, difflib, os, pathlib, queue, re, subprocess, threading, time
import numpy as np
from segmenter import Segmenter
from board_asr import WhisperNPU, invented, log_mel, detok, _vocab, read_wav, edit_distance, norm, align, strip_fillers, PREFIX, SR

HERE = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--window", type=int, default=10, help="encoder input length (s) = max segment length")
ap.add_argument("--bins", default=str(pathlib.Path.home() / "rubikpi/bins"))
ap.add_argument("--overlap", type=float, default=1.5)
ap.add_argument("--min-len", type=float, default=3.0)
ap.add_argument("--soft-len", type=float, default=6.0)
ap.add_argument("--vad", default="energy", choices=["energy", "silero"])
ap.add_argument("--fleurs", type=int, default=0, help="stream the first N sentences of fleurs70/")
ap.add_argument("--gap", type=float, default=0.6, help="pause between FLEURS sentences (s)")
ap.add_argument("--wav"); ap.add_argument("--mic", action="store_true")
ap.add_argument("--astk", nargs="?", const=os.environ.get("ASTK_ENGINE_SOCKET", "/run/audiostack/engine.sock"),
                help="read audio from audiostackd's engine socket and send results back")
ap.add_argument("--once", action="store_true", help="--astk: exit when the daemon says GOODBYE or closes (default: reconnect)")
ap.add_argument("--ref", help="reference text file for --wav/--mic (e.g. read/script1.txt) -> prints CER")
ap.add_argument("--seconds", type=float, default=60)
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--buffer", type=float, default=30)
ap.add_argument("--cut", default="dip", choices=["dip", "early", "last"],
                help="long speech: dip = word gap found when the window is full (default), early = word gap cut as soon as it ends "
                     "after --soft-len s (low latency), last = old")
ap.add_argument("--dip-db", type=float, default=12.0, help="min dip below speech level to count as a word gap")
ap.add_argument("--adapt", action="store_true", help="noise-aware pause and word-gap thresholds (segmenter.py)")
ap.add_argument("--lp-min", type=float, help="drop results with average log prob below this, or dominated by one repeated word (e.g. -0.5); off by default")
ap.add_argument("--partial", type=float, default=0, help="provisional result every N s of a growing segment (0 = off)")
ap.add_argument("--stable", default="off", choices=["off", "partial", "final"],
                help="partial stabilization: tokens two partials in a row agree on (up to a word boundary) are committed and "
                     "fed to the decoder as a forced prefix of later partials (partial) and of the final too (final); "
                     "RESULT partials carry stable_chars")
ap.add_argument("--hold-last", action="store_true",
                help="partials leave out their last word (often cut off mid-word; it comes with the next partial or the final)")
ap.add_argument("--dec", help="decoder .bin name in --bins (default dec_<window>s.bin), e.g. dec_10s_pv.bin")
a = ap.parse_args()
FRAME = 512 if a.vad == "silero" else 480
FD = FRAME / SR

ref = None; spans = None                   # spans: [(start s, end s, sentence text)] for error location
if a.mic or a.astk:
    audio = None
elif a.wav:
    audio = read_wav(a.wav)
    seg_file = pathlib.Path(a.wav).with_suffix(".seg")
    if seg_file.exists():
        spans = [(float(t1), float(t2), txt) for t1, t2, txt in (l.rstrip("\n").split("\t", 2) for l in open(seg_file, encoding="utf-8") if l.strip())]
else:
    rows = [l.rstrip("\n").split("\t") for l in open(HERE / "fleurs70/refs.tsv", encoding="utf-8")][: a.fleurs or 15]
    parts = []
    spans = []
    for w, r in rows:
        x = read_wav(HERE / "fleurs70" / w); n0 = sum(len(p) for p in parts)
        spans.append((n0 / SR, (n0 + len(x)) / SR, r))
        parts += [x, np.zeros(int(a.gap * SR), np.float32)]
    audio = np.concatenate(parts); ref = " ".join(r for _, r in rows)
if a.ref: ref = open(a.ref, encoding="utf-8").read().strip()
total_s = a.seconds if a.mic else 0 if a.astk else len(audio) / SR


class Ring:
    """The writer never waits; samples overwritten before being read are counted in dropped."""
    def __init__(self, sec):
        self.n = int(sec * SR); self.buf = np.zeros(self.n, np.float32)
        self.w = self.r = 0; self.dropped = 0; self.done = False; self.poked = False; self.cv = threading.Condition()
    def write(self, x):
        with self.cv:
            for i in range(0, len(x), self.n):
                c = x[i:i + self.n]; p = self.w % self.n; k = min(len(c), self.n - p)
                self.buf[p:p + k] = c[:k]; self.buf[:len(c) - k] = c[k:]; self.w += len(c)
            if self.w - self.r > self.n: self.dropped += self.w - self.r - self.n; self.r = self.w - self.n
            self.cv.notify_all()
    def close(self):
        with self.cv: self.done = True; self.cv.notify_all()
    def poke(self):
        """Wake a waiting read() even without a full frame (it returns an empty array): session events."""
        with self.cv: self.poked = True; self.cv.notify_all()
    def read(self, k):
        with self.cv:
            while self.w - self.r < k:
                if self.done: return None
                if self.poked: self.poked = False; return np.zeros(0, np.float32)
                self.cv.wait()
            idx = np.arange(self.r, self.r + k) % self.n; self.r += k
            return self.buf[idx].copy()


ring = Ring(a.buffer); segq = queue.Queue(); t0 = None

client = None; marks = collections.deque()          # --astk: (sample pos, "begin"|"end", session id), in stream order
tmap = collections.deque()                          # --astk: (sample pos at the end of an AUDIO chunk, its capture t_us)
_slock = threading.Lock(); outbox = collections.deque(maxlen=512)   # --astk: frames that could not be sent (disconnected)
astk = {"connected": False, "sid": None, "dup": 0, "reconnects": 0, "last_us": 0}
if a.astk:
    from astk_client import AudioStackClient, MsgType, Flags

    def astk_connect():
        """Connect (waits until the daemon's socket accepts). capture thread (auto PONG) and infer thread both send,
        so every send goes through one lock."""
        c = AudioStackClient(a.astk); c.connect()
        _send = c.send_frame
        def _locked_send(*args, **kw):
            with _slock: return _send(*args, **kw)
        c.send_frame = _locked_send
        return c


_olock = threading.RLock()


def astk_send(method, *args, **kw):
    """RESULT / ENGINE_STATUS to the daemon, in order; kept in `outbox` while disconnected and sent after reconnecting."""
    if client is None: return
    with _olock:
        if len(outbox) == outbox.maxlen: print("[astk] outbox full, oldest message dropped", flush=True)
        outbox.append((method, args, kw))
        if astk["connected"]: astk["connected"] = flush_outbox()


def flush_outbox():
    with _olock:
        while outbox:
            method, args, kw = outbox[0]
            try: getattr(client, method)(*args, **kw)
            except OSError as err: print(f"[astk] send failed, kept for after reconnect: {err}", flush=True); return False
            outbox.popleft()
        return True


def t_at(pos, first=False):
    """--astk: CLOCK_MONOTONIC time (s) at which sample `pos` was captured.
    first=True: `pos` starts a span (session or segment start), so it is read from the chunk that begins there, not from
    the chunk that ends there. The two differ when capture paused in between: audiostackd --idle sends no AUDIO between
    'session stop' and 'session start', so the next session's first chunk comes after the idle time."""
    ends_here = None
    for end, t_us in list(tmap):
        if end == pos and first: ends_here = t_us; continue    # the chunk before `pos`: only if nothing came after it
        if end >= pos: return (t_us - (end - pos) * 1e6 / SR) / 1e6
    return ends_here / 1e6 if ends_here is not None else time.monotonic()


def capture_astk():
    """Daemon frames -> ring. Reconnects forever (unless --once); cached replays of audio already received are skipped,
    and a SESSION_BEGIN repeated for the session already open (sent on every connect) is ignored."""
    global t0, client
    t0 = time.perf_counter()
    while True:
        bye = False
        try:
            for f in client.frames():
                if f.type == MsgType.AUDIO:
                    if f.flags & Flags.CACHED and f.t_us <= astk["last_us"]: astk["dup"] += 1; continue
                    astk["last_us"] = max(astk["last_us"], f.t_us)
                    x = np.frombuffer(f.payload, np.int16).astype(np.float32) / 32768
                    tmap.append((ring.w + len(x), f.t_us))
                    while len(tmap) > a.buffer * 100: tmap.popleft()
                    ring.write(x)
                elif f.type == MsgType.HELLO:
                    fmt = f.json().get("format", {})
                    if fmt.get("rate", SR) != SR: raise SystemExit(f"audiostackd sends {fmt}, need {SR} Hz")
                elif f.type == MsgType.SESSION_BEGIN:
                    sid = f.json().get("session_id")
                    if sid == astk["sid"]: continue                       # same session after a reconnect
                    if astk["sid"] is not None:                           # previous one never ended (daemon restarted)
                        print(f"[astk] {astk['sid']} closed without SESSION_END", flush=True)
                        marks.append((ring.w, "end", astk["sid"]))
                    astk["sid"] = sid; marks.append((ring.w, "begin", sid)); ring.poke()
                elif f.type == MsgType.SESSION_END:
                    sid = f.json().get("session_id") or astk["sid"]
                    if sid != astk["sid"]: continue                       # not the open session (already closed)
                    astk["sid"] = None; marks.append((ring.w, "end", sid)); ring.poke()
                elif f.type == MsgType.GOODBYE:
                    print(f"[astk] GOODBYE {f.payload[:100].decode('utf-8', 'replace')}", flush=True)
                    if a.once: bye = True; break      # socket stays open: the flushed finals still go out
        except (OSError, ValueError) as err:
            print(f"[astk] connection error: {err}", flush=True)
        if not bye: astk["connected"] = False
        if a.once: break
        print("[astk] disconnected, waiting for audiostackd ...", flush=True)
        try: client.close()
        except OSError: pass
        client = astk_connect(); astk["reconnects"] += 1
        with _olock: astk["connected"] = flush_outbox()
        astk_send("send_status", event="ready", **STATUS)
        print(f"[astk] reconnected ({astk['reconnects']})", flush=True)
    ring.close()


def capture():
    global t0
    t0 = time.perf_counter()
    if a.mic:
        p = subprocess.Popen(["arecord", "-q", "-f", "S16_LE", "-r", str(SR), "-c", "1", "-t", "raw", "-d", str(int(a.seconds))],
                             stdout=subprocess.PIPE)
        while (b := p.stdout.read(640)):
            ring.write(np.frombuffer(b, np.int16).astype(np.float32) / 32768)
    else:
        blk = 320
        for i in range(0, len(audio), blk):
            ring.write(audio[i:i + blk])
            dt = t0 + (i + blk) / SR / a.speed - time.perf_counter()
            if dt > 0: time.sleep(dt)
    ring.close()


if a.vad == "silero":
    import onnxruntime as ort
    _sv = ort.InferenceSession(str(HERE / "silero_vad.onnx"), providers=["CPUExecutionProvider"])
    _st = [np.zeros((2, 1, 128), np.float32), np.zeros(64, np.float32)]
    def speech_prob(f):
        o, _st[0] = _sv.run(None, {"input": np.concatenate([_st[1], f])[None], "state": _st[0], "sr": np.array(SR, np.int64)})
        _st[1] = f[-64:]; return float(o[0, 0])


pending = [None]; plock = threading.Lock(); pstats = {"run": [], "skipped": 0}


def segmenter():
    def new(off): return Segmenter(FRAME, a.window, a.min_len, a.soft_len, a.overlap, cut=a.cut, dip_db=a.dip_db, adaptive=a.adapt,
                                speech_prob=speech_prob if a.vad == "silero" else None, offset=off)
    sg = new(0); seg_no, last_p = 0, 0
    sess = {"id": None, "start": 0, "base": 0, "t0": None}   # session id, its first sample, first seg_no, first sample's capture time

    def times(s, e):
        """Segment [s, e) samples -> session-relative ms (and capture µs under --astk, from AUDIO.t_us)."""
        if not a.astk: return dict(t0_ms=(s - sess["start"]) * 1000 // SR, t1_ms=(e - sess["start"]) * 1000 // SR)
        if sess["t0"] is None: sess["t0"] = t_at(sess["start"], first=True)
        s = max(s, sess["start"])                        # a session's first frame can begin a little before it
        ts, te = t_at(s, first=True), t_at(e)
        return dict(t0_ms=max(0, round((ts - sess["t0"]) * 1000)), t1_ms=max(0, round((te - sess["t0"]) * 1000)),
                    t0_us=round(ts * 1e6), t1_us=round(te * 1e6))

    def put(segs):
        nonlocal seg_no, last_p
        for s, x, forced, info in segs:
            info.update(sid=sess["id"], seg=seg_no - sess["base"], **times(s, s + len(x)))
            segq.put((seg_no, s, s + len(x), x, forced, info)); seg_no += 1; last_p = 0

    def session_marks(upto, nxt):                       # marks at positions <= upto; nxt = stream position of the next frame
        nonlocal sg
        while marks and marks[0][0] <= upto:
            pos, kind, sid = marks.popleft()
            if kind == "end":                           # close the open segment as a final of the ending session
                put(sg.flush()); sg = new(nxt); segq.put(("end", sid))
                with plock: pending[0] = None
                sess.update(id=None)
            else: sess.update(id=sid, start=pos, base=seg_no, t0=None)

    while (f := ring.read(FRAME)) is not None:
        if not len(f):                                  # poked by a session event: the < 1 frame not read yet is dropped
            session_marks(ring.r + FRAME - 1, ring.r); continue
        session_marks(ring.r - FRAME, ring.r - FRAME)
        put(sg.feed(f))
        if a.partial and sg.speaking:
            n = len(sg.frames) * FRAME
            if n >= 1.5 * SR and n - last_p >= a.partial * SR:
                x = np.concatenate([fr for fr, _, _ in sg.frames]); last_p = n; s = sg.offset + sg.start * FRAME
                info = dict(sid=sess["id"], seg=seg_no - sess["base"], **times(s, s + len(x)))
                with plock:
                    if pending[0] is not None: pstats["skipped"] += 1        # replaced before it could run
                    pending[0] = (seg_no, s, s + len(x), x, info)
    session_marks(float("inf"), ring.r); put(sg.flush())          # marks first: a final SESSION_END flushes exactly once
    segq.put(None)


def merge(prev, new):
    """Overlap-cut segments: drop the start of new if it repeats the end of prev."""
    p, q = norm(prev)[-40:], norm(new); best = 0
    for k in range(min(len(p), len(q)), 1, -1):
        if difflib.SequenceMatcher(None, p[-k:], q[:k]).ratio() >= 0.8: best = k; break
    if not best: return new
    seen = 0
    for i, ch in enumerate(new):
        if re.match(r"\w", ch): seen += 1
        if seen == best: return new[i + 1:].lstrip()
    return ""


model = WhisperNPU(f"{a.bins}/enc_{a.window}s.bin", f"{a.bins}/{a.dec or f'dec_{a.window}s.bin'}")
model.encode(np.zeros((80, a.window * 100), np.float32)); model.decode(1)   # warm up (loads graphs)
out, stats, segs, kinds = [], [], [], []


fstats = {"segs": 0, "chars": 0}


def clean_text(text, final):
    if a.lp_min is None or not text or not invented(text, model.last_lp, a.lp_min): return text
    if final: fstats["segs"] += 1; fstats["chars"] += len(norm(text)); print(f"[filtered lp {model.last_lp:.2f}] {text}", flush=True)
    return ""


pstable = {}                                          # --stable: segment -> (committed tokens, previous partial tokens)


def word_start(t): return _vocab.get(t, b"").startswith(b" ")


def committed(n):
    for k in [k for k in pstable if k < n]: del pstable[k]  # segments already finalized
    return pstable.get(n, ([], []))[0]


def commit(n, full):
    """LocalAgreement-2 on tokens: the common prefix of this and the previous partial, cut back to a word boundary."""
    com, prev = pstable.get(n, ([], []))
    k = 0
    while k < min(len(prev), len(full)) and prev[k] == full[k]: k += 1
    while k > len(com) and not (k < len(full) and word_start(full[k])): k -= 1   # the next token must start a word
    if k > len(com): com = full[:k]
    pstable[n] = (com, full)
    return com


def run_partial(job, done):
    n, s, e, x, info = job
    if n <= done: pstats["skipped"] += 1; return        # its final already went out
    ts = time.perf_counter()
    com = committed(n) if a.stable != "off" else []
    enc_ms = model.encode(log_mel(x, a.window)); toks, dec_ms, _ = model.decode(seconds=len(x) / SR, forced=com)
    full = com + toks
    text = clean_text(detok(full), False); proc = time.perf_counter() - ts
    stable = None
    if a.stable != "off" and text: stable = len(detok(commit(n, full)))
    if a.hold_last:
        text = text.rsplit(" ", 1)[0] if " " in text else ""
        if stable is not None: stable = min(stable, len(text))
    lat = latency(e)
    pstats["run"].append((proc, lat))
    if text: send_result("partial", n, info, text, stable)
    shown = text if stable is None else f"{text[:stable]}|{text[stable:]}"
    print(f"[partial {s/SR:6.1f}~{e/SR:6.1f}s seg {n}] enc {enc_ms:3.0f}ms dec {dec_ms:4.0f}ms ({len(toks)} tok) "
          f"latency {lat:4.2f}s | {shown}", flush=True)


def latency(e):
    """End of the segment's audio -> now (s)."""
    return time.monotonic() - t_at(e) if a.astk else time.perf_counter() - (t0 + e / SR / a.speed)


def send_result(kind, n, info, text, stable=None):
    """--astk RESULT. Results with no open session (between SESSION_END and the next SESSION_BEGIN) are not sent."""
    if client is None: return
    if info["sid"] is None: print(f"[astk] no open session, {kind} not sent", flush=True); return
    extra = {"seg_id": info["seg"], "t_start_us": info["t0_us"], "t_end_us": info["t1_us"]}
    if kind == "final" and not text: extra["empty"] = True
    if stable is not None: extra["stable_chars"] = stable
    astk_send("send_result", text, kind=kind, t_start_ms=info["t0_ms"], t_end_ms=info["t1_ms"], session_id=info["sid"], extra=extra)


def infer():
    prev_forced, last, done = False, "", -1
    finals = collections.Counter()                       # finals sent per session (for session_done)
    while True:
        try: item = segq.get(timeout=0.02)
        except queue.Empty:                              # nothing final waiting -> newest partial, if any
            with plock: job, pending[0] = pending[0], None
            if job: run_partial(job, done)
            continue
        if item is None: break
        if item[0] == "end":                             # session boundary: no overlap merge across it
            prev_forced, last = False, ""
            if client: astk_send("send_status", event="session_done", session_id=item[1], finals=finals.pop(item[1], 0))
            print(f"[astk] session_done {item[1]}", flush=True)
            continue
        n, s, e, x, forced, info = item
        ts = time.perf_counter()
        com = committed(n) if a.stable == "final" else []
        enc_ms = model.encode(log_mel(x, a.window)); toks, dec_ms, _ = model.decode(seconds=len(x) / SR, forced=com)
        text = clean_text(detok(com + toks), True); proc = time.perf_counter() - ts
        if prev_forced: text = merge(last, text)
        out.append(text); last = text or last; prev_forced = forced; done = n
        lat = latency(e)
        send_result("final", n, info, text); finals[info["sid"]] += 1
        stats.append((e - s, proc, lat, segq.qsize())); segs.append((s / SR, e / SR, text)); kinds.append(info["kind"])
        cut = f" cut {info['kind']}" + (f" {info['depth_db']:.0f}dB" if info["depth_db"] is not None else "") if info["kind"] in ("dip", "forced") else ""
        print(f"[{s/SR:6.1f}~{e/SR:6.1f}s{' overlap' if forced else ''}] enc {enc_ms:3.0f}ms dec {dec_ms:4.0f}ms "
              f"({len(toks)} tok) latency {lat:4.2f}s queued {segq.qsize()}{cut} | {text}", flush=True)


def infer_guarded():
    """--astk: an inference failure (e.g. whisper_npu died) is reported as ENGINE_STATUS error and ends the process
    with exit code 2 so systemd restarts it; the capture thread would otherwise keep it alive forever."""
    try: infer()
    except BaseException as err:
        import traceback; traceback.print_exc()
        if client:
            astk_send("send_status", event="error", message=f"{type(err).__name__}: {err}"[:300], session_id=astk["sid"])
            time.sleep(0.2)
        os._exit(2)


STATUS = dict(model="whisper-small-komixv2", backend=os.environ.get("WHISPER_RUNNER", "qnn-htp"), loaded=True,
              decoder=a.dec or f"dec_{a.window}s.bin", window_s=a.window, partial_s=a.partial, pid=os.getpid())
if a.astk:                                           # connect only once the model is loaded: no audio piles up meanwhile
    print(f"[astk] connecting to {a.astk} ...", flush=True)
    client = astk_connect(); astk["connected"] = True
    astk_send("send_status", event="ready", **STATUS)
print(f"input {'audiostackd ' + a.astk if a.astk else f'{total_s:.1f}s'}, window {a.window}s, vad {a.vad}, {a.speed}x", flush=True)
th = [threading.Thread(target=f, daemon=True) for f in (capture_astk if a.astk else capture, segmenter, infer_guarded if a.astk else infer)]
for t in th: t.start()
for t in th: t.join()
model.close()
hyp = " ".join(t for t in out if t)
seg_s = sum(st[0] for st in stats) / SR
print(f"\nbuffer overflow (dropped samples): {ring.dropped}")
if a.astk: print(f"ASTK reconnects {astk['reconnects']}, cached duplicates skipped {astk['dup']} chunks, unsent {len(outbox)}")
print(f"cuts: pause {kinds.count('pause') + kinds.count('end')}, dip {kinds.count('dip')}, forced {kinds.count('forced')} (--cut {a.cut})")
if a.lp_min is not None: print(f"FILTER lp < {a.lp_min}: {fstats['segs']} segments, {fstats['chars']} chars dropped")
if a.partial:
    pr = pstats["run"]
    print(f"PARTIAL every {a.partial}s: {len(pr)} run, {pstats['skipped']} skipped"
          + (f", latency avg {np.mean([l for _, l in pr]):.2f}s max {max(l for _, l in pr):.2f}s, proc avg {np.mean([p for p, _ in pr]):.2f}s" if pr else ""))
if stats:
    print(f"{len(stats)} segments, RTF {sum(st[1] for st in stats) / max(seg_s, 1e-9):.3f}, "
          f"latency avg {np.mean([st[2] for st in stats]):.2f}s max {max(st[2] for st in stats):.2f}s, max queued {max(st[3] for st in stats)}")
if ref:
    e = edit_distance(norm(ref), norm(hyp)); print(f"CER {e/len(norm(ref))*100:.1f}% ({e}/{len(norm(ref))} chars)")
    ef = edit_distance(norm(ref), norm(strip_fillers(hyp)))      # same run, hesitation sounds removed from the output
    print(f"CER_NOFILLER {ef/len(norm(ref))*100:.1f}% ({ef}/{len(norm(ref))} chars)")
if stats:   # speed on the same run: processing time scaled to a 10 s segment, and the worst long segment
    long = [(n / SR, p) for n, p, _, _ in stats if n / SR >= 8]
    p10 = sum(p for _, p in long) / max(sum(d for d, _ in long), 1e-9) * 10
    print(f"SPEED proc per 10s audio {p10:.2f}s (segments>=8s: {len(long)}, max {max((p for _, p in long), default=0):.2f}s), "
          f"latency avg {np.mean([st[2] for st in stats]):.2f}s")
    cer_ok = (not ref) or e / len(norm(ref)) <= 0.10
    print(f"TARGET CER<=10%: {'OK' if cer_ok else 'NO'} | 10s in <=1s: {'OK' if p10 <= 1.0 else 'NO'}")
if ref and spans and stats:                 # where are the errors? (needs sentence times)
    rc, rt = [], []
    for t1, t2, txt in spans:
        c = norm(txt); rc += c; rt += [t1 + (i + 0.5) / len(c) * (t2 - t1) for i in range(len(c))]
    if "".join(rc) == norm(ref):
        ops, n_ins, ins_at = align("".join(rc), norm(hyp))
        print(f"\nERRORS sub {ops.count('sub')}, del {ops.count('del')}, ins {n_ins} (of {len(rc)} ref chars)")
        cuts = sorted({t for s_, e_, _ in segs for t in (s_, e_)})
        def where(t):
            if not any(s_ <= t <= e_ for s_, e_, _ in segs): return "no segment"
            return "near cut (<1.5s)" if min(abs(t - c) for c in cuts) < 1.5 else "inside"
        for op in ("del", "sub"):
            k = {"inside": 0, "near cut (<1.5s)": 0, "no segment": 0}
            for i, o in enumerate(ops):
                if o == op: k[where(rt[i])] += 1
            print(f"{op} by location: " + ", ".join(f"{n} {v}" for n, v in k.items()))
        rows_ = []
        for s_, e_, txt in segs:
            idx = [i for i in range(len(rc)) if s_ <= rt[i] <= e_]
            bad = sum(ops[i] != "ok" for i in idx) + sum(ins_at[i] for i in idx)
            rows_.append((bad, s_, e_, idx, txt))
        print("worst segments (errors = sub+del+ins):")
        for bad, s_, e_, idx, txt in sorted(rows_, key=lambda r: -r[0])[:8]:
            if not bad: break
            print(f"  {s_:6.1f}-{e_:6.1f}s ({e_ - s_:4.1f}s) errors {bad}/{len(idx)}")
            print(f"    ref: {''.join(rc[i] for i in idx)}")
            print(f"    err: {''.join({'ok': '.', 'sub': 'S', 'del': 'D'}[ops[i]] for i in idx)}")
            print(f"    hyp: {txt}")
    else: print("\n(error location skipped: sentence file does not match the reference)")
print(f"\n[hypothesis]\n{hyp}")
