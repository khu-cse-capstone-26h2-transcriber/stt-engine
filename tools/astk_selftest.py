"""Stand-in audiostackd for checking the engine side of the ASTK protocol (board_stream.py --astk) without the daemon.
It serves engine.sock like audiostackd (HELLO, SESSION_BEGIN/END, 20 ms AUDIO with a capture clock in t_us, a cache
replayed with the cached flag on reconnect) and scripts the cases that break integration:
  1. engine connection dropped in the middle of session 1 (--drop-at s, for --drop-for s): audio keeps being "captured",
     the engine reconnects, gets SESSION_BEGIN for the same session and the last --cache s replayed (cached flag)
  2. session rotate at --rotate-at s: SESSION_END(s1) then SESSION_BEGIN(s2); with --idle-gap g, no AUDIO for g s in
     between while the capture clock runs on, like audiostackd --idle between 'session stop' and 'session start'
  3. SESSION_END(s2) at the end of the audio, then waits for the engine's session_done, then GOODBYE
Checks printed at the end (PASS/FAIL): results carry the session they belong to, seg_id from 0 per session, a final per
seg_id and no partial after it, t_start_ms increasing within a session and matching the audio position, session_done
after the last final of each session, no repeated text across the drop.
Only the protocol is checked; text quality and speed are measured on the board.
Usage (two terminals, or the engine in the background):
  python3 astk_selftest.py tedx/talk04.wav --socket /tmp/astk-test/engine.sock --seconds 70 --drop-at 25 --rotate-at 45
  WHISPER_RUNNER="python3 fake_runner.py" python3 board_stream.py --astk /tmp/astk-test/engine.sock --once --bins <onnx dir>
Exit code 0 = all checks passed."""
import argparse, json, os, pathlib, select, socket, sys, time
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from astk_client import AudioStackClient, MsgType, Flags, HEADER
from board_asr import read_wav, SR

ap = argparse.ArgumentParser()
ap.add_argument("wav")
ap.add_argument("--socket", default="/tmp/astk-test/engine.sock")
ap.add_argument("--seconds", type=float, default=70, help="use the first N s of the wav")
ap.add_argument("--speed", type=float, default=1.0, help="send faster than real time (the capture clock stays real)")
ap.add_argument("--drop-at", type=float, default=25, help="close the engine connection at this audio time (s), 0 = never")
ap.add_argument("--drop-for", type=float, default=3, help="audio captured while the engine is away (s)")
ap.add_argument("--proto", choices=["new", "old"], default="new",
                help="new = audio-stack 32124c7+: AUDIO t_us is the chunk's first sample, SESSION_BEGIN carries the session's "
                     "first-sample t_us; old = t_us is the chunk end, SESSION_BEGIN without t_us")
ap.add_argument("--cache", type=float, default=10, help="audio replayed on reconnect (s), like [engine] cache_seconds")
ap.add_argument("--rotate-at", type=float, default=45, help="session rotate at this audio time (s), 0 = never")
ap.add_argument("--idle-gap", type=float, default=0, help="at the rotate, send no AUDIO for this long (s), 0 = rotate at once")
ap.add_argument("--done-timeout", type=float, default=30, help="max wait for session_done after the last SESSION_END (s)")
ap.add_argument("--log", help="write every received RESULT/ENGINE_STATUS as JSON lines here")
a = ap.parse_args()

pcm = (np.clip(read_wav(a.wav)[: int(a.seconds * SR)], -1, 1) * 32767).astype(np.int16)
CH = SR // 50                                          # 20 ms
T0_US = 0                                              # capture clock: sample i at T0 + GAP + i / SR (set when streaming starts)
GAP_US = 0                                             # + time spent idle between the sessions (--idle-gap)
os.makedirs(os.path.dirname(a.socket) or ".", exist_ok=True)
try: os.unlink(a.socket)
except FileNotFoundError: pass
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); srv.bind(a.socket); srv.listen(1)
print(f"[selftest] {a.wav} {len(pcm)/SR:.1f}s, socket {a.socket}: start the engine", flush=True)

got = []                                               # (recv monotonic, frame type, json, session open at the daemon then)
state = {"sid": None, "conn": None, "client": None, "rbuf": b"", "begin_us": None}


def begin():                                           # SESSION_BEGIN for the open session (new protocol: with its origin)
    obj = {"session_id": state["sid"]}
    if a.proto == "new": obj["t_us"] = state["begin_us"]
    send(MsgType.SESSION_BEGIN, obj)


def accept():
    conn, _ = srv.accept(); c = AudioStackClient.__new__(AudioStackClient); c._sock, c._buf, c._seq = conn, b"", 0
    state.update(conn=conn, client=c, rbuf=b"")
    c.send_frame(MsgType.HELLO, json.dumps({"proto": 1, "name": "astk_selftest", "format": {"rate": SR, "channels": 1, "encoding": "s16le"},
                                            "chunk_ms": 20, "cache_seconds": a.cache, "session_id": state["sid"]}).encode())
    if state["sid"]: begin()
    print("[selftest] engine connected", flush=True)


def send(t, obj=None, payload=b"", flags=0, t_us=None):
    c = state["client"]
    if t_us is None: hdr_t = int(time.monotonic() * 1e6)
    else: hdr_t = t_us
    body = json.dumps(obj).encode() if obj is not None else payload
    c._sock.sendall(HEADER.pack(0x4B535441, 1, t, flags, c._seq, hdr_t, len(body)) + body); c._seq += 1


def drain(wait=0.0):
    conn = state["conn"]
    if conn is None: return True
    end = time.monotonic() + wait
    while True:
        r = select.select([conn], [], [], max(0.0, end - time.monotonic()))[0]
        if not r: return True
        data = conn.recv(65536)
        if not data: return False
        state["rbuf"] += data
        b = state["rbuf"]
        while len(b) >= HEADER.size:
            _, _, t, _, _, t_us, n = HEADER.unpack(b[:HEADER.size])
            if len(b) < HEADER.size + n: break
            body = b[HEADER.size:HEADER.size + n]; b = b[HEADER.size + n:]
            if t in (MsgType.RESULT, MsgType.ENGINE_STATUS):
                obj = json.loads(body.decode("utf-8")); got.append((time.monotonic(), t, obj, state["sid"]))
                tag = "RESULT" if t == MsgType.RESULT else "STATUS"
                print(f"[selftest] {tag} (daemon session {state['sid']}) {json.dumps(obj, ensure_ascii=False)[:200]}", flush=True)
        state["rbuf"] = b
        if wait == 0.0 and not select.select([conn], [], [], 0)[0]: return True


def audio(i, cached=False):
    x = pcm[i:i + CH]; t_us = T0_US + GAP_US + int((i + (0 if a.proto == "new" else len(x))) * 1e6 / SR)
    send(MsgType.AUDIO, payload=x.tobytes(), flags=Flags.CACHED if cached else 0, t_us=t_us)


accept()
t_w = time.monotonic()                                 # first connect: wait for the engine's ready (model loaded)
while time.monotonic() - t_w < 300 and not any(t == MsgType.ENGINE_STATUS and o.get("event") == "ready" for _, t, o, _ in got):
    if not drain(0.2): sys.exit("[selftest] engine closed before ready")
T0_US = int(time.monotonic() * 1e6)                    # capture clock starts with the first chunk, so engine latency is real
state.update(sid="sess-test-1", begin_us=T0_US); begin()
sessions = {"sess-test-1": 0}                         # session -> first sample
drop_i = int(a.drop_at * SR) if a.drop_at else -1
rot_i = int(a.rotate_at * SR) if a.rotate_at else -1
t_start = time.perf_counter(); i = 0; dropped = False
while i < len(pcm):
    if i >= drop_i >= 0 and not dropped:              # 1. engine connection lost; capture continues
        dropped = True; print(f"[selftest] dropping the engine connection at {i/SR:.1f}s", flush=True)
        state["conn"].shutdown(socket.SHUT_RDWR); state["conn"].close(); state["conn"] = None
        time.sleep(a.drop_for / a.speed); i_back = i + int(a.drop_for * SR)    # audio captured while away
        accept()
        for j in range(max(0, i_back - int(a.cache * SR)) // CH * CH, i_back, CH): audio(j, cached=True)
        print(f"[selftest] replayed {min(a.cache, i_back / SR):.1f}s cached audio, live from {i_back/SR:.1f}s", flush=True)
        i = i_back; t_start = time.perf_counter() - i / SR / a.speed      # real-time pacing continues from here
        continue
    if i >= rot_i >= 0 and state["sid"] == "sess-test-1":   # 2. rotate
        print(f"[selftest] session rotate at {i/SR:.1f}s", flush=True)
        send(MsgType.SESSION_END, {"session_id": "sess-test-1"})
        if a.idle_gap:                                 # idle between the sessions: no AUDIO, the capture clock runs on
            print(f"[selftest] idle {a.idle_gap:.1f}s (no AUDIO)", flush=True)
            t_idle = time.monotonic()
            while time.monotonic() - t_idle < a.idle_gap / a.speed: drain(0.05)
            GAP_US += int(a.idle_gap * 1e6); t_start += a.idle_gap / a.speed
        state.update(sid="sess-test-2", begin_us=T0_US + GAP_US + int(i * 1e6 / SR)); sessions["sess-test-2"] = i; begin()
    audio(i); i += CH
    time.sleep(max(0.0, t_start + i / SR / a.speed - time.perf_counter()))
    drain()
last = state["sid"]; t_end = time.monotonic()
send(MsgType.SESSION_END, {"session_id": last}); state["sid"] = None           # 3. end
print("[selftest] SESSION_END sent, waiting for session_done", flush=True)
while time.monotonic() - t_end < a.done_timeout:
    if not drain(0.1): break
    if any(t == MsgType.ENGINE_STATUS and o.get("event") == "session_done" and o.get("session_id") == last for _, t, o, _ in got): break
done_wait = time.monotonic() - t_end
send(MsgType.GOODBYE, {"reason": "selftest done"}); drain(1.0)
if state["conn"]: state["conn"].close()
srv.close(); os.unlink(a.socket)
if a.log:
    with open(a.log, "w", encoding="utf-8") as f:
        for tm, t, o, sid in got: f.write(json.dumps({"t": tm, "frame": "RESULT" if t == MsgType.RESULT else "STATUS", "daemon_sid": sid, **o}, ensure_ascii=False) + "\n")

# ---- checks ---------------------------------------------------------------------------------------------------------
res = [o for _, t, o, _ in got if t == MsgType.RESULT]
st = [(k, o) for k, (_, t, o, _) in enumerate(got) if t == MsgType.ENGINE_STATUS]
fails = []
def check(ok, what):
    print(f"{'PASS' if ok else 'FAIL'} {what}"); ok or fails.append(what)

check(all(o.get("session_id") in sessions for o in res), "every result has a known session_id")
for sid, first in sessions.items():
    r = [o for o in res if o.get("session_id") == sid]
    fin = [o for o in r if o["type"] == "final"]
    ids = [o["seg_id"] for o in fin]
    check(ids == list(range(len(ids))) and len(ids) > 0, f"{sid}: finals have seg_id 0..{len(ids)-1} in order")
    order_ok = all(not (o["type"] == "partial" and any(f["seg_id"] == o["seg_id"] for f in r[:k] if f["type"] == "final")) for k, o in enumerate(r))
    check(order_ok, f"{sid}: no partial after its segment's final")
    starts = [o["t_start_ms"] for o in fin]
    check(all(x < y for x, y in zip(starts, starts[1:])), f"{sid}: t_start_ms increases ({starts[:6]}...)")
    dur = (len(pcm) if sid == last else rot_i if rot_i > 0 else len(pcm)) - first
    check(all(0 <= o["t_start_ms"] <= o["t_end_ms"] <= dur * 1000 / SR + 100 for o in fin), f"{sid}: times inside the session (0..{dur/SR:.1f}s)")
    gap = GAP_US if sid == "sess-test-2" else 0
    us = [(o["t_start_us"] - T0_US - gap) / 1e6 - first / SR for o in fin]
    check(all(abs(u * 1000 - o["t_start_ms"]) < 25 for u, o in zip(us, fin)), f"{sid}: t_start_ms = capture clock - session start")
    k_done = [k for k, o in st if o.get("event") == "session_done" and o.get("session_id") == sid]
    k_fin = [k for k, (_, t, o, _) in enumerate(got) if t == MsgType.RESULT and o.get("session_id") == sid and o["type"] == "final"]
    check(len(k_done) == 1 and k_done[0] > max(k_fin, default=-1), f"{sid}: one session_done, after its last final")
    if k_done: check(st[[k for k, _ in st].index(k_done[0])][1].get("finals") == len(fin), f"{sid}: session_done finals = {len(fin)}")
check(sum(1 for _, o in st if o.get("event") == "ready") == (2 if a.drop_at else 1), "ready status on each connect")
txt = [o["text"] for o in res if o["type"] == "final" and o["text"]]
dup = [t for k, t in enumerate(txt) for u in txt[k + 1:k + 4] if len(t) >= 8 and t == u]
check(not dup, f"no repeated final text {dup[:2]}")
print(f"session_done after the last SESSION_END: {done_wait:.2f}s (depends on the runner: board NPU or Mac fake runner)")
print(f"finals {sum(o['type']=='final' for o in res)}, partials {sum(o['type']=='partial' for o in res)}, empty finals {sum(bool(o.get('empty')) for o in res)}")
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
