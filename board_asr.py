"""Transcribe wav files on the Rubik Pi NPU with the persistent runner (whisper_npu).
Needs only numpy. Mel spectrogram, token decoding and CER are done here; encoder/decoder run in whisper_npu.
Usage (on the board):
  python3 board_asr.py --window 10 file1.wav file2.wav ...
  python3 board_asr.py --window 10 --tsv fleurs70/refs.tsv        # tsv: wav<TAB>reference, prints CER
Bins default to ~/rubikpi/bins/{enc,dec}_{window}s.bin"""
import argparse, json, os, pathlib, re, struct, subprocess, sys, time, wave, signal
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SR, HOP, NFFT = 16000, 160, 400
PREFIX = [50258, 50264, 50359, 50363]          # <|startoftranscript|> <|ko|> <|transcribe|> <|notimestamps|>
EOT = 50257


class WhisperNPU:
    def __init__(self, enc, dec, runner=None):
        runner = runner or os.environ.get("WHISPER_RUNNER", str(HERE / "whisper_npu"))   # env override: Mac test with fake_runner.py
        env = dict(os.environ)
        vocab = pathlib.Path(str(dec)[:-4] + ".vocab")      # pruned-vocab decoder: dec_10s_pv.bin + dec_10s_pv.vocab
        if vocab.exists(): env["WHISPER_VOCAB"] = str(vocab)
        self.p = subprocess.Popen(runner.split() + [str(enc), str(dec)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)

    def _read(self, n):
        b = self.p.stdout.read(n)
        if len(b) != n: sys.exit("runner died (see its stderr above)")
        return b

    def abortable(self, on):
        """Following decodes may (on) or may not be cut short by abort(). Also forgets an earlier abort()."""
        self.p.stdin.write(b"A" + bytes([1 if on else 0])); self.p.stdin.flush()

    def abort(self):
        """Stop the running abortable decode after its current step (it returns the tokens so far)."""
        if self.p.poll() is None and not os.environ.get("WHISPER_RUNNER"): os.kill(self.p.pid, signal.SIGUSR1)

    def encode(self, mel):                     # mel: (80, T) float32
        self.p.stdin.write(b"E" + np.ascontiguousarray(mel, np.float32).tobytes()); self.p.stdin.flush()
        return struct.unpack("<f", self._read(4))[0]

    def decode(self, max_new=120, seconds=0.0, forced=()):   # seconds: audio length -> repeat stop in the runner (WHISPER_GUARD=0: off)
        """forced: text tokens fed after PREFIX as given (not returned; the result continues after them)."""
        if os.environ.get("WHISPER_GUARD") == "0": seconds = 0.0
        pre = PREFIX + list(forced)
        self.p.stdin.write(b"G" + struct.pack("<f", seconds) + b"D" + struct.pack(f"<i{len(pre)}ii", len(pre), *pre, max_new)); self.p.stdin.flush()
        n = struct.unpack("<i", self._read(4))[0]
        toks = list(struct.unpack(f"<{n}i", self._read(4 * n))) if n else []
        tot, npu, self.last_lp = struct.unpack("<fff", self._read(12))     # avg log prob of the chosen tokens
        return toks, tot, npu

    def close(self):
        self.p.stdin.write(b"Q"); self.p.stdin.flush(); self.p.wait()


_filters = np.load(HERE / "mel_filters.npy")   # (201, 80), same as the HF Whisper feature extractor
_window = np.hanning(NFFT + 1)[:-1].astype(np.float32)


def log_mel(audio, seconds):
    """Whisper log-mel for a fixed input length (pads or cuts audio to `seconds`)."""
    n = seconds * SR
    x = np.zeros(n, np.float32); x[:min(len(audio), n)] = audio[:n]
    x = np.pad(x, NFFT // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(x, NFFT)[::HOP][: n // HOP]
    power = np.abs(np.fft.rfft(frames * _window, axis=1)) ** 2          # (T, 201)
    mel = np.log10(np.maximum(power @ _filters, 1e-10)).T                # (80, T)
    mel = np.maximum(mel, mel.max() - 8.0)
    return ((mel + 4.0) / 4.0).astype(np.float32)


_vocab = {int(k): bytes.fromhex(v) for k, v in json.load(open(HERE / "vocab_bytes.json")).items()}


def detok(ids):
    # Bytes that do not form a complete UTF-8 character are dropped: the decoder sometimes starts a Hangul syllable as byte
    # tokens and abandons it ('김종<eb a0> 열입니다'); shown as U+FFFD they were junk, and completing them made CER worse
    # (board, TEDx 8 talks, 457 finals: U+FFFD 9 -> 0 with CER unchanged; forcing completion 5.62% -> 5.72%).
    return b"".join(_vocab.get(i, b"") for i in ids if i < EOT).decode("utf-8", errors="ignore").strip()


def read_wav(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2, f"{path}: need 16 kHz 16-bit wav"
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        return x.reshape(-1, w.getnchannels()).mean(1)


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return d[-1]


def invented(text, lp, lp_min=-0.5, rep_max=0.5):
    """Probably not real speech: low model confidence, or one word making up most of the text (distant voices, clicks)."""
    w = text.split()
    return lp < lp_min or (len(w) >= 4 and max(w.count(x) for x in set(w)) / len(w) >= rep_max)


norm = lambda s: re.sub(r"[^\w]", "", s)
# Standalone hesitation sounds (whole words only; ambiguous real words like 그/저/아 are kept).
FILLER = re.compile(r"(?<!\S)(?:어+|음+|으+음|에+|흠+|엄+)[~.,!?…]*(?!\S)")
strip_fillers = lambda s: FILLER.sub(" ", s)


def align(ref, hyp):
    """Character alignment. Returns (ops, n_ins, ins_at): ops[i] in 'ok'/'sub'/'del' for each ref char,
    n_ins = inserted hyp chars, ins_at[i] = insertions right before ref char i (index len(ref) = at the end)."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), np.int32); D[:, 0] = np.arange(n + 1); D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        row, prev = D[i], D[i - 1]
        sub = prev[:-1] + (np.frombuffer(hyp.encode("utf-32-le"), np.uint32) != ord(ref[i - 1]))
        best = np.minimum(prev[1:] + 1, sub)
        c = np.concatenate([[i], best - np.arange(1, m + 1)])   # insertions: row[j] = min_k<=j (c[k]) + j
        row[:] = np.minimum.accumulate(c) + np.arange(m + 1)
    ops, ins_at = [""] * n, [0] * (n + 1); i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops[i - 1] = "ok" if ref[i - 1] == hyp[j - 1] else "sub"; i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1: ops[i - 1] = "del"; i -= 1
        else: ins_at[i] += 1; j -= 1
    return ops, sum(ins_at), ins_at

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="*")
    ap.add_argument("--window", type=int, default=10, help="encoder input length (s): 10/15/20/30")
    ap.add_argument("--tsv", help="wav<TAB>reference per line (paths relative to the tsv)")
    ap.add_argument("--bins", default=os.path.expanduser("~/rubikpi/bins"))
    ap.add_argument("--dec", help="decoder .bin (default <bins>/dec_<window>s.bin), e.g. dec_10s_pv.bin")
    a = ap.parse_args()
    items = [(w, None) for w in a.wavs]
    if a.tsv:
        base = pathlib.Path(a.tsv).parent
        items += [(base / l.split("\t")[0], l.rstrip("\n").split("\t")[1]) for l in open(a.tsv, encoding="utf-8") if l.strip()]
    t_load = time.perf_counter()
    m = WhisperNPU(f"{a.bins}/enc_{a.window}s.bin", a.dec if a.dec and "/" in a.dec else f"{a.bins}/{a.dec or f'dec_{a.window}s.bin'}")
    errs = errs_f = chars = 0; rows = []; steps = []
    NPRE = 1 if os.environ.get("WHISPER_PRE") else len(PREFIX)   # decoder runs for the prefix (prefix decoder: 1)
    for i, (w, ref) in enumerate(items):
        x = read_wav(w)
        t0 = time.perf_counter()
        mel = log_mel(x, a.window); t_mel = time.perf_counter() - t0
        enc_ms = m.encode(mel)
        toks, dec_ms, npu_ms = m.decode(seconds=min(len(x) / SR, a.window))
        total = time.perf_counter() - t0
        text = detok(toks)
        if i == 0: print(f"[startup incl. model load] {time.perf_counter() - t_load:.1f}s", flush=True)
        line = (f"{pathlib.Path(w).name} | audio {len(x)/SR:4.1f}s | mel {t_mel*1000:3.0f}ms enc {enc_ms:3.0f}ms "
                f"dec {dec_ms:4.0f}ms ({len(toks)+NPRE+1} steps, {dec_ms/(len(toks)+NPRE+1):.1f}ms/step, npu {npu_ms:4.0f}ms) "
                f"total {total*1000:4.0f}ms | {text}")
        if ref is not None:
            e = edit_distance(norm(ref), norm(text)); errs += e; chars += len(norm(ref))
            errs_f += edit_distance(norm(ref), norm(strip_fillers(text)))
            line += f" | CER {e/max(len(norm(ref)),1)*100:.0f}%"
        print(line, flush=True); rows.append((len(x) / SR, total)); steps.append((dec_ms, len(toks) + NPRE + 1))
    m.close()
    if chars: print(f"CER {errs/chars*100:.1f}% ({errs}/{chars} chars, {len(items)} files)")
    if chars: print(f"CER_NOFILLER {errs_f/chars*100:.1f}% ({errs_f}/{chars} chars, hesitation sounds removed)")
    if rows:
        au, tt = np.array(rows).T
        print(f"avg total {tt.mean()*1000:.0f}ms for avg {au.mean():.1f}s audio, max {tt.max()*1000:.0f}ms; RTF {tt.sum()/au.sum():.3f}; "
              f"decoder {sum(d for d, _ in steps)/sum(n for _, n in steps):.1f}ms/step")
