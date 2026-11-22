"""Speech segmenter shared by board_stream.py (board) and scripts/sim_segmenter.py (Mac cut-point check).
Feed fixed-size frames (energy VAD: 480 samples = 30 ms, silero: 512); get back finished segments.
  - A segment ends at a pause; the pause needed gets shorter as the segment grows (1.0 s / 0.5 s / 0.2 s).
  - When speech fills the encoder window without such a pause, it must be cut anyway:
      cut="last": quietest single frame in the last 2 s, next segment overlaps by `overlap` s (the 10/13 board runs)
      cut="dip" : quietest `dip_ms` stretch (energy averaged) between `dip_from` s and the window end. If it is at least
                  `dip_db` below the segment's speech level it is a word gap: cut there, no overlap, not marked forced.
                  Otherwise fall back to cut="last" (forced, overlap).
      cut="early": like "dip", but once the segment is longer than `soft_len` a word gap is cut as soon as it ends
                  (deepest point of the run of dip_k-frame windows >= dip_db), not when the window is full. The
                  decision is then ~0.1 s after the cut instead of up to ~4 s (dip waits for the window to fill).
Segments are (start_sample in the stream, audio, forced, info); info = {"kind": pause|dip|forced|end, "depth_db": dB below speech
level at the cut (None for pause/end), "decided": stream sample position when the cut was decided}."""
import numpy as np

SR = 16000


class Segmenter:
    def __init__(self, frame, window=10, min_len=3.0, soft_len=6.0, overlap=1.5, cut="last",
                 dip_from=5.0, dip_ms=150, dip_db=12.0, speech_prob=None, offset=0, adaptive=False, pauses=(1.0, 0.5, 0.2)):
        self.FD = FD = frame / SR
        self.frame, self.speech_prob, self.cut = frame, speech_prob, cut
        self.adaptive = adaptive                   # noise-aware thresholds: pause and word-gap tests follow the measured noise floor
        self.floor = -100.0                        # 5th percentile of the last 10 s (dB), the noise floor estimate
        self.offset = offset                       # stream sample position of this segmenter's first frame
        # pause that ends a segment: shorter than min_len / up to soft_len / longer (s)
        self.pre, self.long_sil, self.end_sil, self.short_sil = int(0.3 / FD), *(int(p / FD) for p in pauses)
        self.min_f, self.soft_f = int(min_len / FD), int(soft_len / FD)
        self.max_f, self.ov_f = int(window / FD) - 1, int(overlap / FD)
        self.dip_lo, self.dip_k, self.dip_db = int(dip_from / FD), max(1, round(dip_ms / 1000 / FD)), dip_db
        self.frames, self.hist = [], []
        self.start = self.pos = self.sil = 0
        self.speaking = False
        self.cand = None                           # cut="early": (depth, boundary) of the deepest window in the current dip

    def _emit(self, frames, forced, kind, depth=None):
        x = np.concatenate([f for f, _, _ in frames])
        return (self.offset + self.start * self.frame, x, forced,
                {"kind": kind, "depth_db": depth, "decided": self.offset + self.pos * self.frame})

    def tentative(self):
        """The pause segment this segment ends with if the silence goes on, as soon as its audio is fixed: a pause cut
        keeps 6 silent frames, so after 6 the audio no longer changes, only the decision waits for the full pause.
        Returns (start sample, audio) once, at that frame, or None (not there yet, or the cut happens right now)."""
        if not self.speaking or self.sil != 6: return None
        n = len(self.frames) - self.sil
        need = self.long_sil if n < self.min_f else self.end_sil if n < self.soft_f else self.short_sil
        if need <= 6: return None
        return self.offset + self.start * self.frame, np.concatenate([f for f, _, _ in self.frames])

    def depth_at(self, frames, i):
        """dB the `dip_k` frames centred on boundary i lie below the median level of the voiced frames."""
        k = self.dip_k; a, b = max(0, i - k // 2), max(0, i - k // 2) + k
        e = np.array([v for _, v, _ in frames])
        speech = np.median([v for _, v, vo in frames if vo] or [e.max()])
        return float(speech - 10 * np.log10(np.mean(10 ** (e[a:b] / 10))))

    def dip_need(self, frames):
        """dB a stretch must lie below speech to count as a word gap. A dip cannot go below the noise, so at low SNR
        a fixed 12 dB is never reached (board 11/02: 45 dip cuts -> 2 at 10 dB SNR). adaptive: at most 60 % of speech - noise."""
        if not self.adaptive: return self.dip_db
        speech = np.median([v for _, v, vo in frames if vo] or [self.floor])
        return float(np.clip(0.6 * (speech - self.floor), 4.0, self.dip_db))

    def _cut_last(self, frames):
        lo = max(len(frames) - int(2 / self.FD), 1)
        return min(range(lo, len(frames)), key=lambda i: frames[i][1]) + 1

    def _cut_dip(self, frames):
        """Boundary (frame index) at the centre of the quietest dip_k-frame stretch in [dip_lo, end)."""
        k = self.dip_k
        p = 10 ** (np.array([v for _, v, _ in frames]) / 10)
        m = np.convolve(p, np.ones(k) / k, mode="valid")            # m[j] = mean power of frames j..j+k-1
        # early: only look after soft_len. A dip in [dip_from, soft_len) found here is cut 5-6 s into the segment but
        # decided only when the window is full, ~4 s later (board 10/16: latency max ~5.5 s on talk17/36)
        start = max(self.dip_lo, self.soft_f) if self.cut == "early" else self.dip_lo
        lo, hi = max(start - k // 2, 0), len(m)
        if lo >= hi: return None
        j = lo + int(np.argmin(m[lo:hi]))
        return j + k // 2 + 1

    def feed(self, f):
        out = []
        if self.speech_prob is not None:
            v = self.speech_prob(f); voiced = v > 0.5
        else:
            v = 10 * np.log10(np.mean(f ** 2) + 1e-10)
            self.hist.append(v); self.hist = self.hist[-int(10 / self.FD):]
            self.floor = np.percentile(self.hist, 5)
            # old rule caps the threshold at -39 dB: anything louder (fan, loud room) is always "voiced" and pauses disappear
            voiced = v > max((self.floor if self.adaptive else min(self.floor, -45)) + 6, -55)
        self.frames.append((f, v, voiced)); self.pos += 1
        if not self.speaking:
            self.frames = self.frames[-self.pre:]
            if voiced: self.speaking = True; self.sil = 0; self.start = self.pos - len(self.frames)
            return out
        self.sil = 0 if voiced else self.sil + 1
        n = len(self.frames) - self.sil
        need = self.long_sil if n < self.min_f else self.end_sil if n < self.soft_f else self.short_sil
        if self.sil >= need:
            out.append(self._emit(self.frames[:len(self.frames) - self.sil + 6], False, "pause"))
            self.frames, self.speaking, self.cand = [], False, None
            return out
        if self.cut == "early" and self.speech_prob is None and len(self.frames) >= self.soft_f + self.dip_k:
            b = len(self.frames) - self.dip_k // 2              # boundary at the centre of the last dip_k frames
            d = self.depth_at(self.frames, b)
            if d >= self.dip_need(self.frames):
                if self.cand is None or d > self.cand[0]: self.cand = (d, b)
            elif self.cand is not None:                         # the dip just ended: cut at its deepest point
                d, c = self.cand; self.cand = None
                out.append(self._emit(self.frames[:c], False, "dip", d))
                self.frames = self.frames[c:]; self.start += c
                return out
        if len(self.frames) >= self.max_f:
            fr = self.frames; self.cand = None
            if self.cut in ("dip", "early") and self.speech_prob is None:
                c = self._cut_dip(fr)
                if c is not None and (d := self.depth_at(fr, c)) >= self.dip_need(fr):
                    out.append(self._emit(fr[:c], False, "dip", d))
                    self.frames = fr[c:]; self.start += c
                    return out
            c = self._cut_last(fr)
            out.append(self._emit(fr[:c], True, "forced", self.depth_at(fr, c) if self.speech_prob is None else None))
            back = max(c - self.ov_f, 0); self.frames = fr[back:]; self.start += back
        return out

    def flush(self):
        if self.speaking and self.frames: return [self._emit(self.frames, False, "end")]
        return []
