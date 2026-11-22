"""Mac stand-in for whisper_npu: same stdin/stdout protocol, runs ONNX models on ONNX Runtime.
Tests board_asr.py / board_stream.py end to end without the board.
  WHISPER_RUNNER="python3 fake_runner.py" python3 board_asr.py --bins <dir with enc_10s.onnx, dec_10s.onnx> ...
Args (given by WhisperNPU): encoder.bin decoder.bin -> *.bin is replaced by *.onnx"""
import struct, sys
import numpy as np, onnxruntime as ort
enc, dec = (p.replace(".bin", ".onnx") for p in sys.argv[1:3])
so = ort.SessionOptions(); so.log_severity_level = 3
E, D = ort.InferenceSession(enc, so), ort.InferenceSession(dec, so)
d_in = {i.name: i for i in D.get_inputs()}; d_out = [o.name for o in D.get_outputs()]; e_out = [o.name for o in E.get_outputs()]
T = E.get_inputs()[0].shape[-1]; CTX = d_in["attention_mask"].shape[-1]
rd, wr = sys.stdin.buffer, sys.stdout.buffer
import os
vmap = np.fromfile(os.environ["WHISPER_VOCAB"], np.int32) if os.environ.get("WHISPER_VOCAB") else None
cross = None
while (c := rd.read(1)):
    if c == b"Q": break
    if c == b"E":
        mel = np.frombuffer(rd.read(80 * T * 4), np.float32).reshape(1, 80, T)
        cross = dict(zip(e_out, E.run(None, {E.get_inputs()[0].name: mel})))
        wr.write(struct.pack("<f", 0.0)); wr.flush()
    elif c == b"G": rd.read(4)
    elif c == b"A": rd.read(1)                            # abortable flag: the fake runner never aborts
    elif c == b"D":
        n = struct.unpack("<i", rd.read(4))[0]; pre = struct.unpack(f"<{n}i", rd.read(4 * n)); max_new = struct.unpack("<i", rd.read(4))[0]
        feed = {k: np.zeros(d_in[k].shape, np.float32) for k in d_in if "self" in k}
        mask = np.full((1, 1, 1, CTX), -100.0, np.float32); tok, out = pre[0], []; lps = []
        for step in range(CTX - 1):
            mask[..., CTX - step - 1] = 0.0
            res = D.run(None, {**feed, **cross, "input_ids": np.array([[tok]], np.int32), "attention_mask": mask, "position_ids": np.array([step], np.int32)})
            for k, v in zip(d_out[1:], res[1:]): feed[k[:-4] + "_in"] = v
            if step + 1 < n: tok = pre[step + 1]; continue
            z = res[0].reshape(-1).astype(np.float64); k = int(z.argmax()); lps.append(float(-np.log(np.exp(z - z[k]).sum())))
            tok = int(vmap[k]) if vmap is not None else k
            if tok == 50257 or len(out) >= max_new: break
            out.append(tok)
        wr.write(struct.pack(f"<i{len(out)}ifff", len(out), *out, 0.0, 0.0, float(np.mean(lps)) if lps else 0.0)); wr.flush()
