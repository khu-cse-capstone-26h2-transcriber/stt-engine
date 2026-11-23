#!/bin/sh
# STT engine for audiostackd on the board: connects to the engine socket, sends partial/final results back, reconnects
# forever (keeps the model loaded). Protocol and result format: ENGINE_CONTRACT.md. Service: whisper-engine.service.
#   sh run_engine.sh                                   # /run/audiostack/engine.sock, head-split 5,557-token decoder, partial every 2 s
#   ASTK_ENGINE_SOCKET=/tmp/astk/engine.sock DEC=dec_10s_pv.bin PARTIAL=0 sh run_engine.sh --once
# Env: ASTK_ENGINE_SOCKET, DEC (decoder .bin in ~/rubikpi/bins), PARTIAL (s, 0 = off), CUT (early|dip|last),
#      NOISE (1 = "--adapt --lp-min -0.5", default; 0 = off), STABLE (1 = "--stable final --hold-last", default; 0 = off),
#      PAUSES (silence that ends a segment, s, for segments <3 s, 3-6 s, longer; default 0.4,0.25,0.2),
#      EXTRA (more board_stream.py options),
#      WHISPER_ZC (1 = zero-copy, default)
# Exit code 2 = inference failed (whisper_npu died); systemd restarts it.
cd "$(dirname "$0")"
export WHISPER_ZC=${WHISPER_ZC:-1}
[ "${NOISE:-1}" = 0 ] && NOISE_OPTS= || NOISE_OPTS="--adapt --lp-min -0.5"
[ "${STABLE:-1}" = 0 ] && STABLE_OPTS= || STABLE_OPTS="--stable final --hold-last"
exec sh run_stream.sh --astk "${ASTK_ENGINE_SOCKET:-/run/audiostack/engine.sock}" --dec "${DEC:-dec_10s_pv_h.bin}" \
  --cut "${CUT:-early}" --partial "${PARTIAL:-2}" --pauses "${PAUSES:-0.4,0.25,0.2}" $NOISE_OPTS $STABLE_OPTS $EXTRA "$@"
