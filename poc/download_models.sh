#!/usr/bin/env bash
set -u
base="https://hf-mirror.com"
fetch() { # url out
  local url="$1" out="$2" i
  for i in 1 2 3; do
    echo "[fetch $i] $url"
    curl -L --fail --connect-timeout 15 --retry 2 -C - -o "$out" "$url" && { echo "OK: $out"; return 0; }
    sleep 3
  done
  echo "FAILED: $url"; return 1
}
fetch "$base/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/model.int8.onnx" poc/models/sensevoice/model.int8.onnx
fetch "$base/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/tokens.txt" poc/models/sensevoice/tokens.txt
fetch "$base/Xenova/bge-small-zh-v1.5/resolve/main/onnx/model.onnx" poc/models/bge-small-zh/model.onnx
fetch "$base/Xenova/bge-small-zh-v1.5/resolve/main/tokenizer.json" poc/models/bge-small-zh/tokenizer.json
echo "=== ALL DONE ==="
ls -la poc/models/sensevoice poc/models/bge-small-zh
