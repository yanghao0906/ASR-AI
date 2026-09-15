#!/usr/bin/env bash
# 下载 sherpa-onnx v1.13.8 的 Java 包并安装到本地 Maven 仓库。
# 官方未发布 Maven Central，只能取 GitHub Release 资产；
# GitHub 直连不可用时走 gh-proxy 代理，失败再回退 SourceForge 镜像。
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p libs
V=1.13.8
API=sherpa-onnx-jvm-$V.jar
NAT=sherpa-onnx-native-lib-win-x64-$V.jar

dl() { # <github文件名> <输出路径>
  local name="$1" out="$2" i url
  if [ -s "$out" ]; then echo "已存在: $out"; return 0; fi
  for i in 1 2 3; do
    url="https://gh-proxy.com/https://github.com/k2-fsa/sherpa-onnx/releases/download/v$V/$name"
    echo "[try $i] $url"
    curl -sL --fail --connect-timeout 20 -o "$out" "$url" && return 0
    url="https://sourceforge.net/projects/sherpa-onnx.mirror/files/v$V/$name/download"
    echo "[try $i] $url"
    curl -sL --fail --connect-timeout 20 -o "$out" "$url" && return 0
    sleep 3
  done
  echo "下载失败: $name" >&2; return 1
}

dl "$API" "libs/$API"
dl "$NAT" "libs/$NAT"

mvn -q install:install-file -Dfile="libs/$API" -DgroupId=com.k2fsa.sherpa.onnx \
  -DartifactId=sherpa-onnx-jvm -Dversion=$V -Dpackaging=jar
mvn -q install:install-file -Dfile="libs/$NAT" -DgroupId=com.k2fsa.sherpa.onnx \
  -DartifactId=sherpa-onnx-native-lib-win-x64 -Dversion=$V -Dpackaging=jar
echo "== 安装完成 =="
