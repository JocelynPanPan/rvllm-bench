#!/usr/bin/env bash

set -Eeuo pipefail

# ===================== 全局变量定义 =====================
datasets_path="/opt/infer/datasets"
results_path="/opt/infer/results/batch-llama.cpp"
server_base="/opt/llm"  # 服务路径基础目录
model_path="/opt/models/Qwen2.5-0.5B-Instruct-int8.gguf"

batch_array=(8 16)

# lmul数组（对应不同构建/配置目录）
lmul_array=(
  "llama.cpp-b4977-auto-gccO2"
)

# 数据集文件数组（位于 $datasets_path 下）
dataset_file=(
    "short2short.json"
    "short2long.json"
    "long2short.json"
    "long2long.json"
)


# ===================== 工具与通用函数 =====================
log_info() {
  printf "[%s] %s\n" "$(date '+%F %T')" "$*"
}

ensure_dep() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "依赖缺失: $cmd，请先安装后再运行。" >&2
    exit 1
  fi
}

min() { [ "$1" -le "$2" ] && echo "$1" || echo "$2"; }

kill_port_if_busy() {
  local port="$1"
  # 尝试多种方式释放端口（忽略错误）
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
  fi
  if command -v lsof >/dev/null 2>&1; then
    local pids
    pids=$(lsof -ti:"$port" || true)
    if [ -n "$pids" ]; then
      kill -9 $pids >/dev/null 2>&1 || true
    fi
  fi
}

wait_http_ready() {
  # 等待端口就绪（最多60秒）
  local url="http://127.0.0.1:8080"
  for i in $(seq 1 60); do
    if curl -s -m 1 -o /dev/null "$url/health"; then
      return 0
    fi
    if curl -s -m 1 -o /dev/null "$url"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_server() {
  local lmul="$1"; shift
  local batch="$1"; shift

  kill_port_if_busy 8080

  local srv_dir="${server_base}/${lmul}/build/bin"
  if [ ! -d "$srv_dir" ]; then
    echo "找不到服务目录: $srv_dir" >&2
    exit 1
  fi

  pushd "$srv_dir" >/dev/null
  if [ ! -x ./llama-server ]; then
    echo "在 $srv_dir 未找到可执行的 ./llama-server" >&2
    popd >/dev/null
    exit 1
  fi

  log_info "启动服务: lmul=$lmul batch=$batch 端口=8080"
  # 后台启动
  ./llama-server \
    -m "$model_path" \
    -c 32000 \
    --batch-size "$batch" \
    -np "$batch" \
    --port 8080 \
    --host 0.0.0.0 \
    >/dev/null 2>&1 &
  server_pid=$!

  if ! wait_http_ready; then
    echo "服务端口未准备就绪，退出。" >&2
    kill "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
    popd >/dev/null
    exit 1
  fi
  # 通过就绪检测后，再额外等待30秒，确保服务完全可用
  log_info "服务就绪，额外等待30秒以确保完全启动"
  sleep 30
  popd >/dev/null
}

stop_server() {
  if [ -n "${server_pid:-}" ] && kill -0 "$server_pid" >/dev/null 2>&1; then
    log_info "停止服务 PID=$server_pid"
    kill "$server_pid" >/dev/null 2>&1 || true
    # 优雅等待2秒，仍未退出则强杀
    for _ in 1 2; do
      if ! kill -0 "$server_pid" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    kill -9 "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
  kill_port_if_busy 8080
}

detect_dataset_format() {
  # 输出 array 或 jsonl
  local file="$1"
  local first
  first=$(tr -d '\r' <"$file" | sed -n '1{/^\s*$/d;p;q}') || true
  if [ "${first:0:1}" = "[" ]; then
    echo "array"
  else
    echo "jsonl"
  fi
}

dataset_length() {
  local file="$1"; local fmt="$2"
  if [ "$fmt" = "array" ]; then
    jq -r 'length' "$file"
  else
    # 非空行计数
    grep -cve '^\s*$' "$file" || true
  fi
}

request_body_by_index() {
  # 按索引（0-based）从数据集中抽取一条并构造 llama.cpp /completion 请求体
  local file="$1"; local fmt="$2"; local idx="$3"
  local item
  if [ "$fmt" = "array" ]; then
    item=$(jq -c ".[${idx}]" "$file")
  else
    # JSONL 第 idx+1 行
    item=$(sed -n "$((idx+1))p" "$file")
  fi

  # 统一映射字段：prompt、n_predict（默认128），禁用流式
  printf '%s' "$item" | jq -c '{
    prompt: (.prompt // .input // .text // ""),
    n_predict: ((.max_tokens // .n_predict // 128) | tonumber),
    stream: false
  }'
}

send_one_request_bg() {
  # 参数：seq file fmt result_dir
  local seq="$1"; shift
  local file="$1"; shift
  local fmt="$1"; shift
  local result_dir="$1"; shift

  (
    set +e
    local req_json
    req_json=$(request_body_by_index "$file" "$fmt" "$seq")

    local resp
    resp=$(curl --request POST \
            --url http://127.0.0.1:8080/completion \
            --header "Content-Type: application/json" \
            --data "$req_json" \
            --silent \
            --show-error 2>&1)

    local curl_rc=$?
    # 将原始响应与请求索引写入总日志
    {
      echo "==== REQUEST $seq ===="
      echo "REQ: $req_json"
      echo "RESP: $resp"
      echo
    } >>"$result_dir/infer.txt"

    local prompt_n=0
    local predicted_n=0
    if [ $curl_rc -eq 0 ]; then
      # 兼容多种返回字段
      prompt_n=$(printf '%s' "$resp" | jq -r '(.prompt_n // .tokens_evaluated // 0) | tonumber' 2>/dev/null || echo 0)
      predicted_n=$(printf '%s' "$resp" | jq -r '(.predicted_n // .tokens_predicted // .n_predict // 0) | tonumber' 2>/dev/null || echo 0)
    fi
    echo "$prompt_n $predicted_n" >"$result_dir/tmp/metrics_${seq}.txt"
  ) &
}

drop_caches() {
  echo 3 > /proc/sys/vm/drop_caches
}


# ===================== 主流程 =====================
ensure_dep jq
ensure_dep curl

mkdir -p "$results_path"

for batch in "${batch_array[@]}"; do
  for lmul in "${lmul_array[@]}"; do
    start_server "$lmul" "$batch"

    for ds in "${dataset_file[@]}"; do
      ds_path="${datasets_path}/${ds}"
      if [ ! -f "$ds_path" ]; then
        echo "数据集不存在: $ds_path，跳过。" >&2
        continue
      fi

      ds_name_noext="${ds%.*}"
      result_dir="${results_path}/${batch}/${lmul}/${ds_name_noext}"
      mkdir -p "$result_dir/tmp"

      # 记录配置头
      {
        echo "===================="
        echo "LMUL: $lmul"
        echo "BATCH: $batch"
        echo "DATASET: $ds"
        echo "MODEL: $model_path"
        echo "TIME: $(date '+%F %T')"
        echo "===================="
      } >>"$result_dir/infer.txt"

      fmt=$(detect_dataset_format "$ds_path")
      total=$(dataset_length "$ds_path" "$fmt")
      if [ -z "$total" ] || [ "$total" -le 0 ]; then
        echo "数据集为空: $ds_path，跳过。" >&2
        continue
      fi

      conc="$total"
      log_info "开始实验: lmul=$lmul batch=$batch dataset=$ds (共$total条, 并发$conc)"

      # 调度与统计
      total_prompt=0
      total_pred=0
      completed=0

      # 清理可能残留的度量文件，并一次性发起所有请求
      rm -f "$result_dir"/tmp/metrics_*.txt >/dev/null 2>&1 || true
      for ((i=0; i<total; i++)); do
        send_one_request_bg "$i" "$ds_path" "$fmt" "$result_dir"
      done

      # 第一波发齐，记录开始时间（纳秒）
      start_ns=$(date +%s%N)

      # 轮询已完成的请求，完成一个就补一个，直到所有请求完成
      while [ "$completed" -lt "$total" ]; do
        for mf in "$result_dir"/tmp/metrics_*.txt; do
          [ -e "$mf" ] || break
          # 读取并删除，避免重复累计
          if [ -s "$mf" ]; then
            read -r p_n pr_n <"$mf" || true
            total_prompt=$((total_prompt + ${p_n:-0}))
            total_pred=$((total_pred + ${pr_n:-0}))
          fi
          rm -f "$mf" || true
          completed=$((completed+1))
        done
        # 若尚未完成全部，则小憩避免busy-wait
        if [ "$completed" -lt "$total" ]; then
          sleep 0.05
        fi
      done

      # 所有请求完成，记录结束时间
      end_ns=$(date +%s%N)
      dur_ns=$((end_ns - start_ns))
      # 纳秒转秒（保留6位小数）
      dur_s=$(awk -v ns="$dur_ns" 'BEGIN{printf "%.6f", ns/1000000000}')
      sum_tokens=$((total_prompt + total_pred))
      tput=$(awk -v tks="$sum_tokens" -v s="$dur_s" 'BEGIN{ if (s>0) printf "%.6f", tks/s; else print 0 }')

      {
        echo "SUMMARY: prompt_n=$total_prompt predicted_n=$total_pred"
        echo "TIME_S: $dur_s"
        echo "THROUGHPUT: $tput tokens/s"
        echo
      } >>"$result_dir/infer.txt"

      # 释放缓存，继续下一个数据集
      drop_caches
    done

    # 一组(lmul,batch)结束，停止服务
    stop_server
  done
done

log_info "全部实验完成。"


