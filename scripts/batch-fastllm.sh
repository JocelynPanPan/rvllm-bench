#!/usr/bin/env bash

set -Eeuo pipefail

# ===================== 全局变量定义（fastllm） =====================
datasets_path="/opt/infer/datasets"
results_path="/opt/infer/results/batch-fastllm"
venv_root="/root/envs"              # 虚拟环境根路径，实际环境为 $venv_root/$lmul
model_path="/opt/models/Qwen2.5-0.5B-Instruct-int8.flm"

batch_array=(8 16)

# lmul数组（对应不同构建/配置/环境名）
lmul_array=(
  "fastllm-auto-gccO2"
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

activate_env() {
  local lmul="$1"
  local venv_dir="${venv_root}/${lmul}"
  if [ -d "$venv_dir" ]; then
    # 优先标准 venv 结构
    if [ -f "$venv_dir/bin/activate" ]; then
      # shellcheck disable=SC1090
      . "$venv_dir/bin/activate"
    else
      echo "未找到激活脚本: $venv_dir/bin/activate" >&2
      exit 1
    fi
  else
    echo "虚拟环境不存在: $venv_dir" >&2
    exit 1
  fi
}

deactivate_env() {
  if command -v deactivate >/dev/null 2>&1; then
    deactivate || true
  fi
}

start_server() {
  local lmul="$1"; shift
  local batch="$1"; shift

  kill_port_if_busy 8080
  activate_env "$lmul"

  # fastllm 启动命令
  log_info "启动 fastllm 服务: lmul=$lmul batch=$batch 端口=8080"
  python -m ftllm.server \
    --dtype int8 \
    --atype float16 \
    -t 4 \
    --max_batch "$batch" \
    -p "$model_path" \
    --port 8080 \
    --model_name "$lmul" \
    >/dev/null 2>&1 &
  server_pid=$!

  if ! wait_http_ready; then
    echo "服务端口未准备就绪，退出。" >&2
    kill "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
    deactivate_env
    exit 1
  fi
  # 通过就绪检测后，再额外等待30秒，确保服务完全可用
  log_info "服务就绪，额外等待30秒以确保完全启动"
  sleep 30
}

stop_server() {
  if [ -n "${server_pid:-}" ] && kill -0 "$server_pid" >/dev/null 2>&1; then
    log_info "停止服务 PID=$server_pid"
    kill "$server_pid" >/dev/null 2>&1 || true
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
  deactivate_env
}

detect_dataset_format() {
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
    grep -cve '^\s*$' "$file" || true
  fi
}

request_body_by_index() {
  # fastllm chat completions 请求体（兼容字段名）
  local file="$1"; local fmt="$2"; local idx="$3"; local model_name="$4"
  local item
  if [ "$fmt" = "array" ]; then
    item=$(jq -c ".[${idx}]" "$file")
  else
    item=$(sed -n "$((idx+1))p" "$file")
  fi

  # prompt 与 max_tokens 映射
  printf '%s' "$item" | jq -c --arg model "$model_name" '{
    model: $model,
    prompt: (.prompt // .input // .text // ""),
    max_tokens: ((.max_tokens // .n_predict // 128) | tonumber)
  }'
}

send_one_request_bg() {
  # 参数：seq file fmt result_dir model
  local seq="$1"; shift
  local file="$1"; shift
  local fmt="$1"; shift
  local result_dir="$1"; shift
  local model_name="$1"; shift

  (
    set +e
    local req_json
    req_json=$(request_body_by_index "$file" "$fmt" "$seq" "$model_name")

    local resp
    resp=$(curl http://127.0.0.1:8080/v1/chat/completions \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer something" \
            -d "$req_json" \
            --silent \
            --show-error 2>&1)

    local curl_rc=$?
    {
      echo "==== REQUEST $seq ===="
      echo "REQ: $req_json"
      echo "RESP: $resp"
      echo
    } >>"$result_dir/infer.txt"

    # 校验响应是否正常（JSON 且包含 usage.prompt_tokens 与 usage.completion_tokens）
    local ok=1
    if [ $curl_rc -eq 0 ]; then
      if printf '%s' "$resp" | jq -e '.usage and (.usage.prompt_tokens|type=="number") and (.usage.completion_tokens|type=="number")' >/dev/null 2>&1; then
        ok=0
      fi
    fi

    if [ $ok -ne 0 ]; then
      # 标记异常，供主循环感知并触发重试流程
      echo "$seq" >>"$result_dir/tmp/abnormal.flag"
      exit 0
    fi

    # fastllm 的响应需要从 usage 中取 tokens 统计
    local prompt_n
    local predicted_n
    prompt_n=$(printf '%s' "$resp" | jq -r '.usage.prompt_tokens')
    predicted_n=$(printf '%s' "$resp" | jq -r '.usage.completion_tokens')
    echo "$prompt_n $predicted_n" >"$result_dir/tmp/metrics_${seq}.txt"
  ) &
}

drop_caches() {
  echo 3 > /proc/sys/vm/drop_caches
}

# 删除当前 (batch,lmul) 配置从指定数据集索引开始的结果目录
remove_results_from_index() {
  local batch="$1"; local lmul="$2"; local start_idx="$3"
  local base_dir="${results_path}/${batch}/${lmul}"
  local n=${#dataset_file[@]}
  for ((ri=start_idx; ri<n; ri++)); do
    local ds="${dataset_file[$ri]}"
    local dsn="${ds%.*}"
    rm -rf "${base_dir}/${dsn}" || true
  done
}


# ===================== 主流程 =====================
ensure_dep jq
ensure_dep curl

mkdir -p "$results_path"

for batch in "${batch_array[@]}"; do
  for lmul in "${lmul_array[@]}"; do
    start_server "$lmul" "$batch"

    # 以索引遍历数据集，便于出现异常时从当前数据集起重试
    ds_count=${#dataset_file[@]}
    ds_idx=0
    while [ "$ds_idx" -lt "$ds_count" ]; do
      ds="${dataset_file[$ds_idx]}"
      ds_path="${datasets_path}/${ds}"
      if [ ! -f "$ds_path" ]; then
        echo "数据集不存在: $ds_path，跳过。" >&2
        ds_idx=$((ds_idx+1))
        continue
      fi

      ds_name_noext="${ds%.*}"
      result_dir="${results_path}/${batch}/${lmul}/${ds_name_noext}"
      mkdir -p "$result_dir/tmp"

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

      # 针对该数据集增加重试（最多3次）
      attempt=1
      max_attempt=3
      while :; do
        # 每次尝试前，确保结果目录与临时目录存在（可能在上次失败时被删除）
        mkdir -p "$result_dir/tmp"
        {
          echo "---- ATTEMPT ${attempt}/${max_attempt} ----"
          echo "TIME: $(date '+%F %T')"
        } >>"$result_dir/infer.txt"
        total_prompt=0
        total_pred=0
        completed=0
        rm -f "$result_dir"/tmp/metrics_*.txt "$result_dir"/tmp/abnormal.flag >/dev/null 2>&1 || true

        # 一次性发起所有请求
        for ((i=0; i<total; i++)); do
          send_one_request_bg "$i" "$ds_path" "$fmt" "$result_dir" "$lmul"
        done

        # 第一波发齐，记录开始时间（纳秒）
        start_ns=$(date +%s%N)

        dataset_failed=0
        while [ "$completed" -lt "$total" ]; do
          # 若检测到异常标志，宣告失败并退出本数据集尝试
          if [ -f "$result_dir/tmp/abnormal.flag" ]; then
            dataset_failed=1
            break
          fi
          for mf in "$result_dir"/tmp/metrics_*.txt; do
            [ -e "$mf" ] || break
            if [ -s "$mf" ]; then
              read -r p_n pr_n <"$mf" || true
              total_prompt=$((total_prompt + ${p_n:-0}))
              total_pred=$((total_pred + ${pr_n:-0}))
            fi
            rm -f "$mf" || true
            completed=$((completed+1))
          done
          if [ "$completed" -lt "$total" ]; then
            sleep 0.05
          fi
        done

        if [ "$dataset_failed" -eq 1 ]; then
          log_info "检测到服务异常，中断当前数据集并准备重试 (attempt=${attempt}/${max_attempt})"
          # 终止当前仍在运行的后台请求，避免对已删除目录继续写入
          bg_pids=$(jobs -pr || true)
          if [ -n "$bg_pids" ]; then
            kill $bg_pids >/dev/null 2>&1 || true
          fi
          # 清理从当前数据集起的结果，释放缓存并重启服务
          remove_results_from_index "$batch" "$lmul" "$ds_idx"
          drop_caches
          stop_server
          start_server "$lmul" "$batch"
          attempt=$((attempt+1))
          if [ "$attempt" -le "$max_attempt" ]; then
            continue
          else
            echo "在 $max_attempt 次重试后仍失败，退出。" >&2
            exit 1
          fi
        fi

        # 正常完成
        end_ns=$(date +%s%N)
        dur_ns=$((end_ns - start_ns))
        dur_s=$(awk -v ns="$dur_ns" 'BEGIN{printf "%.6f", ns/1000000000}')
        sum_tokens=$((total_prompt + total_pred))
        tput=$(awk -v tks="$sum_tokens" -v s="$dur_s" 'BEGIN{ if (s>0) printf "%.6f", tks/s; else print 0 }')

        {
          echo "SUMMARY: prompt_n=$total_prompt predicted_n=$total_pred"
          echo "TIME_S: $dur_s"
          echo "THROUGHPUT: $tput tokens/s"
          echo
        } >>"$result_dir/infer.txt"

        drop_caches
        break
      done

      # 当前数据集成功，进入下一数据集
      ds_idx=$((ds_idx+1))
    done

    stop_server
  done
done

log_info "fastllm 实验全部完成。"


