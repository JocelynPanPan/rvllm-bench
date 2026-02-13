#!/bin/bash

# 全局变量定义
datasets_path="/opt/infer/datasets"
results_path="/opt/infer/results/llama.cpp"
server_path="/opt/llm/"  # 服务路径基础目录

# 模型数组
models=(
    "Qwen2.5-0.5B-Instruct-int8.gguf"
    "Qwen2.5-3B-Instruct-int8.gguf"
    "Qwen2.5-7B-Instruct-int8.gguf"
    "Llama-2-7b-chat-hf-int8.gguf"
)

# lmul数组
lmul_array=(
    "llama.cpp-b4977-auto-gccO2"
)

# 数据集文件数组
dataset_file=(
    "short2short.json"
    "short2long.json"
    "long2short.json"
    "long2long.json"
)

# 遍历lmul数组
for lmul in "${lmul_array[@]}"; do
    echo "=================================================="
    echo "Processing lmul configuration: $lmul"
    echo "=================================================="
    
    # 构建完整的服务路径
    full_server_path="${server_path}${lmul}/build/bin"
    
    # 遍历models数组
    for model in "${models[@]}"; do
        echo "--------------------------------------------------"
        echo "Processing model: $model"
        echo "--------------------------------------------------"
        
        # 启动推理服务
        echo "Starting llama-server for model $model..."
        cd "$full_server_path"
        ./llama-server -m /opt/models/$model -c 22000 -t 8 --batch-size 16 -np 16 --port 8080 --host 0.0.0.0 > /dev/null 2>&1 &
        infer_pid=$!
        
        echo "Inference service started with PID: $infer_pid"
        echo "Waiting 20 seconds for service to initialize..."
        sleep 60
        
        # 检查推理进程是否存在
        if ! kill -0 $infer_pid 2>/dev/null; then
            echo "Error: Failed to start inference service for model $model"
            exit 1
        fi
        
        echo "Service initialized successfully"
        echo "Results will be saved to: $results_path/$lmul/$model/"
        
        # 遍历数据集文件
        for dataset in "${dataset_file[@]}"; do
            echo "Processing dataset: $dataset"
            
            # 检查数据集文件是否存在
            dataset_path="$datasets_path/$dataset"
            if [ ! -f "$dataset_path" ]; then
                echo "Warning: Dataset file $dataset_path not found, skipping..."
                continue
            fi
            
            # 获取数据集文件名（不含扩展名）
            dataset_name=$(basename "$dataset" .json)
            
            # 创建结果目录
            base_result_dir="$results_path/$lmul/$model/$dataset_name"
            mkdir -p "$base_result_dir"
            
            # 解析JSON文件并处理每个推理请求
            index=0
            while IFS= read -r line; do
                # 跳过空行和非对象行
                if [[ ! "$line" =~ ^[[:space:]]*\{.*\}[[:space:]]*,?[[:space:]]*$ ]]; then
                    continue
                fi
                
                # 清理行末的逗号
                clean_line=$(echo "$line" | sed 's/,$//')
                
                # 提取prompt和max_tokens
                prompt=$(echo "$clean_line" | jq -r '.prompt')
                max_tokens=$(echo "$clean_line" | jq -r '.max_tokens')
                
                # 检查提取是否成功
                if [ "$prompt" = "null" ] || [ "$max_tokens" = "null" ]; then
                    continue
                fi
                
                echo "Processing request $index: prompt length=$(echo -n "$prompt" | wc -c), max_tokens=$max_tokens"
                
                # 创建当前请求的结果目录
                result_dir="$base_result_dir/$index"
                mkdir -p "$result_dir"
                
                # 准备JSON数据
                json_data=$(jq -n --arg prompt "$prompt" --argjson max_tokens "$max_tokens" '{prompt: $prompt, n_predict: $max_tokens}')
                
                # 发起推理请求
                echo "Sending inference request $index..."
                
                
                # 发起推理请求并等待完成
                curl --request POST \
                    --url http://localhost:8080/completion \
                    --header "Content-Type: application/json" \
                    --data "$json_data" \
                    --silent \
                    --show-error > "$result_dir/infer.txt" 2>&1
                
                curl_exit_code=$?
                
                
                # 检查推理请求是否成功
                if [ $curl_exit_code -eq 0 ]; then
                    echo "Request $index completed successfully"
                else
                    echo "Warning: Request $index failed with exit code $curl_exit_code"
                fi
                
                # 检查推理进程是否仍在运行
                if ! kill -0 $infer_pid 2>/dev/null; then
                    echo "Error: Inference process $infer_pid has terminated"
                    exit 1
                fi
                
                # 增加索引
                ((index++))
                
                # 请求间隔
                sleep 2
                
            done < <(jq -c '.[]' "$dataset_path")
            
            echo "Completed dataset: $dataset (processed $index requests)"
            
            # 清理缓存
            echo "Clearing cache..."
            sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
            sleep 2
            
        done  # 结束数据集循环
        
        # 停止当前模型的推理服务
        echo "Stopping inference service (PID: $infer_pid) for model $model..."
        kill -TERM $infer_pid 2>/dev/null
        wait $infer_pid 2>/dev/null
        sleep 2
        
        echo "Completed all datasets for model: $model"
        
    done  # 结束models循环
    
    echo "Completed all models for lmul: $lmul"
    
done  # 结束lmul循环

echo "=================================================="
echo "All inference benchmarks completed successfully!"
echo "=================================================="
echo "Results saved in: $results_path/"

# 显示结果目录结构
echo ""
echo "Result directory structure:"
find "$results_path" -type f -name "*.txt" | head -20
if [ $(find "$results_path" -type f -name "*.txt" | wc -l) -gt 20 ]; then
    echo "... and more files"
fi
