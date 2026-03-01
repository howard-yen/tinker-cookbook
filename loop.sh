models=(
    openai/gpt-oss-120b
    # moonshotai/Kimi-K2.5
    # moonshotai/Kimi-K2-Thinking
    # deepseek-ai/DeepSeek-V3.1
    # Qwen/Qwen3-30B-A3B-Instruct-2507
    # Qwen/Qwen3-235B-A22B-Instruct-2507
)

for model in "${models[@]}"; do
    N=5 SPLITS=browsecomp_plus BASE_MODEL=$model MAX_NUM_CALLS=25 bash eval.sh
done
