# Batched LM Generation

This is an opinionated, but convenient set of tools for batch generation of
completions from an LLM. It supports resumption, so that you can recover from
CUDA OOM errors or Slurm timeouts. It saves completions compactly, as *.json.gz* 
files with counters to manage duplicates. You can either use this data directly,
or export it to other formats, such as the BigCode Evaluation Harness format.

Finally, the core features do not depend on any particular model framework
(e.g., Transformers, VLLM, or a web API). Instead, framework-dependent code is isolated into separate modules and get the core features with minimal effort.

Currently supported frameworks and modes:

1. Use batched_lm_generation.automodel_base to generate completions from any
   base model using Transformers.

2. Use batched_lm_generation.automodel_chatcoder to generate code completions
   from a chat or instruction-tuned model using Transformers.

## Examples

### HumanEval with Llama 3 Instruct

We will use these tools to generate HumanEval completions, but evaluate with
the BigCode Evaluation Harness.

1. Let's use Llama 3 to generate 20 completions per prompt for HumanEval:

    ```bash
    python3 -m batched_lm_generation.automodel_chatcoder \
        --model-name meta-llama/Meta-Llama-3-8B-Instruct \
        --dataset openai_humaneval \
        --dataset-split test \
        --output-dir humaneval_llama3_8b_instruct \
        --temperature 0.2 \
        --batch-size 100 \
        --completion-limit 20
    ```

2. When complete, we can convert this to the BigCode Evaluation Harness format:

   ```bash
   python3 -m batched_lm_generation.bigcode_format \
       humaneval_llama3_8b_instruct \
       generations_humaneval.json
   ```

   If you're getting impatient, you can run the code above while Step 1 is still running. You'll get a partial list of completions.

3. Finally, you can evaluate the completions with the BigCode Evaluation Harness:

   ```bash
   python3 bigcode-evaluation-harness/main.py \
     --tasks humaneval \
     --load_generations_path generations_humaneval.json \
     --n_samples 20 \
     --allow_code_execution 
   ```