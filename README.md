# Writing choice

Companion code for *What Contextual LLM Likelihood Adds to Human Writing Choice*, manuscript version 11 October 2026 (eight-page main text). Run the commands from this directory.

Replay HAI and CoAuthor menus, score exact candidate continuations under three contexts, and fit conditional choice models separately by corpus and condition.

## Environment

Run the commands below in Bash. Use Python 3.11 and a CUDA GPU with enough memory for FP32 inference; the 8B runs used GPUs reporting 49,140 MiB. Keep the batch settings below when comparing the reported results.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.4 pandas==2.3.3 scipy==1.16.3 \
  transformers==4.57.2 tokenizers==0.22.1 accelerate==1.14.0 \
  safetensors==0.7.0 sentencepiece==0.2.1 huggingface-hub==0.36.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p data models work
```

Use new output directories for a run. Inference writes score metadata beside the score file; analyses use those files and intermediate outputs from the earlier commands. Raw texts, labels, model weights and generated results are not included. Obtain the sources below under their own terms and retain their notices.

```bash
python -m pip install scikit-learn==1.7.2
```

## Source logs and candidate features

```bash
git clone https://github.com/vishakhpk/hai-diversity.git data/hai
git -C data/hai checkout 027ef445da6999ea3ad0054101ecae117aeb6b69
git clone https://github.com/celinatreuillier/Authorship_Calibration_CoAuthor.git data/coauthor
git -C data/coauthor checkout 8ba055111af24cb4c5fd1eabe759f80948da5fb9
python replay_hai.py --data data/hai --output work/hai_replay
python hai_features.py --replay work/hai_replay --output work/hai_features
python replay_coauthor.py --data data/coauthor --output work/coauthor_replay
python coauthor_features.py --replay work/coauthor_replay --output work/coauthor_features
python prepare_menus.py --output work/menus_lexical
python prepare_native_prefixes.py --output work/menus
python prepare_scoring.py --menus work/menus --output work/inputs
```

HAI requires `processed_data/essays_gpt3.jsonl`, `processed_data/essays_instructgpt.jsonl` and every raw log they reference. CoAuthor requires `data/CoAuthor/add_data/metadata_argumentative.csv`, `metadata_creative.csv` and the session logs under `data/CoAuthor/coauthor-v1.0/`. If the checkout uses Git LFS or omits logs, obtain the actual source files before running; pointer files and partial inventories do not reproduce the analysis. CoAuthor is a research copy of the original corpus, not an independent participant cohort.

Replay validates event endpoints and selected insertions. Lexical reference features exclude the target session for HAI and the target writer for CoAuthor. Native prefixes preserve source whitespace; the model input adds the same separator to every candidate. The expected scoring input has 128,745 rows, equally split across correct, empty and mismatched prefixes.

## Continuation scoring and comparison

```bash
base_revision=49e3418fbbbca6ecbdf9608b4d22e5a407081db4
instruct_revision=b968826d9c46dd6066d109eabc6255188de91218
hf download Qwen/Qwen3-8B-Base --revision "$base_revision" --local-dir "models/$base_revision"
hf download Qwen/Qwen3-8B --revision "$instruct_revision" --local-dir "models/$instruct_revision"
for checkpoint in base instruct; do
  if [ "$checkpoint" = base ]; then
    model="$base_revision"; model_name=Qwen3-8B-Base
  else
    model="$instruct_revision"; model_name=Qwen3-8B
  fi
  python score_continuations.py --model "models/$model" --input work/inputs/inputs.jsonl \
    --output "work/scores/$checkpoint.jsonl" --batch-size 1 --max-tokens 1536 \
    --dtype float32 --sentinel-token '<|endoftext|>' --serialize-empty-prefix \
    --check-batch-equivalence --check-full-forward
  python analyze_choices.py --scores "work/scores/$checkpoint.jsonl" --model-name "$model_name" \
    --bootstrap 2000 --worker-refits 100 --output "work/analysis_$checkpoint"
done
python compare_checkpoints.py --base work/analysis_base --instruct work/analysis_instruct \
  --output work/comparison_checkpoints
```

Use batch size 1 for the continuation runs. The shared sentinel is token 151643. Scores are teacher-forced token likelihoods with no generated answer or scored EOS. This compact entry point processes the full input directly; the original computation used contiguous singleton shards. Statistical outputs retain mean and total likelihood views, the four primary comparisons and the two checkpoint contrasts. Fixed-supply cluster draws and paired worker refits are stored separately.
