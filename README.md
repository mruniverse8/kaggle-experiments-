# Causal Tweet Extraction (Prompt + Tweet + Sentiment + Selected Text)

This folder contains a training pipeline for causal LMs (Qwen/Llama/Gemma) with:

- `L_ce`: causal language modeling loss (generate selected text)
- `L_span_kl`: soft start/end span loss (KL divergence)
- `L_select`: token selection loss (BCE + Dice)

The sample order is exactly:

1. `prompt`
2. `tweet`
3. `sentiment`
4. `selected_text`

using explicit boundary tokens.

The dataset preprocessing mirrors the original Kaggle span handling:

- normalize whitespace
- prepend one leading space (`" " + text`)
- match selected span with robust char-level search
- build soft start/end labels from Jaccard arrays

## Input Format

```text
<PROMPT> You are a professional emotion identifier </PROMPT>
<TWEET> I love this movie! </TWEET>
<SENTIMENT> positive </SENTIMENT>
<ANSWER> love this </ANSWER>
```

- CE is applied only after `<ANSWER>`.
- KL and BCE/Dice are applied only on tokens between `<TWEET>` and `</TWEET>`.

## Files

- `src/dz2_causal/dataset.py`: dataset creation and soft-label targets
- `src/dz2_causal/modeling.py`: causal LM wrapper + start/end/select heads
- `src/dz2_causal/losses.py`: combined losses
- `src/dz2_causal/train.py`: training loop
- `inspect_example.py`: prints one fully prepared sample for debugging
- `train_qwen_span.py`: CLI entrypoint
- `environment.yml`: conda environment definition
- `requirements.txt`: pip package list
- `setup_conda_env.sh`: automated conda + pip setup

## Environment Setup

GPU setup (recommended):

```bash
cd ./curret_foilder
bash setup_conda_env.sh dz2-qwen gpu
conda activate dz2-qwen
```

CPU setup:

```bash
cd ./curret_foilder
bash setup_conda_env.sh dz2-qwen cpu
conda activate dz2-qwen
```

Manual alternative:

```bash
cd ./curret_foilder
conda env create -n dz2-qwen -f environment.yml
conda activate dz2-qwen
python -m pip install --upgrade pip
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Quickstart

```bash
cd ./curret_foilder

python inspect_example.py \
  --model-name Qwen/Qwen3.5-0.8B \
  --prompt "You are a professional emotion identifier" \
  --tweet "I love this movie!" \
  --sentiment "positive" \
  --selected-text "love this"
```

Train:

```bash
python train_qwen_span.py --config config/train_qwen35_08b.json
```

Edit config values in:

- `config/train_qwen35_08b.json`

## Notes

- You need `transformers`, `torch`, and `pandas`.
- For 24GB GPU, start with mixed precision (`--no-amp` disabled).
- If you run out of memory, reduce `--batch-size` and/or set `--max-len`.
