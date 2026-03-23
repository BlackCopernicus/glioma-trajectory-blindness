# BvB / Regime Collapse Kaggle Reproduction

This folder contains a cleaned GitHub-ready reconstruction of the BvB regime-collapse runners from the recovered Kaggle notebook, Colab sweep notebook, and recovered CSV artifacts.

## Files

- `bvb_kaggle_runner.py` - main one-file reproduction runner for GPT-2 small and GPT-2 medium.
- `bvb_llama2_runner.py` - separate PEFT/QLoRA runner for Llama 2 robustness sweeps on Kaggle.
- `requirements_bvb.txt` - Python dependencies.

## Expected input files

Place these text files in the Kaggle working directory or pass `--input-root`:

- `regime_A.txt`
- `regime_B S.txt`
- `math_regime_A_CME.tex`
- `math_regime_B_PCM.tex`

The main runner recreates the recovered experiment spine:

- GPT-2 small language baseline
- GPT-2 small math baseline
- GPT-2 small alignment/fix run
- GPT-2 small reverse-direction run
- GPT-2 small lambda sweep
- GPT-2 medium lambda sweep
- GPT-2 medium calibration runs at lambda = 0.0025
- predictability diagnostics
- summary CSVs and figures

## Kaggle quick start

```bash
pip install -r requirements_bvb.txt
python bvb_kaggle_runner.py --input-root /kaggle/working --output-dir /kaggle/working/bvb_out --mode all
```

To run only the GPT-2 small sweep:

```bash
python bvb_kaggle_runner.py --input-root /kaggle/working --output-dir /kaggle/working/bvb_out --mode small-sweep
```

To run only the GPT-2 medium calibration/sweep:

```bash
python bvb_kaggle_runner.py --input-root /kaggle/working --output-dir /kaggle/working/bvb_out --mode medium
```

## Llama 2 robustness sweeps

The Llama 2 script is separate because it uses PEFT/QLoRA rather than full-weight fine-tuning.

```bash
pip install -r requirements_bvb.txt
export HUGGINGFACE_HUB_TOKEN=YOUR_TOKEN
python bvb_llama2_runner.py   --input-root /kaggle/working   --output-dir /kaggle/working/llama2_out   --models meta-llama/Llama-2-7b-hf   --lambdas 0.0 0.01 0.025   --steps 150 300   --seeds 0
```

## Notes

- The GPT-2 small sweep code was reconstructed from the recovered Colab PDF and merged into the Kaggle notebook spine.
- The GPT-2 medium sweep/calibration logic comes from the recovered notebook cells and CSV artifacts.
- The Llama 2 script is for robustness checks, not exact bit-for-bit reproduction of the GPT-2 numbers.
- Both scripts include skip/resume logic through CSV outputs.
