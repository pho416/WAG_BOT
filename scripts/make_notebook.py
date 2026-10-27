"""Create a portable notebook using the exact same CLI as the Linux GPU runner."""
import json
from pathlib import Path

cells = []


def markdown(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.splitlines(keepends=True)})


markdown("""# WattBot: GPU baseline
Run this from an extracted `wattbot-starter.zip` / `wattbot-server-full.zip`, or attach the starter to Kaggle.
Uses Qwen2.5-3B-Instruct with 4-bit FP16 compute. The initial default is one GPU, five evidence chunks.
Generation has not been measured on the user's GPU yet. CPU retrieval results are separate from answer scores.
For the supplied Linux server with driver 535.161.07, prefer `scripts/setup_gpu.sh` and `scripts/run_server.sh`.
""")
code("""from pathlib import Path
import os, sys, subprocess, shutil, zipfile

# Edit for your environment. On Colab, upload the starter and set PROJECT_ZIP.
PROJECT_ZIP = None
GPU_ID = '2' if not Path('/kaggle').exists() and not Path('/content').exists() else '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

if Path.cwd().name == 'notebooks' and (Path('..') / 'wattbot').exists():
    os.chdir('..')
if not Path('wattbot').exists():
    target = Path('/kaggle/working/wattbot') if Path('/kaggle').exists() else Path('/content/wattbot')
    candidates = list(Path('/kaggle/input').rglob('wattbot/__main__.py')) if Path('/kaggle/input').exists() else []
    if candidates:
        shutil.copytree(candidates[0].parents[1], target, dirs_exist_ok=True)
    else:
        archives = [Path(PROJECT_ZIP)] if PROJECT_ZIP else (list(Path('/kaggle/input').rglob('wattbot-starter.zip')) if Path('/kaggle/input').exists() else [])
        if not archives:
            raise RuntimeError('Extract the starter project or set PROJECT_ZIP to its uploaded path.')
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archives[0]) as archive:
            for member in archive.infolist():
                resolved = (target / member.filename).resolve()
                if not resolved.is_relative_to(target.resolve()):
                    raise ValueError('Unsafe archive path')
            archive.extractall(target)
    os.chdir(target)

def run(*args):
    subprocess.run([sys.executable, '-m', 'wattbot', *args], check=True)
print('Project:', Path.cwd(), 'visible GPU:', GPU_ID)
""")
markdown("""Install into the notebook runtime. Kaggle/Colab must have Internet enabled for model downloads. Keep their existing compatible CUDA PyTorch; the server setup script instead installs a pinned CUDA 12.1 build in a fresh venv.""")
code("""subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-gpu.txt'], check=True)
run('doctor')
import torch
assert torch.cuda.is_available(), 'Select a GPU runtime before generation'
print([(torch.cuda.get_device_name(i), torch.cuda.mem_get_info(i)) for i in range(torch.cuda.device_count())])
subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=open('requirements-runtime.lock.txt', 'w'), check=True)
""")
code("""run('profile')
run('split')
if not Path('data/processed/chunks.jsonl').exists():
    run('download')
    run('ingest')
run('retrieve', '--context-k', '5')
run('retrieval-eval')
""")
markdown("""Start with five questions. Once successful, set `LIMIT=None` and use a new output directory to run the full train set. Predictions and per-question traces survive an interrupted run. A generation/JSON failure stops the run and preserves successful rows; it is never scored as an intentional abstention.""")
code("""MODEL = 'Qwen/Qwen2.5-3B-Instruct'
LIMIT = 5
OUTPUT = 'runs/notebook_smoke'
args = ['generate', '--model', MODEL, '--output-dir', OUTPUT]
if LIMIT is not None:
    args += ['--limit', str(LIMIT)]
run(*args)
run('validate', '--questions', f'{OUTPUT}/questions.csv', '--submission', f'{OUTPUT}/predictions.csv')
""")
code("""# Evaluate exactly the selected IDs, without putting gold fields in inference.
import pandas as pd
chosen = pd.read_csv(f'{OUTPUT}/questions.csv', dtype=str)['id']
gold = pd.read_csv('data/raw/train_QA.csv', keep_default_na=False, dtype=str)
gold[gold['id'].isin(chosen)].to_csv(f'{OUTPUT}/gold_for_evaluation.csv', index=False)
run('evaluate', '--gold', f'{OUTPUT}/gold_for_evaluation.csv', '--submission', f'{OUTPUT}/predictions.csv', '--output-dir', f'{OUTPUT}/evaluation')
""")
markdown("""Next ablation: dense retrieval. Embedding runs on CPU to leave the GPU free. Compare citation recall before generating another full answer run. Reranking is optional via `retrieve --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 --device cpu`. Model revisions should be pinned before the submission run.""")
code("""# Uncomment to run this ablation.
# run('embed', '--model', 'BAAI/bge-small-en-v1.5', '--device', 'cpu')
# run('retrieve', '--dense-dir', 'artifacts/dense', '--context-k', '5', '--output', 'runs/hybrid/train_retrieval.jsonl')
# run('retrieval-eval', '--retrieval', 'runs/hybrid/train_retrieval.jsonl', '--output', 'runs/hybrid/retrieval_metrics.json')
""")
markdown("""Generate test predictions only after comparing validation results. Use `scripts/run_server.sh test` or the CLI commands in README. The current answer path reads PDF text; visual retrieval, VLM table/figure reading, and iterative multi-hop retrieval are subsequent experiments.""")

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
dest = Path(__file__).resolve().parents[1] / "notebooks/WattBot_GPU.ipynb"
dest.parent.mkdir(exist_ok=True)
dest.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(dest)
