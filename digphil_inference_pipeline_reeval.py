import os
from pathlib import Path
import stanza
from stanza.utils.conll import CoNLL
import time
import gc
import re
import torch
from tqdm import tqdm
from eval.execute_eval import run_evaluation

if torch.xpu.is_available():
  device = torch.device("xpu")
elif torch.cuda.is_available():
  device = torch.device("cuda")
else:
    device = torch.device("cpu")

###################################################################################################
##### SETTINGS ####################################################################################
###################################################################################################

conllu_in_dir = Path("/home/urdatorn/git/gh/Diachronic_Treebanks_DigPhil/parsed_data/validated")
conllu_out_root_dir = Path("/home/urdatorn/git/gh/Diachronic_Treebanks_DigPhil/parsed_data/new_eval_of_old_models")
depparse_models_dir = Path("saved_models/depparse/old")
val_file = Path("eval/gold_as_val.conllu")
scores_out_dir = Path("eval/scores/language_mix")
summary_out_file = scores_out_dir / "language_mix_reeval_score_list.py"
use_gpu = device.type in {"cuda", "xpu"}

os.makedirs(conllu_out_root_dir, exist_ok=True)
os.makedirs(scores_out_dir, exist_ok=True)

top_level_models = [p for p in depparse_models_dir.glob("*.pt") if p.is_file()]
one_level_nested_models = [
    p for p in depparse_models_dir.glob("*/*.pt")
    if p.is_file() and not p.parent.name.startswith("kubhist")
]
model_entries = sorted(set(top_level_models + one_level_nested_models))
if not model_entries:
    raise FileNotFoundError(f"No model files found in {depparse_models_dir}")

def get_model_name(model_entry):
    if model_entry.parent == depparse_models_dir:
        return model_entry.stem
    return model_entry.parent.name


run_specs = [("vanilla", None)] + [(get_model_name(model_entry), model_entry) for model_entry in model_entries]


def extract_overall_las_f1(score_file_path):
    """Extract LAS F1 from the final overall scores section of a score file."""
    if not score_file_path.exists():
        return None

    with open(score_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Find all Overall sections and use the last one.
    overall_blocks = re.findall(
        r"Overall scores across all time periods.*?(?=\n\n|\Z)",
        text,
        flags=re.DOTALL,
    )
    if not overall_blocks:
        return None

    last_overall = overall_blocks[-1]
    for line in last_overall.splitlines():
        if line.startswith("LAS\t"):
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                try:
                    return float(parts[3])
                except ValueError:
                    return None
    return None

##### INFERENCE ###################################################################################

t0 = time.time()
total_sentences = 0
model_las_f1_scores = {}

for model_name, model_entry in tqdm(run_specs, desc="Models"):

    conllu_out_dir = conllu_out_root_dir / model_name
    os.makedirs(conllu_out_dir, exist_ok=True)

    if model_entry is None:
        print("Running model: vanilla")
        nlp = stanza.Pipeline(
            "sv",
            processors="tokenize,pos,lemma,depparse",
            tokenize_pretokenized=True, # to keep original tokens
            use_gpu=use_gpu,
            pos_batch_size=3000,
            device=device,
        )
    else:
        print(f"Running model: {model_entry}")
        nlp = stanza.Pipeline(
            "sv",
            processors="tokenize,pos,lemma,depparse",
            tokenize_pretokenized=True, # to keep original tokens
            use_gpu=use_gpu,
            pos_batch_size=3000,
            download_method=stanza.DownloadMethod.NONE,
            depparse_model_path=str(model_entry), # depparse_model_path becomes model_path inside the depparse processor config. explicit model_path is to control where normal Stanza models are stored
            device=device,
        )

    model_sentences = 0

    for fname in tqdm(sorted(os.listdir(conllu_in_dir)), desc=f"Files ({model_name})", leave=False):
        if not fname.endswith(".conllu"):
            continue

        in_path = conllu_in_dir / fname
        out_path = conllu_out_dir / fname

        if out_path.exists():
            print(f"File already parsed, skipping: {out_path}")
            continue

        doc = CoNLL.conll2doc(in_path)

        sentence_count = len(doc.sentences)
        model_sentences += sentence_count
        total_sentences += sentence_count

        with torch.inference_mode():
            parsed_doc = nlp(doc)

        with open(out_path, "w", encoding="utf-8") as f:
            CoNLL.write_doc2conll(parsed_doc, f)

        # Release per-file objects and ask both Python and the torch allocator to
        # reclaim memory that is no longer needed.
        del doc
        del parsed_doc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            memory_max = torch.cuda.max_memory_allocated()
            print(f"Max memory allocated so far: {memory_max / (1024 ** 3):.2f} GB")
        elif device.type == "xpu":
            torch.xpu.empty_cache()
            memory_max = torch.xpu.max_memory_allocated()
            print(f"Max memory allocated so far: {memory_max / (1024 ** 3):.2f} GB")

    print(f"Completed model {model_name}. Parsed sentences this run: {model_sentences}")

    score_out_path = scores_out_dir / f"scores_{model_name}.txt"
    run_evaluation(
        predicted_dir=conllu_out_dir,
        output_file=score_out_path,
        gold_dir=conllu_in_dir,
        val_file=val_file,
    )
    model_las_f1_scores[model_name] = extract_overall_las_f1(score_out_path)

    del nlp
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()

t = time.time() - t0

with open(summary_out_file, "w", encoding="utf-8") as f:
    f.write("# Auto-generated by digphil_inference_pipeline_reeval.py\n")
    f.write("language_mix_reeval_score_list = {\n")
    for model_name in sorted(model_las_f1_scores.keys()):
        score = model_las_f1_scores[model_name]
        if score is None:
            f.write(f"    {model_name!r}: None,\n")
        else:
            f.write(f"    {model_name!r}: {score:.4f},\n")
    f.write("}\n")

###################################################################################################

if device.type == "cuda":
    memory_max = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
elif device.type == "xpu":
    memory_max = torch.xpu.max_memory_allocated()
    torch.xpu.reset_peak_memory_stats()
else:
    memory_max = 0

print(
    "\nFinished parsing.\n"
    f"Total sentences parsed: {total_sentences}\n"
    f"Total time: {t / 60:.2f} minutes\n"
    f"Max memory allocated: {memory_max / (1024 ** 3):.2f} GB\n"
)