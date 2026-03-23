import os
from pathlib import Path
import stanza
from stanza.utils.conll import CoNLL
import time
import gc
import torch
from tqdm import tqdm

if torch.xpu.is_available():
  device = torch.device("xpu")
elif torch.cuda.is_available():
  device = torch.device("cuda")

###################################################################################################
##### SETTINGS ####################################################################################
###################################################################################################

vanilla = False

conllu_in_dir = Path("/home/urdatorn/git/Diachronic_Treebanks_DigPhil/parsed_data/stanza_results_normalized")
conllu_out_dir = Path("/home/urdatorn/git/Diachronic_Treebanks_DigPhil/parsed_data/stanza_results_normalized_transformer_seen_gold_no_silver")

depparse_model_path = "saved_models/depparse/transformer_seen_gold_no_silver.pt"

os.makedirs(conllu_out_dir, exist_ok=True)

##### PREPARE #####################################################################################

if vanilla:
    print("Using vanilla Swedish Stanza pipeline with default models.")
    nlp = stanza.Pipeline(
        "sv",
        processors="tokenize,pos,lemma,depparse",
        tokenize_pretokenized=True, # to keep original tokens 
        use_gpu=True,
        pos_batch_size=3000,
        package=None,
        device=device,
    )

else:
    nlp = stanza.Pipeline(
        "sv",
        processors="tokenize,pos,lemma,depparse",
        tokenize_pretokenized=True, # to keep original tokens 
        use_gpu=True,
        pos_batch_size=3000,
        package=None,
        download_method=None,
        depparse_model_path=depparse_model_path, # depparse_model_path becomes model_path inside the depparse processor config. explicit model_path is to control where normal Stanza models are stored
        device=device,
    )

##### INFERENCE ###################################################################################

t0 = time.time()
total_sentences = 0

for fname in tqdm(sorted(os.listdir(conllu_in_dir)), desc="Files"):
    if not fname.endswith(".conllu"):
        continue

    in_path = conllu_in_dir / fname
    out_path = conllu_out_dir / fname
    
    if out_path.exists():
        print(f"File already parsed, skipping: {out_path}")
        continue

    doc = CoNLL.conll2doc(in_path)

    total_sentences += len(doc.sentences)

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

t = time.time() - t0

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