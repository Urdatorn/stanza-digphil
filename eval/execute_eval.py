try:
    from .mod_conll18_ud_eval import load_conllu, evaluate
except ImportError:
    # Fallback for running this file directly: python eval/execute_eval.py
    from mod_conll18_ud_eval import load_conllu, evaluate
from collections import defaultdict
from pathlib import Path
import argparse
import os
import tempfile
import numpy as np
from pyconll.load import load_from_file

ROOT = Path.cwd()
DEFAULT_GOLD_DIR = Path("/home/urdatorn/git/gh/Diachronic_Treebanks_DigPhil/parsed_data/validated")
DEFAULT_VAL_FILE = ROOT / "eval/gold_as_val.conllu"

def load_val_sentence_ids(val_file_path):
    """Load sentence IDs from the validation file to exclude from evaluation."""
    val_ids = set()
    if not os.path.exists(val_file_path):
        print(f"Warning: validation file {val_file_path} not found. No sentences will be excluded.")
        return val_ids
    val_data = load_from_file(str(val_file_path))
    for sentence in val_data:
        val_ids.add(sentence.id)
    print(f"Loaded {len(val_ids)} sentence IDs to exclude from evaluation (used in validation).")
    return val_ids

def filter_gold_file(gold_file_path, excluded_ids, temp_dir=None):
    """Create a filtered gold file excluding sentences used during validation.
    Returns path to filtered file and the number of excluded sentences."""
    gold_data = load_from_file(gold_file_path)
    filtered_sentences = [s for s in gold_data if s.id not in excluded_ids]
    n_excluded = len(gold_data) - len(filtered_sentences)

    if n_excluded > 0:
        print(f"  Excluded {n_excluded} validation sentences from {os.path.basename(gold_file_path)}")

    # Write filtered data to a temp file
    if temp_dir is None:
        filtered_path = gold_file_path.replace(".conllu", "_eval_filtered.conllu")
    else:
        gold_basename = Path(gold_file_path).name.replace(".conllu", "_eval_filtered.conllu")
        filtered_path = str(Path(temp_dir) / gold_basename)
    with open(filtered_path, "w") as f:
        for sentence in filtered_sentences:
            f.write(sentence.conll())
            f.write("\n\n")
    return filtered_path, len(filtered_sentences)

DEFAULT_MODEL_CONFIGS = [
    (ROOT / "parsed/testset_talbanken_not_seen_gold", ROOT / "eval/scores/scores_talbanken.txt"),
    (ROOT / "parsed/testset_transformer_silver_not_seen_gold", ROOT / "eval/scores/scores_transformer_silver.txt"),
    (ROOT / "parsed/testset_transformer_not_seen_gold_no_silver", ROOT / "eval/scores/scores_transformer_no_silver.txt"),
]

def check_valid_gold(gold_file_path):
    """ Check that the gold trees are valid """
    gold_data = load_from_file(gold_file_path)
    errors = []
    for sentence in gold_data:
        for token in sentence:
            if token.head is None:
                e = ValueError(f"Invalid gold tree in file {gold_file_path}, sentence ID {sentence.id}: token {token.id} '{token.form}' has no head.")
                errors.append(e)
            elif token.head == token.id:
                e = ValueError(f"Invalid gold tree in file {gold_file_path}, sentence ID {sentence.id}: token {token.id} '{token.form}' has itself as head.")
                errors.append(e)
        roots = [token for token in sentence if token.head == '0']
        if len(roots) != 1:
            e = ValueError(f"Invalid gold tree in file {gold_file_path}, sentence ID {sentence.id}: expected 1 root, found {len(roots)}.")
            errors.append(e)
    return errors if len(errors) > 0 else None

def preprocess_system_file(system_file_path, gold_file_path):
    """ Make a new base file with only the trees present in the gold file """
    gold_conllu = load_from_file(gold_file_path)
    system_conllu = load_from_file(system_file_path)
    gold_ids = {sentence.id for sentence in gold_conllu}
    filtered_sentences = [sentence for sentence in system_conllu if sentence.id in gold_ids]
    preprocessed_file_path = system_file_path.replace(".conllu", "_preprocessed.conllu")
    with open(preprocessed_file_path, "w") as outfile:
        for sentence in filtered_sentences:
            outfile.write(sentence.conll())
            outfile.write("\n\n")
    return preprocessed_file_path

def execute_evaluation(gold_file_path, predicted_file_path):
    # Read goldfile
    try:
        gold_data = load_conllu(open(gold_file_path, "r", encoding="utf-8"))
    except Exception as e:
        print(f"Skipping text because of error reading gold file {gold_file_path}: {e}")
        #raise e
        return None
    # Read predicted file and ignore some format errors
    try:
        predicted_data = load_conllu(open(predicted_file_path, "r", encoding="utf-8"), ignore_invalid_format=True)
    except Exception as e:
        print(f"Skipping text because of error reading predicted file {predicted_file_path}: {e}")
        #raise e
        return None
    # Evaluate the predictions against the gold standard
    try:
        raw_results = evaluate(gold_data, predicted_data)
    except Exception as e:
        print(f"Error during evaluation between {gold_file_path} and {predicted_file_path}: {e}")
        raise e
    if isinstance(raw_results, tuple):
        return raw_results[0]
    return raw_results

time_periods = ["1700-1750", "1750-1800", "1800-1850", "1850-1900", "1900-1950"]
metrics = ["UPOS", "UAS", "LAS"] # the relevant metrics

# To look up time period index for a sample
sample_period = {"svediakorp-letter141673-Stalhammar": 0, "svediakorp-sec25-Runius": 0, "svediakorp-sec330-GyllenborgC_SwenskaSpratthoken": 0,
           "svediakorp-sec277-EnbomPU_MedborgeligtSkalde": 1, "svediakorp-sec268-DulciU_VitterhetsNojen3": 1,
           "svediakorp-sec991-spf148": 2, "svediakorp-sec252-BremerF_Teckningar1": 2, "svediakorp-sec324-GranbergPA_Enslighetsalskaren": 2,
           "svediakorp-sec254-CederborghF_BerattelseOmJohnHall": 2, "svediakorp-sec987-spf144": 2, "svediakorp-sec988-spf145": 2,
           "svediakorp-sec452-NyblomH_FantasierFyra": 3, "svediakorp-sec486-SchwartzMS_BellmansSkor": 3, "svediakorp-sec1102-spf259": 3,
           "svediakorp-sec208-Anonym_DetGrasligaMordet": 3, "svediakorp-sec1063-spf220": 3,
           "svediakorp-sec631-HasselskogN_HallaHallaGronkoping": 4, "svediakorp-sec1033-spf190": 4, "svediakorp-sec397-AngeredStrandbergH_UnderSodernsSol": 4,
           "svediakorp-sec613-EngstromA_StrindbergOchJag": 4, "svediakorp-sec639-HeidenstamV_Proletarfilosofiens": 4}


def run_evaluation(predicted_dir, output_file, gold_dir=DEFAULT_GOLD_DIR, val_file=DEFAULT_VAL_FILE):
    """Run evaluation for a single model configuration."""
    predicted_dir = Path(predicted_dir)
    output_file = Path(output_file)
    gold_dir = Path(gold_dir)
    val_file = Path(val_file)

    val_sentence_ids = load_val_sentence_ids(str(val_file))

    print(f"\n{'='*70}")
    print(f"Evaluating: {predicted_dir}")
    print(f"Output: {output_file}")
    print(f"{'='*70}\n")

    # Check if predicted directory exists
    if not predicted_dir.exists():
        print(f"ERROR: Predicted directory does not exist: {predicted_dir}")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Reset per-model state
    score_arrays = [np.zeros((4, len(metrics))), np.zeros((4, len(metrics))), np.zeros((4, len(metrics))), np.zeros((4, len(metrics))), np.zeros((4, len(metrics)))]
    skipped_samples = []
    sample_nsents = defaultdict(int)
    processed_pairs = []  # Track (gold_conllu_text, pred_conllu_text) for overall evaluation

    with open(output_file, "w", encoding="utf-8") as outfile:
        # Loop through each of the validated samples
        for sample in os.listdir(gold_dir):
            sample_name = sample.split("/")[-1].replace(".conllu", "")

            print("DEBUG sample_name:", sample_name)

            period = sample_period.get(sample_name) # int 0-4
            print("DEBUG period value:", period)

            if period is None:
                print(f"Skipping {sample_name}: missing time-period mapping")
                skipped_samples.append(sample_name)
                continue

            with tempfile.TemporaryDirectory(prefix="eval_tmp_") as tmp_dir:
                # Filter gold file to exclude sentences used during validation
                gold_file_path_orig = str(gold_dir / sample)
                gold_file_path, n_filtered_sents = filter_gold_file(gold_file_path_orig, val_sentence_ids, temp_dir=tmp_dir)

                # Skip if all sentences were excluded
                if n_filtered_sents == 0:
                    print(f"Skipping {sample_name}: all sentences were used in validation")
                    skipped_samples.append(sample_name)
                    continue

                # Check that gold trees are valid (using filtered file)
                e = check_valid_gold(gold_file_path)
                if e: # skip invalid gold files
                    for error in e:
                        print(error)
                    skipped_samples.append(sample_name)
                    continue
                # Always regenerate preprocessed predicted file to match filtered gold
                unprocessed_predicted_file_path = str(predicted_dir / sample)
                if os.path.exists(unprocessed_predicted_file_path):
                    # Use filtered gold file to preprocess predictions
                    predicted_file_path = preprocess_system_file(unprocessed_predicted_file_path, gold_file_path)
                else:
                    print(f"Skipping {sample_name}: missing predicted file {unprocessed_predicted_file_path}")
                    skipped_samples.append(sample_name)
                    continue
                # Score base tree (parser output) against validated tree (gold standard)
                results = execute_evaluation(gold_file_path, predicted_file_path)

                if not results: # skip invalid files
                    skipped_samples.append(sample_name)
                # Write results to scores file and save to time period arrays
                else:
                    # Save number of sentences in lookup dict (using filtered count)
                    sample_nsents[sample_name] = n_filtered_sents
                    # Keep content in memory so temporary files can be removed immediately.
                    with open(gold_file_path, "r", encoding="utf-8") as gf:
                        gold_text = gf.read()
                    with open(predicted_file_path, "r", encoding="utf-8") as pf:
                        pred_text = pf.read()
                    processed_pairs.append((gold_text, pred_text))
                    print("DEBUG period value:", period)
                    print("DEBUG time_periods:", time_periods)

                    outfile.write(f"Sample: {sample_name}, time period: {time_periods[period]}\n")
                    outfile.write("Metric\tPrecision\tRecall\tF1\tAligned Accuracy\n")
                    metric_n = 0 # counter for loop below
                    # Loop over each metric: UPOS, UAS, LAS etc.
                    for metric, score in results.items(): # get the metric and its 4 scores
                        if metric in metrics: # only save the relevant metrics
                            precision = getattr(score, "precision", None)
                            recall = getattr(score, "recall", None)
                            f1 = getattr(score, "f1", None)
                            aligned_accuracy = getattr(score, "aligned_accuracy", None)
                            if precision is None or recall is None or f1 is None:
                                continue
                            # Save scores weighted by the number of sentences to arrays
                            score_arrays[period][0][metric_n] += (precision * sample_nsents[sample_name])
                            score_arrays[period][1][metric_n] += (recall * sample_nsents[sample_name])
                            score_arrays[period][2][metric_n] += (f1 * sample_nsents[sample_name])
                            # Write the metric and its file-average scores
                            outfile.write(f"{metric}\t{precision:.4f}\t{recall:.4f}\t{f1:.4f}\t")
                            if aligned_accuracy is None:
                                outfile.write(f"{aligned_accuracy}\n")
                            else:
                                outfile.write(f"{aligned_accuracy:.4f}\n")
                                # Save aligned accuracy to score arrays
                                score_arrays[period][3][metric_n] += (aligned_accuracy * sample_nsents[sample_name])
                            metric_n += 1 # move to next metric in array
                    outfile.write("\n")
        outfile.write(f"{'-'*70}\n")

        # After all sample scores are written, compute average scores for each time period
        for period, period_scores in enumerate(score_arrays):
            # Get number of samples and sentences for the period
            samples_per_period = [sample_name for sample_name in sample_period.keys() if sample_period[sample_name] == period and sample_name not in skipped_samples]
            n_sents_per_period = sum([sample_nsents[sample_name] for sample_name in samples_per_period])
            # Write
            outfile.write(f"Average scores for time period {time_periods[period]} ({len(samples_per_period)} samples with a total of {n_sents_per_period} sentences):\n")
            outfile.write("Metric\tPrecision\tRecall\tF1\tAligned Accuracy\n")
            for metric_n, metric in enumerate(metrics):
                # Normalize weighted averages by the total number of sentences for the time period
                if n_sents_per_period == 0:
                    precision_avg = 0.0
                    recall_avg = 0.0
                    f1_avg = 0.0
                    aligned_acc_avg = 0.0
                else:
                    precision_avg = period_scores[0][metric_n] / n_sents_per_period
                    recall_avg = period_scores[1][metric_n] / n_sents_per_period
                    f1_avg = period_scores[2][metric_n] / n_sents_per_period
                    aligned_acc_avg = period_scores[3][metric_n] / n_sents_per_period
                outfile.write(f"{metric}\t{precision_avg:.4f}\t{recall_avg:.4f}\t{f1_avg:.4f}\t{aligned_acc_avg:.4f}\n")
            outfile.write("\n")

        # Compute overall scores by concatenating all files and running single evaluation
        valid_samples = [sample_name for sample_name in sample_period.keys() if sample_name in sample_nsents and sample_name not in skipped_samples]
        total_sents = sum(sample_nsents[sample_name] for sample_name in valid_samples)

        if len(processed_pairs) == 0:
            outfile.write(f"Overall scores across all time periods ({len(valid_samples)} samples with a total of {total_sents} sentences):\n")
            outfile.write("Metric\tPrecision\tRecall\tF1\tAligned Accuracy\n")
            outfile.write("No valid samples available for overall evaluation.\n")
        else:
            # Concatenate all gold and predicted files
            all_gold_path = str(predicted_dir / "_all_gold_combined.conllu")
            all_pred_path = str(predicted_dir / "_all_pred_combined.conllu")

            with open(all_gold_path, "w") as gold_out, open(all_pred_path, "w") as pred_out:
                for gold_text, pred_text in processed_pairs:
                    gold_out.write(gold_text)
                    pred_out.write(pred_text)

            # Run single evaluation on combined files
            overall_results = execute_evaluation(all_gold_path, all_pred_path)

            outfile.write(f"Overall scores across all time periods ({len(valid_samples)} samples with a total of {total_sents} sentences):\n")
            outfile.write("Metric\tPrecision\tRecall\tF1\tAligned Accuracy\n")

            if overall_results:
                for metric, score in overall_results.items():
                    if metric in metrics:
                        precision = getattr(score, "precision", None)
                        recall = getattr(score, "recall", None)
                        f1 = getattr(score, "f1", None)
                        aligned_accuracy = getattr(score, "aligned_accuracy", None)
                        if precision is None or recall is None or f1 is None:
                            continue
                        outfile.write(f"{metric}\t{precision:.4f}\t{recall:.4f}\t{f1:.4f}\t")
                        if aligned_accuracy is None:
                            outfile.write(f"{aligned_accuracy}\n")
                        else:
                            outfile.write(f"{aligned_accuracy:.4f}\n")
            else:
                outfile.write("Error computing overall scores.\n")

    print(f"Results saved to {output_file}")


def run_all_evaluations(model_configs=DEFAULT_MODEL_CONFIGS, gold_dir=DEFAULT_GOLD_DIR, val_file=DEFAULT_VAL_FILE):
    for predicted_dir, output_file in model_configs:
        run_evaluation(predicted_dir=predicted_dir, output_file=output_file, gold_dir=gold_dir, val_file=val_file)
    print("\nAll evaluations complete!")


def main():
    parser = argparse.ArgumentParser(description="Run UD evaluation on parser output.")
    parser.add_argument("--predicted-dir", type=Path, help="Directory containing predicted .conllu files.")
    parser.add_argument("--output-file", type=Path, help="Path to write scores output.")
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR, help="Directory containing gold .conllu files.")
    parser.add_argument("--val-file", type=Path, default=DEFAULT_VAL_FILE, help="Validation file with sentence IDs to exclude.")
    args = parser.parse_args()

    if (args.predicted_dir is None) != (args.output_file is None):
        parser.error("--predicted-dir and --output-file must be provided together.")

    if args.predicted_dir and args.output_file:
        run_evaluation(
            predicted_dir=args.predicted_dir,
            output_file=args.output_file,
            gold_dir=args.gold_dir,
            val_file=args.val_file,
        )
    else:
        run_all_evaluations(gold_dir=args.gold_dir, val_file=args.val_file)


if __name__ == "__main__":
    main()
