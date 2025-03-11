from reasoning_methods.prompting import multiple_choice_processor
from reasoning_methods.prompting import numeric_processor
from .config import DATASET_CONFIGS

def process_dataset_batch(pipe, dataset, template_name, args, batch_size):
    """
    Dispatcher function to process dataset batches, routing processing depending on the dataset type 
    and self-consistency flag, ensuring that at least 1000 successful samples are processed.
    If errors occur in processing some samples, additional samples are processed until 1000 valid results are obtained.
    For multiple-choice benchmarks (e.g. RACE, ARC, MMLU) no freeform text is generated.
    For numeric tasks (GSM8K, DROP, MATH), text generation and answer extraction are performed.
    """
    correct_total = 0
    total_processed = 0
    results = []
    dataset_size = len(dataset)
    pointer = 0

    # Define which datasets are numeric vs. multiple-choice.
    multiple_choice_datasets = ["race", "arc", "mmlu", "agieval"]

    # Store batch_size in args for access in processor functions
    args.batch_size = batch_size

    while len(results) < 1000:
        target = 1000 - len(results)  # remaining samples needed
        current_batch_size = min(batch_size, target)
        batch_indices = []
        for _ in range(current_batch_size):
            if args.dataset == "mmlu":
                # For mmlu, cycle through dataset if needed.
                batch_indices.append(pointer % dataset_size)
            else:
                # For other datasets, ensure pointer is within dataset range.
                if pointer < dataset_size:
                    batch_indices.append(pointer)
                else:
                    # In the unlikely event we run out, cycle through the dataset
                    print(f"Pointer {pointer} is out of range for dataset size {dataset_size}")
                    batch_indices.append(pointer % dataset_size)
            pointer += 1

        if args.dataset in multiple_choice_datasets:
            mapping = {"A": "0", "B": "1", "C": "2", "D": "3"}
            if args.self_consistency:
                corr, tot, batch_results = multiple_choice_processor.process_mc_self_consistency(
                    pipe, dataset, template_name, args, batch_indices, mapping
                )
            else:
                corr, tot, batch_results = multiple_choice_processor.process_mc_regular(
                    pipe, dataset, template_name, args, batch_indices, mapping
                )
            if args.debug:
                overall_acc = corr / tot if tot else 0
                print(f"\nBatch accuracy (Multiple Choice): {overall_acc:.2%}")
        else:
            if args.self_consistency:
                corr, tot, batch_results = numeric_processor.process_numeric_self_consistency(
                    pipe, dataset, template_name, args, batch_indices
                )
            else:
                corr, tot, batch_results = numeric_processor.process_numeric_batch(
                    pipe, dataset, template_name, args, batch_size, len(batch_indices), sample_indices=batch_indices
                )
        correct_total += corr
        total_processed += tot
        results.extend(batch_results)
    
    # Trim results to exactly 1000 if we got more.
    results = results[:1000]
    return correct_total, total_processed, results