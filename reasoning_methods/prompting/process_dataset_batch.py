from reasoning_methods.prompting import multiple_choice_processor
from reasoning_methods.prompting import numeric_processor
from .config import DATASET_CONFIGS

def process_dataset_batch(pipe, dataset, template_name, args, batch_size):
    """Dispatcher function to process dataset batches, routing processing depending on the dataset type 
    and self-consistency flag.
    
    For multiple-choice benchmarks (e.g. RACE, ARC, MMLU, AGIEVAL) no freeform text is generated. Instead, log likelihood is used.
    For numeric tasks (GSM8K, DROP), text generation and answer extraction are performed.
    """
    correct = 0
    total = 0
    results = []

    # Determine sample indices.
    if args.dataset == "mmlu":
        dataset_size = len(dataset)
        all_indices = []
        while len(all_indices) < 1000:
            all_indices.extend(range(dataset_size))
        sample_indices = all_indices[:1000]  # Exactly 1000 samples.
        max_samples = 1000
    else:
        max_samples = min(1000, len(dataset))
        sample_indices = list(range(max_samples))

    # Define which datasets are numeric vs. multiple-choice.
    multiple_choice_datasets = ["race", "arc", "mmlu", "agieval"]

    # A mapping for answer normalization.
    mapping = {"A": "0", "B": "1", "C": "2", "D": "3"}

    # Store batch_size in args for access in processor functions
    args.batch_size = batch_size

    if args.dataset in multiple_choice_datasets:
        # Multiple Choice Processing
        if args.self_consistency:
            correct, total, results = multiple_choice_processor.process_mc_self_consistency(
                pipe, dataset, template_name, args, sample_indices, mapping
            )
        else:
            correct, total, results = multiple_choice_processor.process_mc_regular(
                pipe, dataset, template_name, args, sample_indices, mapping
            )
        if args.debug:
            overall_acc = correct / total if total else 0
            print(f"\nOverall accuracy (Multiple Choice): {overall_acc:.2%}")
        return correct, total, results

    else:
        # Numeric Processing
        if args.self_consistency:
            correct, total, results = numeric_processor.process_numeric_self_consistency(
                pipe, dataset, template_name, args, sample_indices
            )
        else:
            correct, total, results = numeric_processor.process_numeric_batch(
                pipe, dataset, template_name, args, batch_size, max_samples
            )
        return correct, total, results