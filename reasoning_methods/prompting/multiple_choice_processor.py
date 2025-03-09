from tqdm import tqdm
from collections import Counter
from .prompt_helper import format_prompt
from .config import DATASET_CONFIGS, TEMPERATURE, SELF_CONSISTENCY_PATHS, SEED
import torch
import numpy as np
import random

def get_mc_pred_answer(prompt, options, model, tokenizer, temperature=0.0):
    """
    Given a prompt and a list of candidate options, compute the log likelihood
    of each candidate answer when appended (after an "Answer:" marker) to the prompt.
    The candidate with the highest log likelihood is returned, converted to
    its numeric representation (mapping A->"0", B->"1", etc.).
    
    If temperature > 0, introduces randomness for self-consistency sampling.
    """
    # Ensure the prompt ends with an "Answer:" line.
    if not prompt.strip().endswith("Answer: "):
        base_context = prompt.strip() + "\nThe correct answer is: "
    else:
        base_context = prompt.strip()
        
    # Create candidate letters corresponding to options (i.e. "A", "B", etc.)
    candidate_letters = [chr(65 + i) for i in range(len(options))]
    scores = {}
    
    # Pre-tokenize the base context for reuse.
    base_tokens = tokenizer(base_context, return_tensors="pt").input_ids.to(model.device)
    base_length = base_tokens.shape[1]
    
    for letter in candidate_letters:
        # Append the candidate letter (with a leading space)
        candidate_text = " " + letter
        full_input = base_context + candidate_text
        tokenized = tokenizer(full_input, return_tensors="pt").to(model.device)
        
        # Candidate tokens are those after the base context.
        candidate_len = tokenized.input_ids.shape[1] - base_length
        
        # Set up the labels so that base context tokens are ignored.
        labels = tokenized.input_ids.clone()
        labels[:, :base_length] = -100
        
        with torch.no_grad():
            outputs = model(tokenized.input_ids, labels=labels)
        # The loss returned is the average negative log likelihood over candidate tokens;
        # multiply by candidate token count to get the total (negative) log likelihood.
        loss = outputs.loss.item()
        total_loss = loss * candidate_len
        # Higher log likelihood means lower loss, so we negate.
        loglikelihood = -total_loss  
        scores[letter] = loglikelihood

    # If temperature is 0, just return the highest scoring option
    if temperature == 0.0:
        best_letter = max(scores, key=scores.get)
    else:
        # Convert scores to probabilities using softmax with temperature
        logits = np.array([scores[letter] for letter in candidate_letters])
        # Apply temperature scaling
        logits = logits / temperature
        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        probs = exp_logits / exp_logits.sum()
        # Sample from the distribution
        choice_idx = np.random.choice(len(candidate_letters), p=probs)
        best_letter = candidate_letters[choice_idx]
    
    # Convert candidate letter to a numeric string.
    mapping = {"A": "0", "B": "1", "C": "2", "D": "3"}
    return mapping.get(best_letter, best_letter)

def process_mc_self_consistency(pipe, dataset, template_name, args, sample_indices, mapping):
    """
    Process multiple-choice datasets using self-consistency.
    Loops over sample_indices and obtains multiple self-consistent answers via log-likelihood scoring.
    """
    # Set seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        
    correct = 0
    total = 0
    results = []

    for idx in tqdm(sample_indices, desc=f"Processing {template_name} (Self-Consistency Log Likelihood)"):
        try:
            example = dataset[idx]
            question = example[DATASET_CONFIGS[args.dataset]["question_key"]]
            gold_answer = str(example[DATASET_CONFIGS[args.dataset]["answer_key"]]).strip()

            # Normalize the gold_answer to numeric if it is given as letter.
            if gold_answer.upper() in mapping:
                gold_answer = mapping[gold_answer.upper()]

            # Get available options and (if available) passage context.
            options = None
            passage = None
            if args.dataset == "race":
                options = example.get("options", [])
                passage = example.get("article", "") or example.get("passage", "")
            elif args.dataset == "arc":
                choices = example.get("choices", {})
                if isinstance(choices, dict) and "text" in choices:
                    options = choices["text"]
                else:
                    continue
            elif args.dataset == "mmlu":
                options = example.get("choices")
                if not options:
                    options = [example.get(f"choice_{i}", "") for i in range(4)]
            elif args.dataset == "agieval":
                options = example.get("options", [])
            
            formatted_prompt = format_prompt(template_name, args.dataset, question, options, passage)

            # Run self-consistency paths using log-likelihood with temperature.
            sc_answers = []
            for _ in range(SELF_CONSISTENCY_PATHS):
                pred = get_mc_pred_answer(
                    formatted_prompt, options, pipe.model, pipe.tokenizer, temperature=TEMPERATURE
                ) if options else None
                if pred is not None:
                    sc_answers.append(pred)

            # Majority vote of the self-consistency answers.
            if sc_answers:
                counts = Counter(sc_answers)
                pred_answer = counts.most_common(1)[0][0]
            else:
                pred_answer = None

            is_correct = False
            if pred_answer is not None and gold_answer is not None:
                is_correct = str(pred_answer).upper() == str(gold_answer).upper()

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "sample_index": idx,
                "question": question,
                "prompt": formatted_prompt,
                "generated_text": f"Self-consistency answers: {sc_answers}",
                "pred_answer": pred_answer,
                "gold_answer": gold_answer,
                "is_correct": is_correct
            })

            if args.debug:
                print(f"\nSample index {idx}:")
                print(f"Prompt: {formatted_prompt}")
                print(f"Self-consistency answers: {sc_answers}")
                print(f"Predicted answer: {pred_answer}")
                print(f"Gold answer: {gold_answer}")
                print(f"Correct: {is_correct}")

        except Exception as e:
            if args.debug:
                print(f"Error processing sample index {idx}: {str(e)}")
    return correct, total, results

def process_mc_regular(pipe, dataset, template_name, args, sample_indices, mapping):
    """
    Process multiple-choice datasets without self-consistency.
    For each sample, use log-likelihood scoring once to pick the candidate answer.
    """
    # Set seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        
    correct = 0
    total = 0
    results = []

    for idx in tqdm(sample_indices, desc=f"Processing {template_name} (Log Likelihood)"):
        try:
            example = dataset[idx]
            question = example[DATASET_CONFIGS[args.dataset]["question_key"]]
            gold_answer = str(example[DATASET_CONFIGS[args.dataset]["answer_key"]]).strip()

            # Normalize the gold_answer to numeric.
            if gold_answer.upper() in mapping:
                gold_answer = mapping[gold_answer.upper()]

            options = None
            passage = None
            if args.dataset == "race":
                options = example.get("options", [])
                passage = example.get("article", "") or example.get("passage", "")
            elif args.dataset == "arc":
                choices = example.get("choices", {})
                if isinstance(choices, dict) and "text" in choices:
                    options = choices["text"]
                else:
                    continue
            elif args.dataset == "mmlu":
                options = example.get("choices")
                if not options:
                    options = [example.get(f"choice_{i}", "") for i in range(4)]
            elif args.dataset == "agieval":
                options = example.get("options", [])

            formatted_prompt = format_prompt(template_name, args.dataset, question, options, passage)
            # Use single-pass log likelihood scoring.
            pred_answer = get_mc_pred_answer(
                formatted_prompt, options, pipe.model, pipe.tokenizer
            ) if options else None

            is_correct = False
            if pred_answer is not None and gold_answer is not None:
                is_correct = str(pred_answer).upper() == str(gold_answer).upper()

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "sample_index": idx,
                "prompt": formatted_prompt,
                "generated_text": "",  # No generated text is produced.
                "pred_answer": pred_answer,
                "gold_answer": gold_answer,
                "is_correct": is_correct
            })

            if args.debug:
                print(f"\nSample index {idx}:")
                print(f"Prompt: {formatted_prompt}")
                print(f"Predicted answer (log likelihood): {pred_answer}")
                print(f"Gold answer: {gold_answer}")
                print(f"Correct: {is_correct}")

        except Exception as e:
            if args.debug:
                print(f"Error processing sample index {idx}: {str(e)}")
    return correct, total, results 