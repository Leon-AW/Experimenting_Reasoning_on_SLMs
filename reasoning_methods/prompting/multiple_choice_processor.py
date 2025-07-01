from tqdm import tqdm
from collections import Counter
from .prompt_helper import format_prompt
from .config import DATASET_CONFIGS, TEMPERATURE, SELF_CONSISTENCY_PATHS, SEED
import torch
import numpy as np
import random

def get_mc_pred_answer(prompt, options, model, tokenizer, temperature=0.0, mapping=None):
    """
    Given a prompt and a list of candidate options, compute the log likelihood
    of each candidate answer when appended (after an "Answer:" marker) to the prompt.
    The candidate with the highest log likelihood is returned, converted to
    its numeric representation (mapping A->"0", B->"1", etc.).
    
    If temperature > 0, introduces randomness for self-consistency sampling.
    """
    # Ensure the prompt ends with an "Answer:" line.
    if not prompt.strip().endswith("Answer:"):
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

    # Convert scores to probabilities and select an answer
    logits = np.array([scores[letter] for letter in candidate_letters])
    
    if temperature == 0.0:
        best_letter = max(scores, key=scores.get)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        best_prob = np.max(probs)
    else:
        # Apply temperature scaling
        logits = logits / temperature
        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        probs = exp_logits / exp_logits.sum()
        # Sample from the distribution
        choice_idx = np.random.choice(len(candidate_letters), p=probs)
        best_letter = candidate_letters[choice_idx]
        best_prob = probs[choice_idx]
    
    # Convert candidate letter to a numeric string.
    if mapping is None:
        mapping = {"A": "0", "B": "1", "C": "2", "D": "3", "E": "4"}
    
    pred_answer = mapping.get(best_letter, best_letter)
    return pred_answer, best_prob

def process_mc_self_consistency(pipe, dataset, template_name, args, sample_indices, mapping):
    """
    Process multiple-choice datasets using self-consistency with confidence-weighted voting.
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

    for idx in tqdm(sample_indices, desc=f"Processing {template_name} (Confidence-Weighted SC)"):
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
            elif args.dataset in ["arc", "commonsense_qa"]:
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
            sc_results = []
            for _ in range(SELF_CONSISTENCY_PATHS):
                pred, prob = get_mc_pred_answer(
                    formatted_prompt, options, pipe.model, pipe.tokenizer, temperature=TEMPERATURE, mapping=mapping
                ) if options else (None, None)
                if pred is not None:
                    sc_results.append((pred, prob))

            # Confidence-weighted majority vote.
            if sc_results:
                weighted_scores = {}
                for answer, prob in sc_results:
                    weighted_scores[answer] = weighted_scores.get(answer, 0) + prob
                
                pred_answer = max(weighted_scores, key=weighted_scores.get)
                
                # Calculate confidence as the normalized score of the predicted answer
                total_prob = sum(weighted_scores.values())
                confidence = weighted_scores[pred_answer] / total_prob if total_prob > 0 else 0
            else:
                pred_answer = None
                confidence = 0.0

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
                "generated_text": f"Self-consistency results: {sc_results}",
                "pred_answer": pred_answer,
                "gold_answer": gold_answer,
                "is_correct": is_correct,
                "confidence": confidence,
                "sc_paths": str([res[0] for res in sc_results])
            })

            if args.debug:
                print(f"\nSample index {idx}:")
                print(f"Prompt: {formatted_prompt}")
                print(f"Self-consistency results: {sc_results}")
                print(f"Predicted answer: {pred_answer} (Confidence: {confidence:.2f})")
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
            elif args.dataset in ["arc", "commonsense_qa"]:
                choices = example.get("choices", {})
                if isinstance(choices, dict) and "text" in choices:
                    options = choices["text"]
            elif args.dataset == "mmlu":
                options = example.get("choices")
                if not options:
                    options = [example.get(f"choice_{i}", "") for i in range(4)]
            elif args.dataset == "agieval":
                options = example.get("options", [])

            formatted_prompt = format_prompt(template_name, args.dataset, question, options, passage)
            # Use single-pass log likelihood scoring.
            pred_answer, _ = get_mc_pred_answer(
                formatted_prompt, options, pipe.model, pipe.tokenizer, mapping=mapping
            ) if options else (None, None)

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