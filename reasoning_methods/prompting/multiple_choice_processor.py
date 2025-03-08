from tqdm import tqdm
from collections import Counter
from .prompt_helper import format_prompt
from .config import DATASET_CONFIGS, TEMPERATURE, SELF_CONSISTENCY_PATHS, MAX_NEW_TOKENS, MIN_NEW_TOKENS, TOP_P, TOP_K, SEED, CISC_ENABLED, CISC_TEMPERATURE, CISC_METHOD, CONFIDENCE_PROMPT_BINARY, CONFIDENCE_PROMPT_SCALE
import torch
import numpy as np
import math
import os
import random

# Try to import optimization config if available
try:
    from .optimization_config import is_running_on_a100, get_optimized_memory_config
    OPTIMIZATION_CONFIG_AVAILABLE = True
except ImportError:
    OPTIMIZATION_CONFIG_AVAILABLE = False

def get_mc_pred_answer_with_cot_batch(prompts, options_list, model, tokenizer, temperature=TEMPERATURE, 
                                     cot_max_new_tokens=MAX_NEW_TOKENS, cot_min_new_tokens=MIN_NEW_TOKENS):
    """
    Optimized batch version of get_mc_pred_answer_with_cot.
    Generates chain-of-thought for multiple prompts at once, then computes log-likelihoods.
    
    Parameters:
    - prompts: List of input prompts
    - options_list: List of options for each prompt
    - model: The language model
    - tokenizer: The tokenizer
    - temperature: Temperature for generation
    - cot_max_new_tokens: Maximum number of tokens to generate for CoT
    - cot_min_new_tokens: Minimum number of tokens to generate for CoT
    
    Returns:
    - List of (pred_answer, generated_text) tuples
    """
    if not prompts:
        return []
    
    # Generate CoT with proper attention mask for all prompts at once
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Set generation parameters
    generation_kwargs = {
        "max_new_tokens": cot_max_new_tokens,
        "min_new_tokens": cot_min_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
        "no_repeat_ngram_size": 3  # Prevent repetitive text
    }
    
    # Add temperature for sampling if do_sample is True
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = TOP_P
        generation_kwargs["top_k"] = TOP_K
    else:
        # Explicitly set temperature and top_p to None to override model defaults
        generation_kwargs["temperature"] = None
        generation_kwargs["top_p"] = None
        generation_kwargs["top_k"] = None
    
    # Generate all CoT texts at once
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
        
        # Decode all outputs
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    results = []
    
    # Process each generated text and compute log-likelihoods
    for i, (generated_text, options) in enumerate(zip(generated_texts, options_list)):
        # Ensure the generated text ends with 'Answer:'
        if not generated_text.strip().endswith("Final Answer:"):
            base_context = generated_text.strip() + "\nFinal Answer:"
        else:
            base_context = generated_text.strip()

        # Create candidate letters for options
        candidate_letters = [chr(65 + j) for j in range(len(options))]
        scores = {}

        # Tokenize with proper attention mask
        base_inputs = tokenizer(base_context, return_tensors="pt", padding=True)
        base_inputs = {k: v.to(model.device) for k, v in base_inputs.items()}
        base_length = base_inputs["input_ids"].shape[1]

        # Compute log-likelihoods for all candidate letters at once
        candidate_texts = [base_context + " " + letter for letter in candidate_letters]
        tokenized_candidates = tokenizer(candidate_texts, return_tensors="pt", padding=True)
        tokenized_candidates = {k: v.to(model.device) for k, v in tokenized_candidates.items()}
        
        # Create labels for loss computation (mask out the base context)
        labels = tokenized_candidates["input_ids"].clone()
        for j in range(len(candidate_letters)):
            labels[j, :base_length] = -100
        
        # Compute loss for all candidates at once
        with torch.no_grad():
            outputs = model(
                input_ids=tokenized_candidates["input_ids"],
                attention_mask=tokenized_candidates["attention_mask"],
                labels=labels
            )
        
        # Extract losses for each candidate
        losses = outputs.loss.item() if len(candidate_letters) == 1 else outputs.loss.tolist()
        
        # Convert to loglikelihoods
        for j, letter in enumerate(candidate_letters):
            loss = losses[j] if isinstance(losses, list) else losses
            candidate_len = tokenized_candidates["input_ids"].shape[1] - base_length
            total_loss = loss * candidate_len
            loglikelihood = -total_loss
            scores[letter] = loglikelihood

        # Select the best letter
        if temperature == 0.0:
            best_letter = max(scores, key=scores.get)
        else:
            logits = np.array([scores[letter] for letter in candidate_letters])
            logits = logits / temperature
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            choice_idx = np.random.choice(len(candidate_letters), p=probs)
            best_letter = candidate_letters[choice_idx]
        
        mapping = {"A": "0", "B": "1", "C": "2", "D": "3"}
        results.append((mapping.get(best_letter, best_letter), generated_text))
    
    return results

def get_mc_pred_answer_simple_batch(prompts, options_list, model, tokenizer, temperature=0.0):
    """
    Optimized batch version of get_mc_pred_answer_simple.
    Computes log-likelihoods for multiple prompts at once.
    
    Parameters:
    - prompts: List of input prompts
    - options_list: List of options for each prompt
    - model: The language model
    - tokenizer: The tokenizer
    - temperature: Temperature for sampling
    
    Returns:
    - List of predicted answers
    """
    if not prompts:
        return []
    
    results = []
    
    # Process in smaller sub-batches to avoid OOM
    max_sub_batch = 16  # Adjust based on available memory
    
    for i in range(0, len(prompts), max_sub_batch):
        sub_prompts = prompts[i:i+max_sub_batch]
        sub_options_list = options_list[i:i+max_sub_batch]
        
        sub_results = []
        
        # Process each prompt in the sub-batch
        for prompt, options in zip(sub_prompts, sub_options_list):
            if not prompt.strip().endswith("Answer:"):
                base_context = prompt.strip() + "\nAnswer:"
            else:
                base_context = prompt.strip()
                
            candidate_letters = [chr(65 + j) for j in range(len(options))]
            scores = {}
            
            # Tokenize with proper attention mask
            base_inputs = tokenizer(base_context, return_tensors="pt", padding=True)
            base_inputs = {k: v.to(model.device) for k, v in base_inputs.items()}
            base_length = base_inputs["input_ids"].shape[1]
            
            # Compute log-likelihoods for all candidate letters at once
            candidate_texts = [base_context + " " + letter for letter in candidate_letters]
            tokenized_candidates = tokenizer(candidate_texts, return_tensors="pt", padding=True)
            tokenized_candidates = {k: v.to(model.device) for k, v in tokenized_candidates.items()}
            
            # Create labels for loss computation (mask out the base context)
            labels = tokenized_candidates["input_ids"].clone()
            for j in range(len(candidate_letters)):
                labels[j, :base_length] = -100
            
            # Compute loss for all candidates at once
            with torch.no_grad():
                outputs = model(
                    input_ids=tokenized_candidates["input_ids"],
                    attention_mask=tokenized_candidates["attention_mask"],
                    labels=labels
                )
            
            # Extract losses for each candidate
            losses = outputs.loss.item() if len(candidate_letters) == 1 else outputs.loss.tolist()
            
            # Convert to loglikelihoods
            for j, letter in enumerate(candidate_letters):
                loss = losses[j] if isinstance(losses, list) else losses
                candidate_len = tokenized_candidates["input_ids"].shape[1] - base_length
                total_loss = loss * candidate_len
                loglikelihood = -total_loss
                scores[letter] = loglikelihood
            
            # Select the best letter
            if temperature == 0.0:
                best_letter = max(scores, key=scores.get)
            else:
                logits = np.array([scores[letter] for letter in candidate_letters])
                logits = logits / temperature
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / exp_logits.sum()
                choice_idx = np.random.choice(len(candidate_letters), p=probs)
                best_letter = candidate_letters[choice_idx]
            
            mapping = {"A": "0", "B": "1", "C": "2", "D": "3"}
            sub_results.append(mapping.get(best_letter, best_letter))
        
        results.extend(sub_results)
    
    return results

# Keep the original functions for backward compatibility
def get_mc_pred_answer_with_cot(prompt, options, model, tokenizer, temperature=TEMPERATURE, 
                               cot_max_new_tokens=MAX_NEW_TOKENS, cot_min_new_tokens=MIN_NEW_TOKENS):
    """
    Generate a chain-of-thought (CoT) first then append 'Answer:' and compute candidate loglikelihoods for multiple choice.
    Returns both the predicted answer and the generated CoT text.
    
    Parameters:
    - prompt: The input prompt
    - options: List of answer options
    - model: The language model
    - tokenizer: The tokenizer
    - temperature: Temperature for generation
    - cot_max_new_tokens: Maximum number of tokens to generate for CoT
    - cot_min_new_tokens: Minimum number of tokens to generate for CoT
    """
    results = get_mc_pred_answer_with_cot_batch([prompt], [options], model, tokenizer, temperature, 
                                              cot_max_new_tokens, cot_min_new_tokens)
    return results[0] if results else (None, "")

def get_mc_pred_answer_simple(prompt, options, model, tokenizer, temperature=0.0):
    """
    Berechnet die Loglikelihood für jede Antwortoption, ohne erst CoT zu generieren.
    """
    if not prompt.strip().endswith("Answer:"):
        base_context = prompt.strip() + "\nAnswer:"
    else:
        base_context = prompt.strip()
        
    candidate_letters = [chr(65 + i) for i in range(len(options))]
    scores = {}
    
    # Tokenize with proper attention mask
    base_inputs = tokenizer(base_context, return_tensors="pt", padding=True)
    base_inputs = {k: v.to(model.device) for k, v in base_inputs.items()}
    base_length = base_inputs["input_ids"].shape[1]
    
    for letter in candidate_letters:
        candidate_text = " " + letter
        full_input = base_context + candidate_text
        
        # Tokenize with proper attention mask
        tokenized = tokenizer(full_input, return_tensors="pt", padding=True)
        tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
        
        candidate_len = tokenized["input_ids"].shape[1] - base_length
        labels = tokenized["input_ids"].clone()
        labels[:, :base_length] = -100
        
        with torch.no_grad():
            outputs = model(input_ids=tokenized["input_ids"], 
                           attention_mask=tokenized["attention_mask"], 
                           labels=labels)
        loss = outputs.loss.item()
        total_loss = loss * candidate_len
        loglikelihood = -total_loss  
        scores[letter] = loglikelihood
    
    if temperature == 0.0:
        best_letter = max(scores, key=scores.get)
    else:
        logits = np.array([scores[letter] for letter in candidate_letters])
        logits = logits / temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        choice_idx = np.random.choice(len(candidate_letters), p=probs)
        best_letter = candidate_letters[choice_idx]
    
    mapping = {"A": "0", "B": "1", "C": "2", "D": "3"}
    return mapping.get(best_letter, best_letter)

def extract_confidence(model, tokenizer, prompt, answer_text, method="p_true"):
    """
    Extract confidence score from the model based on its generated answer.
    
    Parameters:
    - model: The language model
    - tokenizer: The tokenizer
    - prompt: The original prompt
    - answer_text: The generated answer text
    - method: Confidence extraction method (p_true, verbal_binary, verbal_scale, response_probability)
    
    Returns:
    - confidence_score: A float value representing the model's confidence
    """
    if method == "response_probability":
        # Use the model's probability of generating the response
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        answer_tokens = tokenizer(answer_text, return_tensors="pt")["input_ids"][0]
        answer_len = answer_tokens.shape[0]
        
        # Calculate log probability of the generated answer
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            logits = outputs.logits
            
            log_probs = 0
            for i in range(answer_len):
                if i < logits.shape[1] - 1:
                    next_token_logits = logits[0, i, :]
                    next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    if i < answer_tokens.shape[0]:
                        token_id = answer_tokens[i]
                        if token_id < next_token_probs.shape[0]:
                            log_probs += torch.log(next_token_probs[token_id]).item()
        
        # Return normalized probability
        return math.exp(log_probs / max(1, answer_len))
    
    elif method in ["p_true", "verbal_binary", "verbal_scale"]:
        # Combine prompt, answer and confidence prompt
        full_text = prompt + answer_text
        
        if method == "p_true" or method == "verbal_binary":
            # Add binary confidence prompt
            confidence_prompt = CONFIDENCE_PROMPT_BINARY
        else:
            # Add scale confidence prompt
            confidence_prompt = CONFIDENCE_PROMPT_SCALE
            
        full_text += confidence_prompt
        
        # Tokenize the full text
        inputs = tokenizer(full_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        if method == "p_true":
            # Calculate P(True) - probability assigned to token "1"
            with torch.no_grad():
                outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                next_token_logits = outputs.logits[0, -1, :]
                next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                
                # Get token IDs for "0" and "1"
                one_token_id = tokenizer.encode("1", add_special_tokens=False)[0]
                confidence = next_token_probs[one_token_id].item()
                return confidence
        else:
            # Generate confidence value
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            gen_tokens = model.generate(
                **inputs,
                max_new_tokens=10,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Extract the confidence value
            confidence_text = tokenizer.decode(gen_tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            if method == "verbal_binary":
                # Extract 0 or 1
                if "1" in confidence_text[:5]:
                    return 1.0
                else:
                    return 0.0
            else:
                # Extract numerical value from scale
                try:
                    # Find numbers in the generated text
                    import re
                    numbers = re.findall(r'\d+', confidence_text)
                    if numbers:
                        confidence = float(numbers[0])
                        return min(100, max(0, confidence)) / 100.0  # Normalize to [0, 1]
                except:
                    return 0.5  # Default if parsing fails
    
    return 0.5  # Default confidence

def process_mc_self_consistency(pipe, dataset, template_name, args, sample_indices, mapping):
    """
    Process multiple-choice datasets using self-consistency with optimized batching.
    Efficiently generates multiple paths in parallel for maximum performance.
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
    
    # Determine efficient batch size for self-consistency paths
    # We need to balance: (original batch size) * (paths per sample)
    if OPTIMIZATION_CONFIG_AVAILABLE and is_running_on_a100():
        # A100 optimization - we can process more samples at once
        memory_config = get_optimized_memory_config()
        effective_paths_per_batch = memory_config["sc_paths_per_batch"]
        samples_per_batch = max(1, args.batch_size // 2)  # More conservative for MC tasks
    else:
        # Conservative settings for other GPUs
        effective_paths_per_batch = min(SELF_CONSISTENCY_PATHS, 2)
        samples_per_batch = max(1, args.batch_size // 3)  # More conservative for MC tasks
    
    # Process in batches of samples
    num_batches = math.ceil(len(sample_indices) / samples_per_batch)
    
    # Main progress bar for batches
    with tqdm(total=num_batches, desc="Processing batches") as pbar:
        for batch_idx in range(num_batches):
            batch_start = batch_idx * samples_per_batch
            batch_end = min((batch_idx + 1) * samples_per_batch, len(sample_indices))
            batch_indices = sample_indices[batch_start:batch_end]
            
            print(f"Processing batch {batch_idx+1}/{num_batches} with {len(batch_indices)} samples")
            
            # Prepare all samples in the batch
            batch_data = []
            
            # Progress bar for samples within batch
            for sample_idx in tqdm(batch_indices, desc=f"Processing samples in batch {batch_idx+1}/{num_batches}", leave=False):
                try:
                    example = dataset[sample_idx]
                    question = example[DATASET_CONFIGS[args.dataset]["question_key"]]
                    gold_answer = str(example[DATASET_CONFIGS[args.dataset]["answer_key"]]).strip()

                    # Normalize the gold_answer to numeric if it is given as letter
                    if gold_answer.upper() in mapping:
                        gold_answer = mapping[gold_answer.upper()]

                    # Get available options and (if available) passage context
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
                    
                    batch_data.append({
                        "sample_idx": sample_idx,
                        "question": question,
                        "passage": passage if passage else "",
                        "gold_answer": gold_answer,
                        "prompt": formatted_prompt,
                        "options": options,
                        "sc_answers": [],
                        "sc_texts": [],  # Store all generated texts
                        "sc_confidences": [],  # Store confidence for each path
                        "generated_text": ""
                    })
                except Exception as e:
                    if args.debug:
                        print(f"Error preparing sample {sample_idx}: {e}")
                    continue
            
            # Process self-consistency paths in sub-batches
            if template_name == "simple":
                # For simple template, we can't batch the log-likelihood scoring
                for sample_data in batch_data:
                    for _ in range(SELF_CONSISTENCY_PATHS):
                        pred = get_mc_pred_answer_simple(sample_data["prompt"], sample_data["options"], pipe.model, pipe.tokenizer, temperature=TEMPERATURE) if sample_data["options"] else None
                        if pred is not None:
                            sample_data["sc_answers"].append(pred)
                            # Add default confidence for simple template (no generated text)
                            sample_data["sc_confidences"].append(0.5)
            else:
                # For CoT templates, we process paths in batches
                for path_batch_idx in range(0, SELF_CONSISTENCY_PATHS, effective_paths_per_batch):
                    path_batch_end = min(path_batch_idx + effective_paths_per_batch, SELF_CONSISTENCY_PATHS)
                    paths_in_current_batch = path_batch_end - path_batch_idx
                    
                    # For each path in the current batch
                    for _ in range(paths_in_current_batch):
                        # Prepare prompts and options for all samples
                        all_prompts = [sample["prompt"] for sample in batch_data]
                        all_options = [sample["options"] for sample in batch_data]
                        
                        # Generate CoT and get predictions for all samples at once
                        pred_results = get_mc_pred_answer_with_cot_batch(all_prompts, all_options, pipe.model, pipe.tokenizer, temperature=TEMPERATURE)
                        
                        # Store predictions and generated text for each sample
                        for i, (pred, generated_text) in enumerate(pred_results):
                            if i < len(batch_data):
                                if pred is not None:
                                    batch_data[i]["sc_answers"].append(pred)
                                    batch_data[i]["sc_texts"].append(generated_text)
                                    
                                    # Extract confidence if CISC is enabled
                                    if CISC_ENABLED:
                                        confidence = extract_confidence(
                                            pipe.model, 
                                            pipe.tokenizer, 
                                            batch_data[i]["prompt"], 
                                            generated_text, 
                                            method=CISC_METHOD
                                        )
                                        batch_data[i]["sc_confidences"].append(confidence)
                                    else:
                                        batch_data[i]["sc_confidences"].append(1.0)  # Equal weights when disabled
                                
                                # Store the first generated text for display
                                if not batch_data[i]["generated_text"] and generated_text:
                                    batch_data[i]["generated_text"] = generated_text
            
            # Process final answers for each sample in the batch
            for sample_data in batch_data:
                if sample_data["sc_answers"]:
                    if CISC_ENABLED:
                        # Using CISC: Confidence-weighted majority voting
                        answer_confidences = {}
                        
                        # Softmax normalization of confidence scores with temperature
                        confidences = np.array(sample_data["sc_confidences"])
                        
                        # Apply temperature scaling
                        scaled_confidences = confidences / CISC_TEMPERATURE
                        
                        # Softmax normalization
                        max_conf = np.max(scaled_confidences)
                        exp_confidences = np.exp(scaled_confidences - max_conf)
                        normalized_confidences = exp_confidences / np.sum(exp_confidences)
                        
                        # Weighted majority vote
                        for answer, confidence in zip(sample_data["sc_answers"], normalized_confidences):
                            if answer in answer_confidences:
                                answer_confidences[answer] += confidence
                            else:
                                answer_confidences[answer] = confidence
                        
                        # Get the answer with highest weighted count
                        pred_answer = max(answer_confidences, key=answer_confidences.get)
                    else:
                        # Standard self-consistency: use most frequent answer
                        answer_counts = Counter(sample_data["sc_answers"])
                        pred_answer = answer_counts.most_common(1)[0][0]
                    
                    is_correct = str(pred_answer).upper() == str(sample_data["gold_answer"]).upper()
                    if is_correct:
                        correct += 1
                    
                    total += 1
                    
                    # Include all SC paths in results
                    results.append({
                        "sample_index": sample_data["sample_idx"],
                        "question": sample_data["question"],
                        "passage": sample_data["passage"],
                        "prompt": sample_data["prompt"],
                        "generated_text": sample_data["generated_text"],
                        "pred_answer": pred_answer,
                        "gold_answer": sample_data["gold_answer"],
                        "is_correct": is_correct,
                        "sc_paths": [
                            {"answer": ans, "text": txt, "confidence": conf} 
                            for ans, txt, conf in zip(
                                sample_data["sc_answers"], 
                                sample_data["sc_texts"] if sample_data["sc_texts"] else [""] * len(sample_data["sc_answers"]),
                                sample_data["sc_confidences"]
                            )
                        ]
                    })
                    
                    if args.debug:
                        if CISC_ENABLED:
                            confidence = answer_confidences[pred_answer]
                            print(f"Sample {sample_data['sample_idx']} - Predicted: {pred_answer}, Gold: {sample_data['gold_answer']}, "
                                  f"Correct: {is_correct}, CISC Confidence: {confidence:.2f}, "
                                  f"SC Paths: {len(sample_data['sc_answers'])}/{SELF_CONSISTENCY_PATHS}")
                        else:
                            confidence = answer_counts[pred_answer] / len(sample_data["sc_answers"])
                            print(f"Sample {sample_data['sample_idx']} - Predicted: {pred_answer}, Gold: {sample_data['gold_answer']}, "
                                  f"Correct: {is_correct}, Frequency: {confidence:.2f}, "
                                  f"SC Paths: {len(sample_data['sc_answers'])}/{SELF_CONSISTENCY_PATHS}")
                else:
                    # No answers extracted
                    total += 1
                    if args.debug:
                        print(f"Sample {sample_data['sample_idx']} - No answers extracted")
            
            # Print batch progress
            accuracy = correct / total if total > 0 else 0
            pbar.set_postfix({"Accuracy": f"{accuracy:.2%}", "Correct": f"{correct}/{total}"})
            pbar.update(1)
    
    return correct, total, results

def process_mc_regular(pipe, dataset, template_name, args, sample_indices, mapping):
    """
    Process multiple-choice datasets without self-consistency using optimized batching.
    Efficiently processes samples in parallel for maximum performance.
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
    
    # Determine batch size based on available memory
    batch_size = args.batch_size
    
    # Process in batches
    num_batches = math.ceil(len(sample_indices) / batch_size)
    
    # Main progress bar for batches
    with tqdm(total=num_batches, desc="Processing batches") as pbar:
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(sample_indices))
            batch_indices = sample_indices[batch_start:batch_end]
            
            print(f"Processing batch {batch_idx+1}/{num_batches} with {len(batch_indices)} samples")
            
            # Prepare all samples in the batch
            batch_data = []
            
            # Progress bar for samples within batch
            for sample_idx in tqdm(batch_indices, desc=f"Processing samples in batch {batch_idx+1}/{num_batches}", leave=False):
                try:
                    example = dataset[sample_idx]
                    question = example[DATASET_CONFIGS[args.dataset]["question_key"]]
                    gold_answer = str(example[DATASET_CONFIGS[args.dataset]["answer_key"]]).strip()

                    # Normalize the gold_answer to numeric
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
                    
                    batch_data.append({
                        "sample_idx": sample_idx,
                        "question": question,
                        "passage": passage if passage else "",
                        "gold_answer": gold_answer,
                        "prompt": formatted_prompt,
                        "options": options,
                        "generated_text": ""
                    })
                except Exception as e:
                    if args.debug:
                        print(f"Error preparing sample {sample_idx}: {e}")
                    continue
            
            if not batch_data:
                continue
            
            all_prompts = [data["prompt"] for data in batch_data]
            all_options = [data["options"] for data in batch_data]
            
            try:
                if template_name == "simple":
                    # Use single-pass log likelihood scoring.
                    for i, sample_data in enumerate(batch_data):
                        pred_answer = get_mc_pred_answer_simple(sample_data["prompt"], sample_data["options"], pipe.model, pipe.tokenizer) if sample_data["options"] else None
                        sample_data["pred_answer"] = pred_answer
                        sample_data["generated_text"] = ""  # No generated text for simple template
                else:
                    results_batch = get_mc_pred_answer_with_cot_batch(
                        all_prompts,
                        all_options,
                        pipe.model,
                        pipe.tokenizer,
                        temperature=TEMPERATURE
                    )
                    
                    for sample_data, (pred_answer, generated_text) in zip(batch_data, results_batch):
                        sample_data["pred_answer"] = pred_answer
                        sample_data["generated_text"] = generated_text
            
            except Exception as e:
                if args.debug:
                    print(f"Error in batch generation: {e}")
                continue
            
            for sample_data in batch_data:
                pred_answer = sample_data.get("pred_answer")
                
                if pred_answer is not None:
                    is_correct = str(pred_answer).upper() == str(sample_data["gold_answer"]).upper()
                    if is_correct:
                        correct += 1
                    
                    result = {
                        "sample_index": sample_data["sample_idx"],
                        "question": sample_data["question"],
                        "passage": sample_data["passage"],
                        "prompt": sample_data["prompt"],
                        "generated_text": sample_data["generated_text"],
                        "pred_answer": pred_answer,
                        "gold_answer": sample_data["gold_answer"],
                        "is_correct": is_correct
                    }
                    
                    results.append(result)
                    
                    if args.debug:
                        print(f"Sample {sample_data['sample_idx']} - Predicted: {pred_answer}, Gold: {sample_data['gold_answer']}, Correct: {is_correct}")
                
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            pbar.set_postfix({"Accuracy": f"{accuracy:.2%}", "Correct": f"{correct}/{total}"})
            pbar.update(1)
    
    return correct, total, results 