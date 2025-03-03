from tqdm import tqdm
from collections import Counter
from .prompts import format_prompt
from .config import DATASET_CONFIGS, TEMPERATURE, SELF_CONSISTENCY_PATHS, MAX_NEW_TOKENS, MIN_NEW_TOKENS
import torch
import numpy as np

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
    # Generate CoT with proper attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Set generation parameters
    generation_kwargs = {
        "max_new_tokens": cot_max_new_tokens,
        "min_new_tokens": cot_min_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "pad_token_id": tokenizer.eos_token_id,
        "no_repeat_ngram_size": 3  # Prevent repetitive text
    }
    
    output = model.generate(**inputs, **generation_kwargs)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Ensure the generated text ends with 'Answer:'
    if not generated_text.strip().endswith("Answer:"):
        base_context = generated_text.strip() + "\nAnswer:"
    else:
        base_context = generated_text.strip()

    # Create candidate letters for options
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
    return mapping.get(best_letter, best_letter), generated_text

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

def process_mc_self_consistency(pipe, dataset, template_name, args, sample_indices, mapping):
    """
    Process multiple-choice datasets using self-consistency with proper batching.
    Each batch processes a set of samples, with each sample going through all self-consistency paths.
    """
    correct = 0
    total = 0
    results = []
    
    # Process in batches
    for i in range(0, len(sample_indices), args.batch_size):
        batch_indices = sample_indices[i:i+args.batch_size]
        batch_results = []
        
        # Process each sample in the batch
        for idx in tqdm(batch_indices, desc=f"Processing batch {i//args.batch_size + 1}/{(len(sample_indices) + args.batch_size - 1)//args.batch_size} (Self-Consistency)"):
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
                generated_texts = []
                
                for _ in range(SELF_CONSISTENCY_PATHS):
                    if template_name == "simple":
                        pred = get_mc_pred_answer_simple(formatted_prompt, options, pipe.model, pipe.tokenizer, temperature=TEMPERATURE) if options else None
                        if pred is not None:
                            sc_answers.append(pred)
                    else:
                        pred, gen_text = get_mc_pred_answer_with_cot(formatted_prompt, options, pipe.model, pipe.tokenizer, temperature=TEMPERATURE) if options else (None, "")
                        if pred is not None:
                            sc_answers.append(pred)
                            generated_texts.append(gen_text)

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

                # Store the generated text for the result
                generated_text_for_result = ""
                if generated_texts:
                    # Use the first generated text for the result
                    generated_text_for_result = generated_texts[0]

                result = {
                    "sample_index": idx,
                    "question": question,
                    "prompt": formatted_prompt,
                    "generated_text": generated_text_for_result,
                    "pred_answer": pred_answer,
                    "gold_answer": gold_answer,
                    "is_correct": is_correct
                }
                
                batch_results.append(result)
                results.append(result)

                if args.debug:
                    print(f"\n{'='*80}")
                    print(f"SAMPLE INDEX: {idx}")
                    print(f"{'='*80}")
                    print(f"\n--- PROMPT ---\n{formatted_prompt}")
                    
                    if generated_texts:
                        print(f"\n--- GENERATED CHAIN-OF-THOUGHT ---\n{generated_texts[0]}")
                    
                    print(f"\n--- RESULTS ---")
                    print(f"Self-consistency answers: {sc_answers}")
                    print(f"Predicted answer: {pred_answer}")
                    print(f"Gold answer: {gold_answer}")
                    print(f"Correct: {is_correct}")
                    print(f"{'='*80}")

            except Exception as e:
                if args.debug:
                    print(f"\n{'='*80}")
                    print(f"ERROR processing sample index {idx}: {str(e)}")
                    print(f"{'='*80}")
        
        # Print batch summary if debug is enabled
        if args.debug and batch_results:
            batch_correct = sum(1 for r in batch_results if r["is_correct"])
            batch_total = len(batch_results)
            batch_acc = batch_correct / batch_total if batch_total else 0
            print(f"\nBatch {i//args.batch_size + 1} Accuracy: {batch_acc:.2%} ({batch_correct}/{batch_total})")
    
    return correct, total, results

def process_mc_regular(pipe, dataset, template_name, args, sample_indices, mapping):
    """
    Process multiple-choice datasets without self-consistency using batching.
    Each batch processes a set of samples in parallel.
    """
    correct = 0
    total = 0
    results = []

    # Process in batches
    for i in range(0, len(sample_indices), args.batch_size):
        batch_indices = sample_indices[i:i+args.batch_size]
        batch_results = []
        
        # Process each sample in the batch
        for idx in tqdm(batch_indices, desc=f"Processing batch {i//args.batch_size + 1}/{(len(sample_indices) + args.batch_size - 1)//args.batch_size}"):
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
                generated_text = ""
                if template_name == "simple":
                    pred_answer = get_mc_pred_answer_simple(formatted_prompt, options, pipe.model, pipe.tokenizer) if options else None
                else:
                    pred_answer, generated_text = get_mc_pred_answer_with_cot(formatted_prompt, options, pipe.model, pipe.tokenizer) if options else (None, "")

                is_correct = False
                if pred_answer is not None and gold_answer is not None:
                    is_correct = str(pred_answer).upper() == str(gold_answer).upper()

                if is_correct:
                    correct += 1
                total += 1

                result = {
                    "sample_index": idx,
                    "question": question,
                    "prompt": formatted_prompt,
                    "generated_text": generated_text,
                    "pred_answer": pred_answer,
                    "gold_answer": gold_answer,
                    "is_correct": is_correct
                }
                
                batch_results.append(result)
                results.append(result)

                if args.debug:
                    print(f"\n{'='*80}")
                    print(f"SAMPLE INDEX: {idx}")
                    print(f"{'='*80}")
                    print(f"\n--- PROMPT ---\n{formatted_prompt}")
                    
                    if generated_text:
                        print(f"\n--- GENERATED CHAIN-OF-THOUGHT ---\n{generated_text}")
                    
                    print(f"\n--- RESULTS ---")
                    print(f"Predicted answer (log likelihood): {pred_answer}")
                    print(f"Gold answer: {gold_answer}")
                    print(f"Correct: {is_correct}")
                    print(f"{'='*80}")

            except Exception as e:
                if args.debug:
                    print(f"\n{'='*80}")
                    print(f"ERROR processing sample index {idx}: {str(e)}")
                    print(f"{'='*80}")
        
        # Print batch summary if debug is enabled
        if args.debug and batch_results:
            batch_correct = sum(1 for r in batch_results if r["is_correct"])
            batch_total = len(batch_results)
            batch_acc = batch_correct / batch_total if batch_total else 0
            print(f"\nBatch {i//args.batch_size + 1} Accuracy: {batch_acc:.2%} ({batch_correct}/{batch_total})")
    
    return correct, total, results 