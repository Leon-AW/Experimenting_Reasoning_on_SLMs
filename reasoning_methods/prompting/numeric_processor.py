from tqdm import tqdm
from .answer_extraction import extract_numeric_answer, extract_gold_gsm8k_answer, extract_drop_answer
from .prompts import format_prompt
from .config import (
    DATASET_CONFIGS, MIN_NEW_TOKENS, MAX_NEW_TOKENS, TEMPERATURE, 
    SELF_CONSISTENCY_PATHS, TOP_P, TOP_K, DO_SAMPLE, NUM_RETURN_SEQUENCES, 
    SEED, CISC_ENABLED, CISC_TEMPERATURE, CISC_METHOD,
    CONFIDENCE_PROMPT_BINARY, CONFIDENCE_PROMPT_SCALE
)
import torch
from collections import Counter
import math
import numpy as np
import random
import re  # For regex extraction

# Import the confidence extraction function from multiple_choice_processor to avoid duplication
try:
    from .multiple_choice_processor import extract_confidence
except ImportError:
    # If import fails, define the function here
    def extract_confidence(model, tokenizer, prompt, answer_text, method="p_true"):
        """
        Extract confidence score from the model based on its generated answer.
        See multiple_choice_processor.py for full implementation.
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
                        numbers = re.findall(r'\d+', confidence_text)
                        if numbers:
                            confidence = float(numbers[0])
                            return min(100, max(0, confidence)) / 100.0  # Normalize to [0, 1]
                    except:
                        return 0.5  # Default if parsing fails
        
        return 0.5  # Default confidence

def process_numeric_self_consistency(pipe, dataset, template_name, args, sample_indices):
    """
    Process numeric datasets (e.g., GSM8K, DROP) using self-consistency with efficient batching.
    Now enhanced with Confidence-Informed Self-Consistency (CISC) for more efficient sampling.
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
    if torch.cuda.is_available() and any("A100" in torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())):
        effective_paths_per_batch = min(SELF_CONSISTENCY_PATHS, 5)
        samples_per_batch = max(1, args.batch_size // effective_paths_per_batch)
    else:
        effective_paths_per_batch = min(SELF_CONSISTENCY_PATHS, 2)
        samples_per_batch = max(1, args.batch_size // effective_paths_per_batch)
    
    num_batches = math.ceil(len(sample_indices) / samples_per_batch)
    
    # Main progress bar for batches
    with tqdm(total=num_batches, desc="Processing batches") as pbar:
        for batch_idx in range(num_batches):
            batch_start = batch_idx * samples_per_batch
            batch_end = min((batch_idx + 1) * samples_per_batch, len(sample_indices))
            batch_indices = sample_indices[batch_start:batch_end]
            
            batch_results = []
            
            # Progress bar for samples within batch
            for sample_idx in tqdm(batch_indices, desc=f"Processing samples in batch {batch_idx+1}/{num_batches}", leave=False):
                try:
                    # Handle wrapped indices for smaller datasets
                    actual_idx = sample_idx % len(dataset)
                    example = dataset[actual_idx]
                    question = example[DATASET_CONFIGS[args.dataset]["question_key"]]
                    
                    if args.dataset == "drop":
                        answers_spans = example[DATASET_CONFIGS[args.dataset]["answer_key"]]
                        if isinstance(answers_spans, dict) and 'spans' in answers_spans and len(answers_spans['spans']) > 0:
                            raw_gold_answer = answers_spans['spans'][0]
                        else:
                            raw_gold_answer = str(answers_spans)
                    else:
                        raw_gold_answer = str(example[DATASET_CONFIGS[args.dataset]["answer_key"]]).strip()

                    if args.dataset == "gsm8k":
                        gold_answer = extract_gold_gsm8k_answer(raw_gold_answer)
                    elif args.dataset == "drop":
                        gold_answer = raw_gold_answer.strip()
                    else:
                        gold_answer = extract_numeric_answer(raw_gold_answer)

                    passage = None
                    if args.dataset == "drop" and "passage" in example:
                        passage = example["passage"]

                    options = None
                    formatted_prompt = format_prompt(template_name, args.dataset, question, options, passage)
                    
                    batch_results.append({
                        "sample_idx": sample_idx,  # Keep original index for final results
                        "actual_idx": actual_idx,  # Store actual dataset index
                        "question": question,
                        "passage": passage if args.dataset == "drop" else "",
                        "prompt": formatted_prompt,
                        "gold_answer": gold_answer,
                        "sc_answers": [],
                        "sc_texts": [],  # Store all generated texts
                        "sc_confidences": [],  # Store confidence scores for CISC
                    })
                except Exception as e:
                    if args.debug:
                        print(f"Error preparing sample {sample_idx} (actual index {sample_idx % len(dataset)}): {e}")
                    continue
            
            # Progress bar for SC paths
            for path_batch_idx in tqdm(range(0, SELF_CONSISTENCY_PATHS, effective_paths_per_batch), 
                                     desc=f"Generating SC paths for batch {batch_idx+1}", leave=False):
                path_batch_end = min(path_batch_idx + effective_paths_per_batch, SELF_CONSISTENCY_PATHS)
                paths_in_current_batch = path_batch_end - path_batch_idx
                
                all_prompts = [result["prompt"] for result in batch_results]
                if not all_prompts:
                    continue
                    
                tokenized_inputs = pipe.tokenizer(
                    all_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                tokenized_inputs = {k: v.to(pipe.model.device) for k, v in tokenized_inputs.items()}
                
                generation_kwargs = {
                    "min_new_tokens": MIN_NEW_TOKENS,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "do_sample": True,
                    "pad_token_id": pipe.tokenizer.eos_token_id,
                    "num_return_sequences": paths_in_current_batch,
                }
                
                try:
                    with torch.no_grad():
                        outputs = pipe.model.generate(
                            **tokenized_inputs,
                            **generation_kwargs
                        )
                        
                        for sample_idx, result in enumerate(batch_results):
                            prompt_len = len(pipe.tokenizer.encode(result["prompt"]))
                            
                            for path_idx in range(paths_in_current_batch):
                                output_idx = sample_idx * paths_in_current_batch + path_idx
                                
                                if output_idx >= len(outputs):
                                    break
                                    
                                output_seq = outputs[output_idx]
                                generated_text = pipe.tokenizer.decode(output_seq, skip_special_tokens=True)
                                model_response = generated_text[len(result["prompt"]):].strip()
                                
                                if args.dataset == "drop":
                                    extracted_answer = extract_drop_answer(model_response)
                                    if extracted_answer is not None:
                                        result["sc_answers"].append(extracted_answer)
                                        result["sc_texts"].append(model_response)  # Store the generated text
                                        
                                        # Extract confidence score for CISC if enabled
                                        if CISC_ENABLED:
                                            confidence = extract_confidence(
                                                pipe.model,
                                                pipe.tokenizer,
                                                result["prompt"],
                                                model_response,
                                                method=CISC_METHOD
                                            )
                                            result["sc_confidences"].append(confidence)
                                else:
                                    numeric_extracted = extract_numeric_answer(model_response)
                                    if numeric_extracted is not None:
                                        result["sc_answers"].append(str(int(numeric_extracted)))
                                        result["sc_texts"].append(model_response)  # Store the generated text
                                        
                                        # Extract confidence score for CISC if enabled
                                        if CISC_ENABLED:
                                            confidence = extract_confidence(
                                                pipe.model,
                                                pipe.tokenizer,
                                                result["prompt"],
                                                model_response,
                                                method=CISC_METHOD
                                            )
                                            result["sc_confidences"].append(confidence)
                                
                                if path_idx == 0 and "generated_text" not in result:
                                    result["generated_text"] = model_response
                
                except Exception as batch_error:
                    if args.debug:
                        print(f"Error in batch generation: {batch_error}")
            
            for result in batch_results:
                if result["sc_answers"]:
                    if CISC_ENABLED and result["sc_confidences"]:
                        # Using CISC: Confidence-weighted majority voting
                        answer_confidences = {}
                        
                        # Ensure we have confidence scores for all answers
                        if len(result["sc_confidences"]) < len(result["sc_answers"]):
                            # Add default confidence scores for any missing values
                            result["sc_confidences"].extend([0.5] * (len(result["sc_answers"]) - len(result["sc_confidences"])))
                            
                        # Softmax normalization of confidence scores with temperature
                        confidences = np.array(result["sc_confidences"])
                        
                        # Apply temperature scaling
                        scaled_confidences = confidences / CISC_TEMPERATURE
                        
                        # Softmax normalization
                        max_conf = np.max(scaled_confidences)
                        exp_confidences = np.exp(scaled_confidences - max_conf)
                        normalized_confidences = exp_confidences / np.sum(exp_confidences)
                        
                        # Weighted majority vote
                        for answer, confidence in zip(result["sc_answers"], normalized_confidences):
                            if answer in answer_confidences:
                                answer_confidences[answer] += confidence
                            else:
                                answer_confidences[answer] = confidence
                        
                        # Get the answer with highest weighted count
                        pred_answer = max(answer_confidences, key=answer_confidences.get)
                        
                        # Store the confidence value for logging
                        confidence_value = answer_confidences[pred_answer]
                    else:
                        # Standard self-consistency: use most frequent answer
                        answer_counts = Counter(result["sc_answers"])
                        pred_answer = answer_counts.most_common(1)[0][0]
                        confidence_value = answer_counts[pred_answer] / len(result["sc_answers"])
                    
                    # For DROP dataset, normalize answers before comparison
                    if args.dataset == "drop" and pred_answer is not None and result["gold_answer"] is not None:
                        # Normalize answers by removing trailing punctuation and normalizing whitespace
                        normalized_pred = pred_answer.strip().rstrip('.,:;!?').strip()
                        normalized_gold = result["gold_answer"].strip().rstrip('.,:;!?').strip()
                        
                        # Check for exact match after normalization
                        is_correct = normalized_pred == normalized_gold
                        
                        # If not an exact match, check if any word in the predicted answer matches the gold answer
                        if not is_correct and ' ' in normalized_pred:
                            # Split the predicted answer into words
                            pred_words = normalized_pred.split()
                            # Check if any word in the predicted answer matches the gold answer
                            is_correct = normalized_gold in pred_words
                    else:
                        is_correct = pred_answer == result["gold_answer"]
                    
                    if is_correct:
                        correct += 1
                    
                    total += 1
                    
                    # Include all SC paths in results
                    results.append({
                        "sample_index": result["sample_idx"],
                        "question": result["question"],
                        "passage": result["passage"],
                        "prompt": result["prompt"],
                        "generated_text": result.get("generated_text", ""),
                        "pred_answer": pred_answer,
                        "gold_answer": result["gold_answer"],
                        "is_correct": is_correct,
                        "confidence": confidence_value,
                        "sc_paths": [
                            {"answer": ans, "text": txt, "confidence": conf} 
                            for ans, txt, conf in zip(
                                result["sc_answers"], 
                                result["sc_texts"], 
                                result["sc_confidences"] if CISC_ENABLED and result["sc_confidences"] else [1.0] * len(result["sc_answers"])
                            )
                        ],
                        "sc_answers": result["sc_answers"],
                        "sc_texts": result["sc_texts"],
                        "sc_confidences": result["sc_confidences"] if CISC_ENABLED and result["sc_confidences"] else [1.0] * len(result["sc_answers"])
                    })
                    
                    if args.debug:
                        print(f"Sample {result['sample_idx']} - Predicted: {pred_answer}, Gold: {result['gold_answer']}, "
                              f"Correct: {is_correct}, {'CISC' if CISC_ENABLED else 'SC'} Confidence: {confidence_value:.2f}, "
                              f"SC Paths: {len(result['sc_answers'])}/{SELF_CONSISTENCY_PATHS}")
                else:
                    # Skip samples with no extracted answers
                    if args.debug:
                        print(f"Sample {result['sample_idx']} - No answers extracted")
            
            # Print batch progress
            accuracy = correct / total if total > 0 else 0
            pbar.set_postfix({"Accuracy": f"{accuracy:.2%}", "Correct": f"{correct}/{total}"})
            pbar.update(1)
    
    return correct, total, results

def process_numeric_batch(pipe, dataset, template_name, args, batch_size, max_samples):
    """
    Process numeric datasets (e.g., GSM8K, DROP) using normal generation.
    Enhanced for better performance with efficient batching.
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

    # Get the sample indices from the process_dataset_batch function
    sample_indices = list(range(0, max_samples))
    if hasattr(args, 'current_sample_indices') and args.current_sample_indices:
        sample_indices = args.current_sample_indices[:max_samples]

    # Process in batches with progress bar
    with tqdm(total=max_samples, desc="Processing samples") as pbar:
        for i in range(0, max_samples, batch_size):
            batch_indices = sample_indices[i:min(i + batch_size, max_samples)]
            batch_questions = []
            batch_gold_answers = []
            batch_examples = []
            batch_prompts = []
            batch_passages = []
            batch_real_indices = []  # To keep track of the original dataset indices
            
            for sample_idx in batch_indices:
                try:
                    # Get the actual index in the dataset (which might be wrapped around for smaller datasets)
                    actual_idx = sample_idx % len(dataset)
                    example = dataset[actual_idx]
                    batch_examples.append(example)
                    batch_real_indices.append(sample_idx)
                    
                    question = example[DATASET_CONFIGS[args.dataset]["question_key"]]
                    batch_questions.append(question)

                    if args.dataset == "drop":
                        answers_spans = example[DATASET_CONFIGS[args.dataset]["answer_key"]]
                        if isinstance(answers_spans, dict) and 'spans' in answers_spans and len(answers_spans['spans']) > 0:
                            raw_gold_answer = answers_spans['spans'][0]
                        else:
                            raw_gold_answer = str(answers_spans)
                    else:
                        raw_gold_answer = str(example[DATASET_CONFIGS[args.dataset]["answer_key"]]).strip()

                    if args.dataset == "gsm8k":
                        gold_answer = extract_gold_gsm8k_answer(raw_gold_answer)
                    elif args.dataset == "drop":
                        gold_answer = raw_gold_answer.strip()
                    else:
                        gold_answer = extract_numeric_answer(raw_gold_answer)
                        
                    batch_gold_answers.append(gold_answer)

                    if args.dataset == "drop" and "passage" in example:
                        passage = example["passage"]
                    else:
                        passage = ""
                    options = None
                    formatted_prompt = format_prompt(template_name, args.dataset, question, options, passage)
                    batch_prompts.append(formatted_prompt)
                    batch_passages.append(passage)
                except Exception as e:
                    if args.debug:
                        print(f"Error processing example at index {sample_idx}: {str(e)}")
                    total += 1
                    pbar.update(1)
                    continue

            if not batch_prompts:
                continue
                
            tokenized_inputs = pipe.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            tokenized_inputs = {k: v.to(pipe.model.device) for k, v in tokenized_inputs.items()}

            generation_kwargs = {
                "min_new_tokens": MIN_NEW_TOKENS,
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "do_sample": DO_SAMPLE,
                "pad_token_id": pipe.tokenizer.eos_token_id,
                "num_return_sequences": NUM_RETURN_SEQUENCES
            }
            
            try:
                with torch.no_grad():
                    outputs = pipe.model.generate(
                        **tokenized_inputs,
                        **generation_kwargs
                    )
                    
                    for batch_idx, (prompt, gold_answer, question, real_idx) in enumerate(zip(batch_prompts, batch_gold_answers, batch_questions, batch_real_indices)):
                        output_idx = batch_idx * NUM_RETURN_SEQUENCES
                        if output_idx >= len(outputs):
                            continue
                            
                        output_seq = outputs[output_idx]
                        generated_text = pipe.tokenizer.decode(output_seq, skip_special_tokens=True)
                        model_response = generated_text[len(prompt):].strip()
                        
                        if args.dataset == "drop":
                            pred_answer = extract_drop_answer(model_response)
                            pred_answer_str = pred_answer if pred_answer is not None else None
                        else:
                            pred_answer = extract_numeric_answer(model_response)
                            pred_answer_str = str(int(pred_answer)) if pred_answer is not None else None
                        
                        # For DROP dataset, normalize answers before comparison
                        if args.dataset == "drop" and pred_answer_str is not None and gold_answer is not None:
                            # Normalize answers by removing trailing punctuation and normalizing whitespace
                            normalized_pred = pred_answer_str.strip().rstrip('.,:;!?').strip()
                            normalized_gold = gold_answer.strip().rstrip('.,:;!?').strip()
                            
                            # Check for exact match after normalization
                            is_correct = normalized_pred == normalized_gold
                            
                            # If not an exact match, check if any word in the predicted answer matches the gold answer
                            if not is_correct and ' ' in normalized_pred:
                                # Split the predicted answer into words
                                pred_words = normalized_pred.split()
                                # Check if any word in the predicted answer matches the gold answer
                                is_correct = normalized_gold in pred_words
                        else:
                            is_correct = pred_answer_str == gold_answer
                        
                        if is_correct:
                            correct += 1
                        
                        passage = batch_passages[batch_idx] if args.dataset == "drop" else ""
                        result = {
                            "sample_index": real_idx,
                            "question": batch_questions[batch_idx],
                            "passage": passage,
                            "prompt": prompt,
                            "generated_text": model_response,
                            "pred_answer": pred_answer_str,
                            "gold_answer": gold_answer,
                            "is_correct": is_correct,
                        }
                        
                        results.append(result)
                        total += 1
                        
                        if args.debug:
                            print(f"Example {real_idx}:")
                            if args.dataset == "drop":
                                print(f"Passage: {passage}")
                            print(f"Question: {question}")
                            print(f"Response: {model_response[:100]}...")
                            print(f"Extracted Answer: {pred_answer_str}")
                            print(f"Gold Answer: {gold_answer}")
                            print(f"Is Correct: {is_correct}")
                            print("-" * 50)
                        
                        pbar.update(1)

            except Exception as e:
                if args.debug:
                    print(f"Error processing batch {i//batch_size}: {str(e)}")
                pbar.update(len(batch_indices))
            
            accuracy = correct / total if total > 0 else 0
            pbar.set_postfix({"Accuracy": f"{accuracy:.2%}", "Correct": f"{correct}/{total}"})

    return correct, total, results 