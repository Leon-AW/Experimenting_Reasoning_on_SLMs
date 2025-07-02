from tqdm import tqdm
from .answer_extraction import extract_numeric_answer, extract_gold_gsm8k_answer, extract_drop_answer
from .prompt_helper import format_prompt
from .config import (
    DATASET_CONFIGS, MIN_NEW_TOKENS, MAX_NEW_TOKENS, TEMPERATURE, 
    SELF_CONSISTENCY_PATHS, TOP_P, TOP_K, DO_SAMPLE, NUM_RETURN_SEQUENCES, 
    SEED
)
import torch
from collections import Counter
import math
import numpy as np
import random
import re

def process_numeric_self_consistency(pipe, dataset, template_name, args, sample_indices):
    """
    Process numeric datasets (e.g., GSM8K, DROP) using self-consistency with efficient batching.
    Uses standard majority voting for self-consistency like in multiple_choice_processor.
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
                    example = dataset[sample_idx]
                    question = example[DATASET_CONFIGS[args.dataset]["question_key"]]
                    
                    if args.dataset == "drop":
                        answers_spans = example[DATASET_CONFIGS[args.dataset]["answer_key"]]
                        if isinstance(answers_spans, dict) and 'spans' in answers_spans and len(answers_spans['spans']) > 0:
                            raw_gold_answer = answers_spans['spans'][0]
                        else:
                            raw_gold_answer = str(answers_spans)
                    else:
                        raw_gold_answer = str(example[DATASET_CONFIGS[args.dataset]["answer_key"]]).strip()

                    if args.dataset == "drop":
                        gold_answer = raw_gold_answer.strip()
                    else:
                        gold_answer = extract_numeric_answer(raw_gold_answer)

                    passage = None
                    if args.dataset == "drop" and "passage" in example:
                        passage = example["passage"]

                    options = None
                    formatted_prompt = format_prompt(template_name, args.dataset, question, options, passage)
                    
                    batch_results.append({
                        "sample_idx": sample_idx,
                        "question": question,
                        "passage": passage if args.dataset == "drop" else "",
                        "prompt": formatted_prompt,
                        "gold_answer": gold_answer,
                        "sc_answers": [],
                        "sc_texts": [],  # Store all generated texts
                    })
                except Exception as e:
                    if args.debug:
                        print(f"Error preparing sample {sample_idx}: {e}")
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
                    "max_new_tokens": MAX_NEW_TOKENS if template_name != "direct" else 5,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "do_sample": True,
                    "pad_token_id": pipe.tokenizer.eos_token_id,
                    "num_return_sequences": paths_in_current_batch,
                    "output_scores": True,
                    "return_dict_in_generate": True,
                }
                
                try:
                    with torch.no_grad():
                        generation_outputs = pipe.model.generate(
                            **tokenized_inputs,
                            **generation_kwargs
                        )
                        
                        outputs = generation_outputs.sequences
                        scores = generation_outputs.scores

                        # Calculate confidence scores from logits
                        prompt_len = tokenized_inputs['input_ids'].shape[1]
                        generated_tokens = outputs[:, prompt_len:]
                        
                        non_pad_mask = (generated_tokens != pipe.tokenizer.pad_token_id).float()
                        generated_lengths = non_pad_mask.sum(dim=1)

                        sequence_log_probs = torch.zeros(outputs.shape[0], device=pipe.model.device)

                        # Handle the case where scores length might not match generated tokens
                        max_gen_len = min(len(scores), generated_tokens.shape[1])

                        for i in range(max_gen_len):
                            score_t = scores[i]
                            
                            # Sanitize scores to prevent NaN/inf issues
                            safe_scores = torch.nan_to_num(score_t)
                            log_probs_t = torch.nn.functional.log_softmax(safe_scores, dim=-1)
                            
                            token_id = generated_tokens[:, i].unsqueeze(-1)
                            
                            # Clamp token_id to be within the vocabulary size to prevent gather errors
                            vocab_size = log_probs_t.shape[-1]
                            clamped_token_id = torch.clamp(token_id, 0, vocab_size - 1)
                            
                            token_log_prob = torch.gather(log_probs_t, 1, clamped_token_id).squeeze(-1)
                            
                            # Ensure we only add log_probs for non-pad tokens
                            sequence_log_probs += token_log_prob * non_pad_mask[:, i]

                        # Avoid division by zero for sequences with no generated tokens
                        safe_lengths = torch.max(generated_lengths, torch.ones_like(generated_lengths))
                        avg_log_probs = sequence_log_probs / safe_lengths
                        confidences = torch.exp(avg_log_probs)

                        # Replace any remaining NaN confidences with 0
                        confidences = torch.nan_to_num(confidences, nan=0.0)
                        
                        for sample_idx, result in enumerate(batch_results):
                            # Correctly get the prompt text by decoding the input tokens
                            prompt_text = pipe.tokenizer.decode(tokenized_inputs['input_ids'][sample_idx], skip_special_tokens=True)
                            
                            for path_idx in range(paths_in_current_batch):
                                output_idx = sample_idx * paths_in_current_batch + path_idx
                                
                                if output_idx >= len(outputs):
                                    break
                                    
                                output_seq = outputs[output_idx]
                                generated_text = pipe.tokenizer.decode(output_seq, skip_special_tokens=True)
                                
                                # Use the decoded prompt text to reliably extract the model's response
                                if generated_text.startswith(prompt_text):
                                    model_response = generated_text[len(prompt_text):].strip()
                                else:
                                    # Fallback for safety, though should be rare
                                    model_response = generated_text[len(result["prompt"]):].strip()
                                
                                confidence = confidences[output_idx].item()

                                if args.debug and path_idx < 3:  # Only print first few for brevity
                                    print(f"Debug: sample {result['sample_idx']}, path {path_idx}, confidence: {confidence:.6f}")

                                if args.dataset == "drop":
                                    extracted_answer = extract_drop_answer(model_response)
                                    if extracted_answer is not None:
                                        result["sc_answers"].append((extracted_answer, confidence))
                                        result["sc_texts"].append(model_response)
                                else:
                                    numeric_extracted = extract_numeric_answer(model_response)
                                    if numeric_extracted is not None:
                                        result["sc_answers"].append((str(int(numeric_extracted)), confidence))
                                        result["sc_texts"].append(model_response)
                                
                                if path_idx == 0 and "generated_text" not in result:
                                    result["generated_text"] = model_response
                
                except Exception as batch_error:
                    if args.debug:
                        print(f"Error in batch generation: {batch_error}")
            
            for result in batch_results:
                if result["sc_answers"]:
                    # Confidence-weighted self-consistency
                    answer_confidences = {}
                    for answer, confidence in result["sc_answers"]:
                        answer_confidences.setdefault(answer, 0)
                        answer_confidences[answer] += confidence

                    if args.debug:
                        print(f"Debug: answer_confidences for sample {result['sample_idx']}: {answer_confidences}")

                    if not answer_confidences:
                        if args.debug:
                            print(f"Sample {result['sample_idx']} - No valid answers to weigh.")
                        continue

                    pred_answer = max(answer_confidences, key=answer_confidences.get)
                    
                    total_confidence = sum(answer_confidences.values())
                    confidence_value = answer_confidences[pred_answer] / total_confidence if total_confidence > 0 else 0
                    
                    if args.debug:
                        print(f"Debug: total_confidence={total_confidence:.6f}, pred_answer_confidence={answer_confidences[pred_answer]:.6f}, final_confidence={confidence_value:.6f}")
                    
                    # For DROP dataset, normalize answers before comparison
                    if args.dataset == "drop" and pred_answer is not None and result["gold_answer"] is not None:
                        # Normalize answers by removing trailing punctuation and normalizing whitespace
                        normalized_pred = pred_answer.strip().rstrip('.,:;!?').strip().lower()
                        normalized_gold = result["gold_answer"].strip().rstrip('.,:;!?').strip().lower()
                        
                        # Check for exact match after normalization
                        is_correct = normalized_pred == normalized_gold
                        
                        if not is_correct:
                            # Try comparing numbers
                            pred_nums = re.findall(r'[-+]?\d*\.\d+|\d+', normalized_pred)
                            gold_nums = re.findall(r'[-+]?\d*\.\d+|\d+', normalized_gold)
                            if pred_nums and gold_nums and pred_nums == gold_nums:
                                is_correct = True
                            # Try substring matching
                            elif normalized_pred in normalized_gold or normalized_gold in normalized_pred:
                                is_correct = True
                    else:
                        is_correct = pred_answer == result["gold_answer"]
                    
                    if is_correct:
                        correct += 1
                    
                    total += 1
                    
                    sc_answers_only = [ans for ans, conf in result["sc_answers"]]
                    sc_confidences = [conf for ans, conf in result["sc_answers"]]
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
                            for (ans, conf), txt in zip(
                                result["sc_answers"], 
                                result["sc_texts"]
                            )
                        ],
                        "sc_answers": sc_answers_only,
                        "sc_texts": result["sc_texts"],
                        "sc_confidences": sc_confidences,
                    })
                    
                    if args.debug:
                        print(f"Sample {result['sample_idx']} - Predicted: {pred_answer}, Gold: {result['gold_answer']}, "
                              f"Correct: {is_correct}, SC Confidence: {confidence_value:.2f}, "
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

def process_numeric_batch(pipe, dataset, template_name, args, batch_size, max_samples, sample_indices=None):
    """
    Process numeric datasets (e.g., GSM8K, DROP) using normal generation.
    Enhanced for better performance with efficient batching.
    If sample_indices is provided, it will be used instead of generating sequential indices from 0 to max_samples.
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

    if sample_indices is None:
        sample_indices = list(range(max_samples))
    else:
        max_samples = len(sample_indices)

    # Process in batches with progress bar
    with tqdm(total=len(sample_indices), desc="Processing samples") as pbar:
        for i in range(0, len(sample_indices), batch_size):
            current_batch = sample_indices[i:i+batch_size]
            batch_questions = []
            batch_gold_answers = []
            batch_examples = []
            batch_prompts = []
            batch_passages = []
            
            for idx in current_batch:
                try:
                    example = dataset[idx]
                    batch_examples.append(example)

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

                    if args.dataset == "drop":
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
                        print(f"Error processing example at index {idx}: {str(e)}")
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
                "max_new_tokens": MAX_NEW_TOKENS if template_name != "direct" else 5,
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
                    
                    for idx, (prompt, gold_answer, question) in enumerate(zip(batch_prompts, batch_gold_answers, batch_questions)):
                        output_idx = idx * NUM_RETURN_SEQUENCES
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
                            normalized_pred = pred_answer_str.strip().rstrip('.,:;!?').strip().lower()
                            normalized_gold = gold_answer.strip().rstrip('.,:;!?').strip().lower()
                            
                            # Check for exact match after normalization
                            is_correct = normalized_pred == normalized_gold
                            
                            if not is_correct:
                                # Try comparing numbers
                                pred_nums = re.findall(r'[-+]?\d*\.\d+|\d+', normalized_pred)
                                gold_nums = re.findall(r'[-+]?\d*\.\d+|\d+', normalized_gold)
                                if pred_nums and gold_nums and pred_nums == gold_nums:
                                    is_correct = True
                                # Try substring matching
                                elif normalized_pred in normalized_gold or normalized_gold in normalized_pred:
                                    is_correct = True
                        else:
                            is_correct = pred_answer_str == gold_answer
                        
                        if is_correct:
                            correct += 1
                        
                        passage = batch_passages[idx] if args.dataset == "drop" else ""
                        result = {
                            "sample_index": i + idx,
                            "question": batch_questions[idx],
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
                            print(f"Example {i + idx}:")
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
                pbar.update(len(current_batch))
            
            accuracy = correct / total if total > 0 else 0
            pbar.set_postfix({"Accuracy": f"{accuracy:.2%}", "Correct": f"{correct}/{total}"})

    return correct, total, results 