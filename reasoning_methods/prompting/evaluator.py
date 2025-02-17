from tqdm import tqdm
from collections import Counter
from .answer_extraction import get_answer_extractor, extract_numeric_answer
from .prompts import format_prompt
from .config import MIN_NEW_TOKENS, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, TOP_K, DO_SAMPLE, NUM_RETURN_SEQUENCES, SELF_CONSISTENCY_PATHS, DATASET_CONFIGS
import torch  # Make sure torch is imported


def get_mmlu_pred_answer(prompt, options, model, tokenizer):
    """
    Given a prompt and a list of candidate options, compute the log likelihood
    of each candidate answer when appended (after an "Answer:" marker) to the prompt.
    The candidate with the highest log likelihood is returned, converted to
    its numeric representation (mapping A->"0", B->"1", etc.).
    """
    # Ensure the prompt ends with an "Answer:" line.
    if not prompt.strip().endswith("Answer:"):
        base_context = prompt.strip() + "\nAnswer:"
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

    # Select the candidate with the highest log likelihood.
    best_letter = max(scores, key=scores.get)
    # Convert candidate letter to a numeric string.
    mapping = {"A": "0", "B": "1", "C": "2", "D": "3"}
    return mapping.get(best_letter, best_letter)


def process_dataset_batch(pipe, dataset, template_name, args, batch_size):
    """Process dataset using batched inference with HF Dataset API.

    For multiple-choice benchmarks (e.g. RACE, ARC, MMLU, AGIEVAL) we no longer generate
    any freeform text. Instead, we use the model's log likelihood over candidate tokens.
    For numeric tasks (GSM8K, DROP) we keep the generation and extraction pipeline.
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
    numeric_datasets = ["gsm8k", "drop"]
    multiple_choice_datasets = ["race", "arc", "mmlu", "agieval"]

    if args.dataset in multiple_choice_datasets:
        # --- MULTIPLE CHOICE: Log-Likelihood Candidate Scoring (No Text Generation) ---
        for idx in tqdm(sample_indices, desc=f"Processing {template_name} (Log Likelihood)"):
            try:
                example = dataset[idx]
                question = example[DATASET_CONFIGS[args.dataset]["question_key"]]
                gold_answer = str(example[DATASET_CONFIGS[args.dataset]["answer_key"]]).strip()

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
                        # Fallback: use individual choice fields if necessary.
                        options = [example.get(f"choice_{i}", "") for i in range(4)]
                elif args.dataset == "agieval":
                    options = example.get("options", [])

                formatted_prompt = format_prompt(template_name, args.dataset, question, options, passage)
                
                # Use log-likelihood scoring to choose the candidate answer.
                pred_answer = get_mmlu_pred_answer(formatted_prompt, options, pipe.model, pipe.tokenizer) if options else None

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

        if args.debug:
            overall_acc = correct / total if total else 0
            print(f"\nOverall accuracy (Multiple Choice): {overall_acc:.2%}")
        return correct, total, results

    else:
        # --- NUMERIC TASKS (gsm8k, drop): Use Generation and Extraction Pipeline ---
        # The existing logic for self-consistency and single-path processing is kept unchanged.
        if args.self_consistency:
            # [Existing self-consistency branch for numeric tasks]
            for start_idx in tqdm(range(0, max_samples, batch_size),
                                  desc=f"Processing {template_name} in batches (self consistency)"):
                batch_meta = []
                replicated_prompts = []
                for idx in range(start_idx, min(start_idx + batch_size, max_samples)):
                    try:
                        example = dataset[idx]
                        question = example[DATASET_CONFIGS[args.dataset]["question_key"]]
                        raw_gold_answer = str(example[DATASET_CONFIGS[args.dataset]["answer_key"]]).strip()
                        
                        # Extract numeric gold answer.
                        gold_answer = extract_numeric_answer(raw_gold_answer)
                        if gold_answer is not None:
                            gold_answer = str(int(gold_answer))
                        
                        passage = None
                        options = None
                        if args.dataset == "gsm8k" or args.dataset == "drop":
                            # For numeric tasks, options typically are not used.
                            pass
                        formatted_prompt = format_prompt(template_name, args.dataset, question, options, passage)
                        batch_meta.append({
                            "sample_index": idx,
                            "question": question,
                            "gold_answer": gold_answer,
                            "prompt": formatted_prompt,
                            "options": options
                        })
                        # Replicate each prompt for the self-consistency paths.
                        replicated_prompts.extend([formatted_prompt] * SELF_CONSISTENCY_PATHS)
                    except Exception as e:
                        if args.debug:
                            print(f"Error preparing self-consistency batch for sample index {idx}: {str(e)}")
                        continue

                if not replicated_prompts:
                    continue

                try:
                    outputs = pipe(
                        replicated_prompts,
                        min_new_tokens=MIN_NEW_TOKENS,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        top_k=TOP_K,
                        do_sample=True,
                        num_return_sequences=1,
                        pad_token_id=pipe.tokenizer.eos_token_id
                    )
                except Exception as e:
                    if args.debug:
                        print(f"Error in batch self-consistency generation for samples {start_idx} to {min(start_idx + batch_size, max_samples)-1}: {str(e)}")
                    continue

                for i, meta in enumerate(batch_meta):
                    start_pos = i * SELF_CONSISTENCY_PATHS
                    end_pos = start_pos + SELF_CONSISTENCY_PATHS
                    sample_outputs = outputs[start_pos:end_pos]

                    answers = []
                    model_responses = []

                    for output in sample_outputs:
                        try:
                            if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
                                generated_text = output[0].get('generated_text', '')
                            else:
                                generated_text = output.get("generated_text", "") if isinstance(output, dict) else str(output)

                            if generated_text:
                                model_response = generated_text[len(meta["prompt"]):].strip()
                                model_responses.append(model_response)
                                # Using the extractor for numeric answers.
                                answer = extract_numeric_answer(generated_text)
                                if answer is not None:
                                    answers.append(str(int(answer)))
                        except Exception as e:
                            if args.debug:
                                print(f"Error processing self-consistency output for sample index {meta['sample_index']}: {str(e)}")
                            continue

                    # For numeric tasks, use majority vote over the extracted answers.
                    pred_answer = None
                    if answers:
                        counts = Counter(answers)
                        max_count = max(counts.values())
                        candidates = [k for k, v in counts.items() if v == max_count]
                        pred_answer = candidates[0] if candidates else None

                    is_correct = False
                    if pred_answer is not None and meta["gold_answer"] is not None:
                        try:
                            pred_num = float(pred_answer)
                            gold_num = float(meta["gold_answer"])
                            is_correct = abs(pred_num - gold_num) < 1e-7
                        except Exception as e:
                            if args.debug:
                                print(f"Error comparing numeric answers for sample index {meta['sample_index']}: {e}")
                    if is_correct:
                        correct += 1
                    total += 1

                    model_response_text = "\n".join(model_responses)
                    results.append({
                        "sample_index": meta["sample_index"],
                        "prompt": meta["prompt"],
                        "generated_text": model_response_text,
                        "pred_answer": pred_answer,
                        "gold_answer": meta["gold_answer"],
                        "is_correct": is_correct
                    })
                    if args.debug:
                        print(f"\nResult for sample index {meta['sample_index']}: {'Correct' if is_correct else 'Incorrect'}")
                if args.debug:
                    batch_results = results[-len(batch_meta):]
                    print(f"\nBatch {start_idx//batch_size + 1} Summary (Self-Consistency):")
                    print(f"Batch accuracy: {sum(1 for r in batch_results if r['is_correct'])/len(batch_results):.2%}")
                    print(f"Overall accuracy: {correct/total:.2%}")
                    print("-" * 50)
        else:
            # --- NUMERIC TASKS Single-Path Branch ---
            for start_idx in tqdm(range(0, max_samples, batch_size), desc=f"Processing {template_name} in batches"):
                batch_prompts = []
                batch_examples = []
                batch_correct = 0  
                for idx in range(start_idx, min(start_idx + batch_size, max_samples)):
                    try:
                        example = dataset[idx]
                        question = example[DATASET_CONFIGS[args.dataset]["question_key"]]
                        raw_gold_answer = str(example[DATASET_CONFIGS[args.dataset]["answer_key"]]).strip()

                        gold_answer = extract_numeric_answer(raw_gold_answer)
                        if gold_answer is not None:
                            gold_answer = str(int(gold_answer))

                        options = None
                        passage = None
                        if args.dataset == "gsm8k" or args.dataset == "drop":
                            pass
                        formatted_prompt = format_prompt(template_name, args.dataset, question, options, passage)
                        batch_examples.append({
                            "sample_index": idx,
                            "question": question,
                            "gold_answer": gold_answer,
                            "prompt": formatted_prompt,
                            "options": options
                        })
                        batch_prompts.append(formatted_prompt)
                    except Exception as e:
                        if args.debug:
                            print(f"Error preparing batch for sample index {idx}: {str(e)}")
                        continue

                try:
                    outputs = pipe(
                        batch_prompts,
                        min_new_tokens=MIN_NEW_TOKENS,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        top_k=TOP_K,
                        do_sample=DO_SAMPLE,
                        num_return_sequences=NUM_RETURN_SEQUENCES,
                        pad_token_id=pipe.tokenizer.eos_token_id
                    )
                except Exception as e:
                    if args.debug:
                        print(f"Error in batch single-path processing for samples {start_idx} to {min(start_idx + batch_size, max_samples)-1}: {str(e)}")
                    continue

                for i, output in enumerate(outputs):
                    idx = batch_examples[i]["sample_index"]
                    formatted_prompt = batch_examples[i]["prompt"]
                    if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
                        generated_text = output[0].get('generated_text', '')
                    else:
                        generated_text = output.get("generated_text", "") if isinstance(output, dict) else str(output)

                    if generated_text:
                        model_response = generated_text[len(formatted_prompt):].strip()
                        pred_answer = extract_numeric_answer(formatted_prompt + model_response)
                        if args.debug:
                            print(f"\nBatch sample index: {idx}")
                            print(f"Prompt: {formatted_prompt}")
                            print(f"Model response: {model_response}")
                            print(f"Extracted answer: {pred_answer}")
                            print(f"Gold answer: {batch_examples[i]['gold_answer']}")
                            print(f"Correct: {pred_answer == batch_examples[i]['gold_answer']}")
                        is_correct = False
                        if pred_answer is not None and batch_examples[i]["gold_answer"] is not None:
                            try:
                                pred_num = float(pred_answer.replace(',', ''))
                                gold_num = float(batch_examples[i]["gold_answer"])
                                is_correct = abs(pred_num - gold_num) < 1e-7
                            except Exception as e:
                                if args.debug:
                                    print(f"Error comparing numeric answers for sample index {idx}: {e}")
                        results.append({
                            "sample_index": idx,
                            "prompt": formatted_prompt,
                            "generated_text": model_response,
                            "pred_answer": pred_answer,
                            "gold_answer": batch_examples[i]["gold_answer"],
                            "is_correct": is_correct
                        })
                        if is_correct:
                            correct += 1
                            batch_correct += 1
                        total += 1
                    else:
                        if args.debug:
                            print(f"No generated text found in batch output for sample index {idx}.")

                if args.debug:
                    print(f"\nBatch {start_idx//batch_size + 1} Summary (Single-Path):")
                    print(f"Batch accuracy: {batch_correct/len(batch_examples):.2%}")
                    print(f"Overall accuracy: {correct/total:.2%}")
                    print(f"Total correct so far: {correct}/{total}")
                    print("-" * 50)

        return correct, total, results 