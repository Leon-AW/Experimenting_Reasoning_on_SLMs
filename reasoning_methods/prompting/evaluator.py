from tqdm import tqdm
from collections import Counter
from .answer_extraction import get_answer_extractor, extract_numeric_answer
from .prompts import format_prompt
from .config import MIN_NEW_TOKENS, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, TOP_K, DO_SAMPLE, NUM_RETURN_SEQUENCES, SELF_CONSISTENCY_PATHS, DATASET_CONFIGS


def process_dataset_batch(pipe, dataset, template_name, args, batch_size):
    """Process dataset using batched inference with HF Dataset API"""
    if args.debug:
        print("Debug mode is ON in process_dataset_batch")

    correct = 0
    total = 0
    results = []
    max_samples = min(1000, len(dataset))

    if args.self_consistency:
        # Process in batches for self-consistency (multiple paths per example)
        for start_idx in tqdm(range(0, max_samples, batch_size),
                              desc=f"Processing {template_name} in batches (self consistency)"):
            batch_meta = []      # For each sample in the batch, store metadata and the original prompt.
            replicated_prompts = []  # Will store (prompt repeated SELF_CONSISTENCY_PATHS times) for each sample.

            # Prepare the batch: for each sample replicate the prompt
            for idx in range(start_idx, min(start_idx + batch_size, max_samples)):
                try:
                    example = dataset[idx]
                    question = example[DATASET_CONFIGS[args.dataset]["question_key"]]
                    raw_gold_answer = str(example[DATASET_CONFIGS[args.dataset]["answer_key"]]).strip()
                    
                    # Extract just the numeric value from the gold answer
                    if args.dataset in ["gsm8k", "drop"]:
                        gold_answer = extract_numeric_answer(raw_gold_answer)
                        if gold_answer is not None:
                            gold_answer = str(int(gold_answer))  # Convert to integer string
                    else:
                        gold_answer = raw_gold_answer

                    # Extract options and passage if available
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
                        options = [example.get(f"choice_{i}", "") for i in range(4) if example.get(f"choice_{i}")]
                    elif args.dataset == "agieval":
                        options = example.get("options", [])

                    formatted_prompt = format_prompt(template_name, args.dataset, question, options, passage)
                    # Append the same prompt SELF_CONSISTENCY_PATHS times
                    replicated_prompts.extend([formatted_prompt] * SELF_CONSISTENCY_PATHS)
                    batch_meta.append({
                        "sample_index": idx,
                        "question": question,
                        "gold_answer": gold_answer,
                        "prompt": formatted_prompt
                    })
                except Exception as e:
                    if args.debug:
                        print(f"Error preparing self-consistency batch for sample index {idx}: {str(e)}")
                    continue

            if not replicated_prompts:
                continue

            try:
                # Note: Since we already replicated each prompt, set num_return_sequences=1.
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

                if args.debug:
                    expected_count = len(batch_meta) * SELF_CONSISTENCY_PATHS
                    print(f"\nGenerated batch self-consistency outputs for sample indices {start_idx} to {min(start_idx + batch_size, max_samples)-1}")
                    print(f"Total outputs generated: {len(outputs)} (expected {expected_count})")
                    if outputs:
                        print(f"First output in batch: {outputs[0]}")
            except Exception as e:
                if args.debug:
                    print(f"Error in batch self-consistency generation for samples {start_idx} to "
                          f"{min(start_idx + batch_size, max_samples)-1}: {str(e)}")
                continue

            # Group outputs for each sample (each sample should have SELF_CONSISTENCY_PATHS outputs)
            for i, meta in enumerate(batch_meta):
                start_pos = i * SELF_CONSISTENCY_PATHS
                end_pos = start_pos + SELF_CONSISTENCY_PATHS
                sample_outputs = outputs[start_pos:end_pos]

                answers = []
                model_responses = []

                for output in sample_outputs:
                    try:
                        # Fix: Extract generated_text from the pipeline output structure
                        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
                            generated_text = output[0].get('generated_text', '')
                        else:
                            generated_text = output.get("generated_text", "") if isinstance(output, dict) else str(output)

                        if generated_text:
                            # Remove the prompt from the generated output
                            model_response = generated_text[len(meta["prompt"])].strip()
                            model_responses.append(model_response)
                            answer = get_answer_extractor(args.dataset)(generated_text)
                            if answer is not None:
                                answers.append(str(answer).upper())

                            if args.debug:
                                print(f"\nSample index {meta['sample_index']} self-consistency path generated text:")
                                print(generated_text)
                                print(f"Extracted answer: {answer}")
                    except Exception as e:
                        if args.debug:
                            print(f"Error processing self-consistency output for sample index {meta['sample_index']}: {str(e)}")
                        continue

                # Majority vote on the answers
                pred_answer = None
                if answers:
                    counts = Counter(answers)
                    max_count = max(counts.values())  # not used further, but shows max frequency
                    candidates = [k for k, v in counts.items() if v == max_count]
                    pred_answer = candidates[0] if candidates else None

                    if args.debug:
                        print(f"\nAll answers for sample index {meta['sample_index']}: {answers}")
                        print(f"Selected answer: {pred_answer}")
                        print(f"Gold answer: {meta['gold_answer']}")

                # Compare prediction with gold answer
                is_correct = False
                if pred_answer is not None and meta["gold_answer"] is not None:
                    if args.dataset in ["gsm8k", "drop"]:
                        try:
                            pred_num = float(pred_answer.replace(',', ''))
                            gold_num = float(meta["gold_answer"])
                            is_correct = abs(pred_num - gold_num) < 1e-7
                        except (ValueError, TypeError) as e:
                            if args.debug:
                                print(f"Error comparing numeric answers for sample index {meta['sample_index']}: {e}")
                                print(f"Predicted answer: {pred_answer}")
                                print(f"Gold answer: {meta['gold_answer']}")
                    else:
                        is_correct = str(pred_answer).upper() == str(meta["gold_answer"]).upper()

                if is_correct:
                    correct += 1
                total += 1

                model_response_text = "\n".join(model_responses)

                if args.debug:
                    print(f"Result for sample index {meta['sample_index']}: {'Correct' if is_correct else 'Incorrect'}\n")

                results.append({
                    "sample_index": meta["sample_index"],
                    "prompt": meta["prompt"],
                    "generated_text": model_response_text,
                    "pred_answer": pred_answer,
                    "gold_answer": meta["gold_answer"],
                    "is_correct": is_correct
                })

            # Print batch summary for self-consistency
            if args.debug:
                batch_results = results[-len(batch_meta):]
                print(f"\nBatch {start_idx//batch_size + 1} Summary (Self-Consistency):")
                print(f"Batch accuracy: {sum(1 for r in batch_results if r['is_correct'])/len(batch_results):.2%}")
                print(f"Overall accuracy: {correct/total:.2%}")
                print(f"Total correct so far: {correct}/{total}")
                print("-" * 50)

    else:
        # Single path processing in batches
        for start_idx in tqdm(range(0, max_samples, batch_size), desc=f"Processing {template_name} in batches"):
            batch_prompts = []
            batch_examples = []
            batch_correct = 0  # Add counter for correct answers in current batch
            
            for idx in range(start_idx, min(start_idx + batch_size, max_samples)):
                try:
                    example = dataset[idx]
                    question = example[DATASET_CONFIGS[args.dataset]["question_key"]]
                    raw_gold_answer = str(example[DATASET_CONFIGS[args.dataset]["answer_key"]]).strip()
                    
                    # Extract just the numeric value from the gold answer
                    if args.dataset in ["gsm8k", "drop"]:
                        gold_answer = extract_numeric_answer(raw_gold_answer)
                        if gold_answer is not None:
                            gold_answer = str(int(gold_answer))  # Convert to integer string
                    else:
                        gold_answer = raw_gold_answer

                    # Extract options and passage if available
                    options = None
                    passage = None
                    if args.dataset == "race":
                        options = example.get("options", [])
                        passage = example.get("article", "")
                    elif args.dataset == "arc":
                        choices = example.get("choices", {})
                        if isinstance(choices, dict) and "text" in choices:
                            options = choices["text"]
                        else:
                            continue
                    elif args.dataset == "mmlu":
                        options = [example.get(f"choice_{i}", "") for i in range(4) if example.get(f"choice_{i}")]
                    elif args.dataset == "agieval":
                        options = example.get("options", [])

                    formatted_prompt = format_prompt(template_name, args.dataset, question, options, passage)
                    batch_prompts.append(formatted_prompt)
                    batch_examples.append({
                        "sample_index": idx,
                        "question": question,
                        "gold_answer": gold_answer,
                        "prompt": formatted_prompt
                    })
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

                if args.debug:
                    print(f"\nGenerated batch output for sample indices {start_idx} to {min(start_idx + batch_size, max_samples)-1}")
                    print(f"Output structure: {type(outputs)}")
                    if outputs:
                        print(f"First output in batch: {outputs[0]}")
            except Exception as e:
                if args.debug:
                    print(f"Error in batch single-path processing for samples {start_idx} to {min(start_idx + batch_size, max_samples)-1}: {str(e)}")
                continue

            for i, output in enumerate(outputs):
                idx = batch_examples[i]["sample_index"]
                question = batch_examples[i]["question"]
                gold_answer = batch_examples[i]["gold_answer"]
                formatted_prompt = batch_examples[i]["prompt"]

                # Fix: Extract generated_text from the pipeline output structure
                if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
                    generated_text = output[0].get('generated_text', '')
                else:
                    generated_text = output.get("generated_text", "") if isinstance(output, dict) else str(output)

                if generated_text:
                    # Remove the prompt from the generated output by finding where it starts
                    model_response = generated_text[len(formatted_prompt):].strip()
                    
                    # Create full text by concatenating prompt and model response
                    full_text = formatted_prompt + model_response
                    
                    # Use the full text for answer extraction
                    pred_answer = get_answer_extractor(args.dataset)(full_text)

                    if args.debug:
                        print(f"\nBatch sample index: {idx}")
                        print(f"Prompt: {formatted_prompt}")
                        print(f"Model response: {model_response}")
                        print(f"Extracted answer: {pred_answer}")
                        print(f"Gold answer: {gold_answer}")
                        print(f"Correct: {pred_answer == gold_answer}")

                    is_correct = False
                    if pred_answer is not None and gold_answer is not None:
                        if args.dataset in ["gsm8k", "drop"]:
                            try:
                                pred_num = float(pred_answer.replace(',', ''))
                                gold_num = float(gold_answer)
                                is_correct = abs(pred_num - gold_num) < 1e-7
                            except (ValueError, TypeError) as e:
                                if args.debug:
                                    print(f"Error comparing numeric answers for sample index {idx}: {e}")
                                    print(f"Predicted answer: {pred_answer}")
                                    print(f"Gold answer: {gold_answer}")
                        else:
                            is_correct = str(pred_answer).upper() == str(gold_answer).upper()

                    results.append({
                        "sample_index": idx,
                        "prompt": formatted_prompt,
                        "generated_text": model_response,
                        "pred_answer": pred_answer,
                        "gold_answer": gold_answer,
                        "is_correct": is_correct
                    })

                    if is_correct:
                        correct += 1
                        batch_correct += 1  # Increment batch correct counter
                    total += 1
                else:
                    if args.debug:
                        print(f"No generated text found in batch output for sample index {idx}.")

            # Print batch accuracy after processing each batch
            if args.debug:
                print(f"\nBatch {start_idx//batch_size + 1} Summary (Single-Path):")
                print(f"Batch accuracy: {batch_correct/len(batch_examples):.2%}")
                print(f"Overall accuracy: {correct/total:.2%}")
                print(f"Total correct so far: {correct}/{total}")
                print("-" * 50)

    if args.debug:
        print(f"\nBatch Summary:")
        print(f"Correct answers in this batch: {sum(1 for r in results[-batch_size:] if r['is_correct'])}")
        print(f"Total processed in this batch: {len(results[-batch_size:])}")
        print(f"Running accuracy: {correct/total:.2%}")

    return correct, total, results 