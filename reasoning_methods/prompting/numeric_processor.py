from tqdm import tqdm
from .answer_extraction import extract_numeric_answer, extract_gold_gsm8k_answer
from .prompts import format_prompt
from .config import DATASET_CONFIGS, MIN_NEW_TOKENS, MAX_NEW_TOKENS, TEMPERATURE, SELF_CONSISTENCY_PATHS, TOP_P, TOP_K, DO_SAMPLE, NUM_RETURN_SEQUENCES
import torch

def process_numeric_self_consistency(pipe, dataset, template_name, args, sample_indices):
    """
    Process numeric datasets (e.g., GSM8K, DROP) using self-consistency with proper batching.
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
        for idx in tqdm(batch_indices, desc=f"Processing batch {i//args.batch_size + 1}/{(len(sample_indices) + args.batch_size - 1)//args.batch_size} (Self-Consistency Numeric)"):
            try:
                example = dataset[idx]
                question = example[DATASET_CONFIGS[args.dataset]["question_key"]]

                # Handle DROP dataset specially.
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
                else:
                    gold_answer = extract_numeric_answer(raw_gold_answer)

                passage = None
                if args.dataset == "drop" and "passage" in example:
                    passage = example["passage"]

                options = None
                formatted_prompt = format_prompt(template_name, args.dataset, question, options, passage)

                # Create inputs for the model
                inputs = pipe.tokenizer(formatted_prompt, return_tensors="pt", padding=True)
                inputs = {k: v.to(pipe.model.device) for k, v in inputs.items()}
                
                # Set generation parameters for multiple sequences
                generation_kwargs = {
                    "min_new_tokens": MIN_NEW_TOKENS,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "do_sample": DO_SAMPLE,
                    "num_return_sequences": SELF_CONSISTENCY_PATHS,
                    "pad_token_id": pipe.tokenizer.eos_token_id
                }
                
                try:
                    # Generate all self-consistency paths at once
                    with torch.no_grad():
                        # Expand inputs for multiple sequences
                        expanded_inputs = {
                            k: v.repeat(SELF_CONSISTENCY_PATHS, 1) if v.dim() == 2 else v
                            for k, v in inputs.items()
                        }
                        
                        outputs = pipe.model.generate(**expanded_inputs, **generation_kwargs)
                        
                        sc_answers = []
                        for i in range(SELF_CONSISTENCY_PATHS):
                            output_seq = outputs[i, :]
                            generated_text = pipe.tokenizer.decode(output_seq, skip_special_tokens=True)
                            
                            if generated_text:
                                model_response = generated_text[len(formatted_prompt):].strip()
                                numeric_extracted = extract_numeric_answer(model_response)
                                if numeric_extracted is not None:
                                    sc_answers.append(str(int(numeric_extracted)))
                
                except Exception as inner_e:
                    if args.debug:
                        print(f"Error during self consistency generation for sample index {idx}: {inner_e}")
                    sc_answers = []

                if sc_answers:
                    from collections import Counter
                    counts = Counter(sc_answers)
                    pred_answer = counts.most_common(1)[0][0]
                else:
                    pred_answer = None

                is_correct = False
                if pred_answer is not None and gold_answer is not None:
                    try:
                        pred_value = float(pred_answer)
                        gold_value = float(gold_answer)
                        is_correct = abs(pred_value - gold_value) < 1e-7
                    except Exception as e:
                        if args.debug:
                            print(f"Error comparing numeric answers for sample index {idx}: {e}")
                
                if is_correct:
                    correct += 1
                total += 1

                result = {
                    "sample_index": idx,
                    "question": question,
                    "prompt": formatted_prompt,
                    "generated_text": f"Self-consistency answers: {sc_answers}",
                    "pred_answer": pred_answer,
                    "gold_answer": gold_answer,
                    "is_correct": is_correct
                }
                
                batch_results.append(result)
                results.append(result)

                if args.debug:
                    print(f"\nSample index {idx}:")
                    print(f"Prompt: {formatted_prompt}")
                    print(f"Self-consistency numeric answers: {sc_answers}")
                    print(f"Predicted numeric answer: {pred_answer}")
                    print(f"Gold numeric answer: {gold_answer}")
                    print(f"Correct: {is_correct}")

            except Exception as e:
                if args.debug:
                    print(f"Error processing sample index {idx}: {str(e)}")
        
        # Print batch summary if debug is enabled
        if args.debug and batch_results:
            batch_correct = sum(1 for r in batch_results if r["is_correct"])
            batch_total = len(batch_results)
            batch_acc = batch_correct / batch_total if batch_total else 0
            print(f"\nBatch {i//args.batch_size + 1} Accuracy: {batch_acc:.2%} ({batch_correct}/{batch_total})")
    
    return correct, total, results

def process_numeric_batch(pipe, dataset, template_name, args, batch_size, max_samples):
    """
    Process numeric datasets in batch mode (i.e., single-path generation).
    Each batch processes a set of samples in parallel.
    """
    correct = 0
    total = 0
    results = []
    
    # Process in batches
    for i in range(0, max_samples, args.batch_size):
        batch_indices = list(range(i, min(i + args.batch_size, max_samples)))
        batch_prompts = []
        batch_examples = []
        batch_correct = 0 
        
        # Prepare all prompts for the batch
        for idx in tqdm(batch_indices, desc=f"Preparing batch {i//args.batch_size + 1}/{(max_samples + args.batch_size - 1)//args.batch_size}"):
            try:
                example = dataset[idx]
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
                else:
                    gold_answer = extract_numeric_answer(raw_gold_answer)
                
                if gold_answer is not None:
                    gold_answer = str(int(gold_answer))
                
                passage = None
                if args.dataset == "drop" and "passage" in example:
                    passage = example["passage"]
                    
                options = None
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

        if not batch_prompts:
            continue
        
        # Process the batch of prompts in parallel using the model directly
        try:
            # Tokenize all prompts in the batch
            inputs = pipe.tokenizer(batch_prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(pipe.model.device) for k, v in inputs.items()}
            
            # Set generation parameters
            generation_kwargs = {
                "min_new_tokens": MIN_NEW_TOKENS,
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "do_sample": DO_SAMPLE,
                "pad_token_id": pipe.tokenizer.eos_token_id
            }
            
            # Generate outputs for all prompts in the batch
            with torch.no_grad():
                outputs = pipe.model.generate(**inputs, **generation_kwargs)
                
                # Decode all outputs
                generated_texts = pipe.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        except Exception as e:
            if args.debug:
                print(f"Error in batch processing for samples {i} to {min(i + args.batch_size, max_samples)-1}: {str(e)}")
            continue

        # Process the results for each example in the batch
        for j, generated_text in enumerate(tqdm(generated_texts, desc=f"Processing results for batch {i//args.batch_size + 1}/{(max_samples + args.batch_size - 1)//args.batch_size}")):
            if j >= len(batch_examples):
                continue
                
            idx = batch_examples[j]["sample_index"]
            formatted_prompt = batch_examples[j]["prompt"]

            if generated_text:
                # Extract the model response by removing the prompt
                model_response = generated_text[len(formatted_prompt):].strip()
                pred_answer = extract_numeric_answer(model_response)
                
                is_correct = False
                if pred_answer is not None and batch_examples[j]["gold_answer"] is not None:
                    try:
                        pred_num = float(pred_answer.replace(',', ''))
                        gold_num = float(batch_examples[j]["gold_answer"])
                        is_correct = abs(pred_num - gold_num) < 1e-7
                    except Exception as e:
                        if args.debug:
                            print(f"Error comparing numeric answers for sample index {idx}: {e}")
                
                result = {
                    "sample_index": idx,
                    "question": batch_examples[j]["question"],
                    "prompt": formatted_prompt,
                    "generated_text": model_response,
                    "pred_answer": pred_answer,
                    "gold_answer": batch_examples[j]["gold_answer"],
                    "is_correct": is_correct
                }
                
                results.append(result)
                
                if is_correct:
                    correct += 1
                    batch_correct += 1
                total += 1
                
                if args.debug:
                    print(f"\nBatch sample index: {idx}")
                    print(f"Prompt: {formatted_prompt}")
                    print(f"Model response: {model_response}")
                    print(f"Extracted answer: {pred_answer}")
                    print(f"Gold answer: {batch_examples[j]['gold_answer']}")
                    print(f"Correct: {is_correct}")
            else:
                if args.debug:
                    print(f"No generated text found in batch output for sample index {idx}.")

        # Print batch summary if debug is enabled
        if args.debug and batch_examples:
            batch_total = len(batch_examples)
            batch_acc = batch_correct / batch_total if batch_total else 0
            print(f"\nBatch {i//args.batch_size + 1}/{(max_samples + args.batch_size - 1)//args.batch_size} Summary:")
            print(f"Batch accuracy: {batch_acc:.2%} ({batch_correct}/{batch_total})")
            print(f"Overall accuracy: {correct/total:.2%} ({correct}/{total})")
            print("-" * 50)

    return correct, total, results 