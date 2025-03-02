from tqdm import tqdm
from .answer_extraction import extract_numeric_answer, extract_gold_gsm8k_answer
from .prompts import format_prompt
from .config import DATASET_CONFIGS, MIN_NEW_TOKENS, MAX_NEW_TOKENS, TEMPERATURE, SELF_CONSISTENCY_PATHS, TOP_P, TOP_K, DO_SAMPLE, NUM_RETURN_SEQUENCES

def process_numeric_self_consistency(pipe, dataset, template_name, args, sample_indices):
    """
    Process numeric datasets (e.g., GSM8K, DROP) using self-consistency.
    For each sample, multiple generation paths are taken followed by numeric answer extraction.
    """
    correct = 0
    total = 0
    results = []

    for idx in tqdm(sample_indices, desc=f"Processing {template_name} (Self-Consistency Numeric)"):
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

            sc_answers = []
            # Generate multiple answers for self-consistency.
            for _ in range(SELF_CONSISTENCY_PATHS):
                try:
                    output = pipe(
                        formatted_prompt,
                        min_new_tokens=MIN_NEW_TOKENS,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        top_k=TOP_K,
                        do_sample=DO_SAMPLE,
                        num_return_sequences=1,
                        pad_token_id=pipe.tokenizer.eos_token_id
                    )
                    if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
                        generated_text = output[0].get('generated_text', '')
                    else:
                        generated_text = output.get("generated_text", "") if isinstance(output, dict) else str(output)
                    
                    if generated_text:
                        model_response = generated_text[len(formatted_prompt):].strip()
                        numeric_extracted = extract_numeric_answer(model_response)
                        if numeric_extracted is not None:
                            sc_answers.append(str(int(numeric_extracted)))
                except Exception as inner_e:
                    if args.debug:
                        print(f"Error during self consistency generation for sample index {idx}: {inner_e}")
                    continue

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
                print(f"Self-consistency numeric answers: {sc_answers}")
                print(f"Predicted numeric answer: {pred_answer}")
                print(f"Gold numeric answer: {gold_answer}")
                print(f"Correct: {is_correct}")

        except Exception as e:
            if args.debug:
                print(f"Error processing sample index {idx}: {str(e)}")
    return correct, total, results

def process_numeric_batch(pipe, dataset, template_name, args, batch_size, max_samples):
    """
    Process numeric datasets in batch mode (i.e., single-path generation).
    Processes a batch of samples at a time and then extracts numeric answers.
    """
    correct = 0
    total = 0
    results = []
    
    for start_idx in tqdm(range(0, max_samples, batch_size), desc=f"Processing {template_name} in batches"):
        batch_prompts = []
        batch_examples = []
        batch_correct = 0 
        for idx in range(start_idx, min(start_idx + batch_size, max_samples)):
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
                print(f"Error in batch processing for samples {start_idx} to {min(start_idx + batch_size, max_samples)-1}: {str(e)}")
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
                pred_answer = extract_numeric_answer(model_response)
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