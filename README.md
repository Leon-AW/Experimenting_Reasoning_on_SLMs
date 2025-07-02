# Robustness Testing and Comparing Reasoning Techniques for Small Language Models

This repository contains the study project "Robustness Testing and Comparing Reasoning Techniques for Small Language Models". It investigates the effectiveness of different reasoning enhancement techniques on a 1-billion parameter Llama 3.2 model, comparing its performance against a baseline 1B model, and a larger 3B parameter Llama 3.2 model.

## Project Goal

The primary goal of this project was to explore and evaluate methods to improve the reasoning capabilities of Small Language Models (SLMs). As described in the project exposé, we investigated three distinct approaches:

1.  **Prompting:** Utilizing techniques like Chain-of-Thought (CoT) and Self-Consistency on pre-trained models without further finetuning.
2.  **Multi-Stage Finetuning:** Finetuning the base model on various reasoning datasets, inspired by the Orca 2 paper.
3.  **Hybrid Method (STaR):** A hybrid approach combining finetuning and prompting, where the model is finetuned on its own generated reasoning steps (rationales).

The core research questions were:
- Can reasoning techniques substantially improve the performance of a 1B model?
- Are these improvements sufficient to match or exceed the performance of a larger 3B model?
- Which approaches are most effective?

## Prompting Strategies

The following prompt templates were used to guide the model's reasoning process.

### Simple Prompt
*   **Numeric:**
    ```
    Problem: {question}
    Solution: 
    ```
*   **Multiple Choice:**
    ```
    Question: {question}

    Options:
    {options}

    Answer:
    ```

### Chain-of-Thought (CoT)
*   **Numeric:**
    ```
    Problem: {question}

    Solve and conclude your solution with 'The final answer is: <insert your answer here>'.

    Let's think step by step: 
    ```
*   **Multiple Choice:**
    ```
    Question: {question}

    Options:
    {options}

    Let's think step by step: 
    ```

### Role-Setting Prompt
*   **Numeric:**
    ```
    User: From now on, you are an excellent math teacher and always teach your students math problems correctly. And I am one of your students.
    Assistant: That's great to hear! As your math teacher, I'll do my best to explain mathematical concepts correctly so that you can understand them easily. Finally I will conclude it with 'The final answer is: <insert your answer here>'. Feel free to ask any math problems or questions you have, and I'll be glad to assist you.
    User: {question}
    Assistant: 
    ```
*   **Multiple Choice:**
    ```
    User: From now on, you are an excellent math teacher and always teach your students math problems correctly. And I am one of your students.
    Assistant: That's great to hear! As your math teacher, I'll do my best to explain mathematical concepts correctly so that you can understand them easily. Feel free to ask any math problems or questions you have, and I'll be glad to assist you.
    User: {question}
    Options:
    {options}
    Assistant: 
    ```

### Plan-and-Solve Prompt
*   **Numeric:**
    ```
    Problem: {question}

    Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a plan. Then, let's carry out the plan, calculate intermediate results (pay attention to calculation and common sense), solve the problem step by step, and then conclude it with 'The final answer is: <insert your answer here>'.
            Let's begin: 
    ```
*   **Multiple Choice:**
    ```
    Question: {question}

    Options:
    {options}

    Let's approach this systematically:
    1. First, let's understand the question
    2. Then, analyze each option carefully
    3. Finally, choose the highest probability answer. 
            Let's begin: 
    ```

## Experimental Setup

- **Base Models:** Llama 3.2 1B and Llama 3.2 3B.
- **Benchmarks:** GSM8K, ARC, Race, MMLU, DROP, CommonsenseQA.
- **Evaluation:** Performance was compared against the Llama 3.2 1B and 3B models using a simple, direct-answering prompt.

### Models Tested

- **`1b` / `3b`**: The base Llama 3.2 models.
- **`1b-instruct`**: The instruction-finetuned version of Llama 3.2 1B from Meta.
- **`1b-sft-full` / `1b-sft-full-slimorca-100k` / `1b-sft-lora-all`**: Models finetuned on the SlimOrca dataset with different configurations.
- **`1b-sft-mixed-best`**: A 1B model finetuned on a carefully curated mix of datasets, including SlimOrca (42.5%), ARC (12.5%), CommonsenseQA (12.5%), GSM8K (10%), MetaMathQA (12.5%), and SQuAD (10%).
- **`star`**: A 1B model finetuned using the Self-Taught Reasoner (STaR) hybrid method for one iteration.

### Hyperparameters

All experiments were conducted with consistent hyperparameters to ensure fair comparisons. The only exception was the rationale generation for the STaR model, which used a higher temperature to encourage more diverse reasoning paths.

- **`MIN_NEW_TOKENS`**: 1
- **`MAX_NEW_TOKENS`**: 256 (128 for STaR rationale generation)
- **`TEMPERATURE`**: 0.5 (0.7 for STaR rationale generation)
- **`TOP_P`**: 0.9
- **`DO_SAMPLE`**: True
- **`SELF_CONSISTENCY_PATHS`**: 20
- **`SEED`**: 42

## Methodology

### Answer Extraction

Two distinct methods were used to extract answers depending on the dataset format.

#### Numeric Datasets (GSM8K, DROP)

For datasets requiring a numeric answer, two specialized functions were used. For `gsm8k`, a function called `extract_numeric_answer` was used to parse the model's generated text, while for `drop` the `extract_drop_answer` function was used. Both functions use a series of regular expressions to find the final answer, even when it's embedded in complex reasoning steps or calculations. This allows the model to "think step-by-step" and still have its final, precise answer evaluated correctly.

#### Multiple-Choice Datasets (ARC, RACE, MMLU, CommonsenseQA)

For multiple-choice questions, answers were retrieved using **Log-Likelihood**. This method calculates the probability of each possible answer choice and selects the one with the highest likelihood.

While this approach is efficient and consistent—key considerations given limited GPU availability—it has a methodological limitation: it prevents the model from generating a full step-by-step reasoning chain to arrive at its final answer. This may have limited the potential performance boost from reasoning-intensive prompting techniques on these specific datasets.

### Confidence-Weighted Self-Consistency

The Self-Consistency (SC) method used in these experiments was enhanced with a confidence-weighting mechanism. This approach, known as Confidence-Informed Self-Consistency (CISC), was introduced by Taubenfeld et al. in "Confidence Improves Self‑Consistency in LLMs" (arXiv, Feb 10, 2025).

Instead of a simple majority vote over multiple reasoning paths, CISC weighs each answer using the model's own confidence scores. This has two key benefits:
1.  **Efficiency:** It can achieve better or equivalent results with over 40% fewer reasoning paths, reducing inference costs.
2.  **Model Introspection:** It leverages the model's ability to assess the quality of its own answers, prioritizing high-confidence responses.

## Reproducing the Experiments

Each of the three core methodologies has its own dedicated directory and a detailed `README` with specific instructions for running the code. Below is a summary of how to reproduce the experiments for each approach.

### 1. Prompting

The prompting experiments are designed to be run in a comprehensive "sweep" mode, which evaluates all configured models, datasets, and templates automatically.

-   **Detailed Documentation:** [`reasoning_methods/prompting/README_PROMPTING.md`](reasoning_methods/prompting/README_PROMPTING.md)
-   **Main Script:** `reasoning_methods/prompting/main.py`

To run a full evaluation sweep across all configured settings, execute the main script as a module:
```bash
python -m reasoning_methods.prompting.main
```
To run a specific experiment, you can use command-line arguments:
```bash
# Evaluate GSM8K with the 1B model using the 'cot' template
python -m reasoning_methods.prompting.main --dataset gsm8k --model_size 1b --template cot
```
Results and debug logs are saved to the `reasoning_methods/prompting/results/` and `reasoning_methods/prompting/debug_csvs/` directories, respectively.

### 2. Multi-Stage Finetuning (Orca-Style)

This approach involves a two-step process: creating a specialized dataset and then fine-tuning the model on it.

-   **Detailed Documentation:** [`reasoning_methods/fine-tuning/README_FINETUNING.md`](reasoning_methods/fine-tuning/README_FINETUNING.md)

**Step 1: Create the Dataset**
The script processes and combines several datasets, applying the "prompt erasure" technique to create a training set.
```bash
python reasoning_methods/fine-tuning/create_mixed_dataset.py
```
This saves the dataset to `reasoning_methods/fine-tuning/mixed_finetuning_dataset`.

**Step 2: Run Supervised Fine-Tuning (SFT)**
Use the `sft.py` script to fine-tune a base model on the newly created dataset.
```bash
python reasoning_methods/fine-tuning/sft.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B" \
    --dataset_name "reasoning_methods/fine-tuning/mixed_finetuning_dataset" \
    --output_dir "./models/1b-sft-mixed-best" \
    --learning_rate 2e-5 \
    --num_train_epochs 3
```

### 3. Hybrid Method (STaR)

The Self-Taught Reasoner (STaR) implementation is modular and follows the algorithm from the original paper. It iteratively collects rationales and fine-tunes the model.

-   **Detailed Documentation:** [`reasoning_methods/hybrid/README_HYBRID.md`](reasoning_methods/hybrid/README_HYBRID.md)
-   **Main Script:** `reasoning_methods/hybrid/star_main.py`

To run the full STaR process with automatic iterations for a specific dataset:
```bash
# Run 3 iterations of STaR on the GSM8K dataset
python reasoning_methods/hybrid/star_main.py --dataset gsm8k --num_iterations 3
```
The implementation also allows for running each phase (rationale collection, fine-tuning, evaluation) independently. Generated rationales are saved in `reasoning_methods/hybrid/collected_rationales/` and the fine-tuned models are saved in `reasoning_methods/hybrid/star_models/`.

## Key Findings

The experiments revealed several key insights into how different reasoning strategies impact SLM performance.

### How did the experiments go?

Overall, the experiments were successful in demonstrating that reasoning techniques can significantly enhance the performance of a 1B parameter model. However, the effectiveness of each technique varied greatly depending on the base model's initial capabilities and the specific task.

- **Prompting:** Advanced prompting techniques like Chain-of-Thought (`cot`) were often detrimental to the performance of the base 1B model. However, when applied to the `1b-sft-mixed-best` or `1b-instruct` model, these same techniques yielded substantial improvements, suggesting that a model must first be properly finetuned for complex prompting to be effective.

- **Finetuning:** Finetuning on a general instruction dataset like SlimOrca provided only marginal gains. In contrast, the `1b-sft-mixed-best` model, which was finetuned on a mix of general and task-specific data, showed remarkable performance increases across multiple benchmarks.

- **Hybrid Method (STaR):** The `star` model, after only one iteration of self-improvement, showed a clear boost in performance on math reasoning (`gsm8k`), more than doubling the score of the base 1B model. However, its performance on other benchmarks was often lower than the base model, indicating that the reasoning skills learned for one domain did not generalize well without further training.

### Comparison to Baselines

When comparing the enhanced 1B models to the simple-prompted Llama 3.2 1B and 3B models, we found:

- All specialized 1B models significantly outperformed the **base 1B model** on their respective strong suits. For example, `1b-instruct` excelled in general reasoning, `1b-sft-mixed-best` was a strong all-rounder, and `star` showed improvements for mathematical reasoning.

- The ultimate goal was to see if a 1B model could match the **3B model**. Our results show this is indeed possible, but it requires the right approach.

### Which 1B approach could beat the 3B model?

Yes, several 1B models were able to outperform the Llama 3.2 3B model (using a simple prompt) on specific benchmarks.

- **`1b-sft-mixed-best` Model:** This finetuned model, for which we have full transparency into the training data, stands out. It beat the 3B baseline on `gsm8k` (**15.3%** vs. 12.2%) and `commonsense_qa` (**58.3%** vs. 54.4%). Using a Plan-and-Solve prompt with self-consistency, its `gsm8k` score jumped to an impressive **29.1%**, making it the best-performing model with a transparent training process on this benchmark.

- **`1b-instruct` Model:** While Meta's instruction-finetuned model achieved a high score of **51.8%** on `gsm8k` with CoT and self-consistency, the lack of transparency into its training data makes it difficult to draw fair comparisons. It's possible the model was exposed to test-like data during its proprietary finetuning process.

- **`star` Model:** The STaR model also showed significant improvement on `gsm8k`, scoring **10.1%** with a simple prompt. While this is a large jump from the base 1B model's 5.5%, it did not surpass the 3B model's score of 12.2% in this case.

## Summary of Results

This table summarizes key results and compares them against the 1B and 3B models with a simple prompt. The delta columns show the absolute performance difference.

| Dataset          | Model                | Template | Self-Consistency | Accuracy (%) | vs 1B Simple (%) | vs 3B Simple (%) |
|------------------|----------------------|----------|------------------|--------------|------------------|------------------|
| **gsm8k**        | 1B (Baseline)        | simple   | False            | **5.5**      | -                | -6.7             |
|                  | 3B (Baseline)        | simple   | False            | **12.2**     | +6.7             | -                |
|                  | 1b-instruct          | cot      | True             | 51.8         | +46.3            | +39.6            |
|                  | 1b-sft-mixed-best    | plan     | True             | 29.1         | +23.6            | +16.9            |
|                  | star                 | simple   | False            | 10.1         | +4.6             | -2.1             |
| **arc**          | 1B (Baseline)        | simple   | False            | **39.8**     | -                | -23.1            |
|                  | 3B (Baseline)        | simple   | False            | **62.9**     | +23.1            | -                |
|                  | 1b-instruct          | simple   | False            | 56.1         | +16.3            | -6.8             |
|                  | 1b-sft-mixed-best    | simple   | False            | 45.6         | +5.8             | -17.3            |
| **race**         | 1B (Baseline)        | simple   | False            | **44.0**     | -                | -24.3            |
|                  | 3B (Baseline)        | simple   | False            | **68.3**     | +24.3            | -                |
|                  | 1b-sft-mixed-best    | simple   | False            | 53.9         | +9.9             | -14.4            |
|                  | 3b                   | plan     | False            | 57.8         | +13.8            | -10.5            |
| **mmlu**         | 1B (Baseline)        | simple   | False            | **41.0**     | -                | -14.7            |
|                  | 3B (Baseline)        | simple   | False            | **55.7**     | +14.7            | -                |
|                  | 1b-sft-mixed-best    | simple   | False            | 40.2         | -0.8             | -15.5            |
|                  | 3b                   | plan     | False            | 46.4         | +5.4             | -9.3             |
| **drop**         | 1B (Baseline)        | simple   | False            | **6.7**      | -                | -7.6             |
|                  | 3B (Baseline)        | simple   | False            | **14.3**     | +7.6             | -                |
|                  | star                 | role     | False            | 12.2         | +5.5             | -2.1             |
| **commonsense_qa** | 1B (Baseline)        | simple   | False            | **42.1**     | -                | -12.3            |
|                  | 3B (Baseline)        | simple   | False            | **54.4**     | +12.3            | -                |
|                  | 1b-instruct          | plan     | False            | 56.1         | +14.0            | +1.7             |
|                  | 1b-sft-mixed-best    | plan     | True             | 58.7         | +16.6            | +4.3             |

## Additional Analysis

Based on a deeper dive into the results, here are some more specific insights.

### Best Performance by Dataset

This table shows the single best-performing method for each dataset across all tested models and configurations. It highlights that no single method is universally superior; the best approach is task-dependent.

| Dataset        | Best Method                 | Accuracy   |
|----------------|-----------------------------|------------|
| **GSM8K**      | Plan+SC (1B-SFT-MIXED-BEST) | 29.1%      |
| **ARC**        | Simple (3B)                 | 62.9%      |
| **RACE**       | Simple (3B)                 | 68.3%      |
| **MMLU**       | Simple (3B)                 | 55.7%      |
| **DROP**       | Role (1B-INSTRUCT)          | 54.7%      |
| **CommonsenseQA** | Plan+SC (1B-SFT-MIXED-BEST) | 58.7%      |

### The Impact of Model Size

Increasing model size from 1B to 3B parameters generally provides a significant performance boost. The following table shows the improvement on the `gsm8k` benchmark.

| Template   | 1B Accuracy | 3B Accuracy | Improvement   |
|------------|-------------|-------------|---------------|
| Simple     | 5.5%        | 12.2%       | +6.7pp        |
| CoT        | 4.9%        | 14.3%       | +9.4pp        |
| Plan       | 3.2%        | 10.5%       | +7.3pp        |
| Role       | 2.8%        | 7.0%        | +4.2pp        |

This underscores the challenge: a larger model often sets a high baseline that reasoning techniques on smaller models must overcome. The average improvement on `gsm8k` when scaling from 1B to 3B was **6.9 percentage points**.

### The Effect of Self-Consistency

Self-consistency (SC) is a powerful technique, but its effectiveness varies. For math-heavy tasks like `gsm8k` and `drop`, it provides a substantial boost. However, for multiple-choice reasoning tasks like `arc` and `mmlu`, it sometimes *hurts* performance on the base 1B model, suggesting it may amplify biases if the model's underlying reasoning is not already sound.

The average effect of Self-Consistency on the base 1B model was a **+0.9pp** improvement, but this hides the high variability shown below.

| Dataset   | Template   | No SC   | With SC   | SC Effect   |
|-----------|------------|---------|-----------|-------------|
| **gsm8k** | Cot        | 4.9%    | 8.8%      | **+3.9pp**  |
| **drop**  | Simple     | 6.7%    | 9.5%      | **+2.8pp**  |
| **arc**   | Simple     | 39.8%   | 37.8%     | **-2.0pp**  |
| **mmlu**  | Simple     | 41.0%   | 38.3%     | **-2.7pp**  |

## Limitations and Future Work

This study, while providing valuable insights, has several limitations that offer avenues for future research:

-   **Incomplete Robustness Testing:** The original plan included a comprehensive evaluation of robustness using metrics like `Semantic Consistency Score` and `Logical Contradiction Rate`. Due to time constraints, this analysis could not be completed. Future work should implement these metrics to provide a more rigorous assessment of how well these reasoning techniques hold up under adversarial or out-of-distribution inputs.

-   **Methodology for Multiple-Choice Tasks:** The use of log-likelihood for multiple-choice questions was an efficient method for answer extraction given resource constraints. However, this approach has a significant methodological drawback: it prevents the model from generating a full reasoning chain (Chain-of-Thought) before selecting an answer. This likely limited the potential performance gains from advanced prompting techniques on benchmarks like ARC, RACE, and MMLU. Future experiments should explore methods that allow for full rationale generation on these tasks.

-   **Limited Generalization of STaR:** The Self-Taught Reasoner (`star`) model showed promising results on `gsm8k` after just one iteration of finetuning. However, its performance did not generalize well to other domains, and in some cases, was worse than the base model. This suggests that the reasoning skills learned were highly task-specific. The original STaR paper performed several iterations of this process; due to computational and time constraints, this study was limited to a single iteration. Future work could explore multiple iterations of the STaR method and finetuning on a more diverse set of rationales to improve generalization.

-   **Computational Cost of Self-Consistency:** While self-consistency can significantly improve performance, it comes at a steep computational cost. Generating multiple reasoning paths (e.g., 20 paths in this study) means that inference takes 20 times longer. This trade-off is critical: a smaller model with self-consistency might achieve higher accuracy but can be as computationally expensive as a much larger model running a single-path inference. The choice between a larger model and a smaller model with self-consistency depends heavily on the available computational budget and the specific performance gains observed for a given task.

-   **Transparency of Pre-trained Models:** The `1b-instruct` model from Meta performed exceptionally well, but it is effectively a "black box." The lack of transparency into its finetuning dataset makes it difficult to draw definitive conclusions or fair comparisons. It is possible the model was exposed to data similar to the evaluation benchmarks, which would inflate its performance.

-   **Resource Constraints:** The scope of this project was constrained by time and GPU availability. This limited the number of self-consistency paths, the number of STaR iterations, and the overall breadth of experimentation. More extensive hyperparameter tuning and longer training runs could yield further improvements.

## Conclusion: Answering the Research Questions

The results of this project provide clear answers to the initial research questions:

1.  **Are performance improvements substantial?** Yes, the improvements are not just marginal but can lead to performance boosts of 2-5x on specific tasks (e.g., `gsm8k`), elevating the 1B model into a much higher capability class.

2.  **Is it just reasoning, or are there other factors?** The underlying model is critical. The `1b-instruct` model served as a much better foundation for prompting than the base model. Furthermore, the finetuning data composition was a decisive factor, as shown by the success of the `1b-sft-mixed-best` model. Therefore, performance stems from a combination of the base model's alignment, the reasoning technique applied, and the data used for finetuning.

3.  **Can these approaches work on a 1B model?** Absolutely. This project confirms that reasoning enhancement is not exclusive to large models and can be highly effective for models with as few as one billion parameters.

4.  **Can techniques be combined?** The success of the `1b-sft-mixed-best` model, which was later enhanced with `cot` prompting, shows that combining finetuning with sophisticated prompting is a powerful strategy.

In conclusion, this study demonstrates that by intelligently applying finetuning and prompting techniques, a small 1B parameter language model can achieve and even surpass the performance of a model three times its size on specific, complex tasks. The key lies in selecting the right combination of model, method, and data.