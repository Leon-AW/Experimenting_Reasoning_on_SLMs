# Robustness Testing and Comparing Reasoning Techniques for Small Language Models

This repository contains the study project "Robustness Testing and Comparing Reasoning Techniques for Small Language Models". It investigates the effectiveness of different reasoning enhancement techniques on a 1-billion parameter Llama 3.2 model, comparing its performance against a baseline 1B model, and a larger 3B parameter Llama 3.2 model.

## Project Goal

The primary goal of this project was to explore and evaluate methods to improve the reasoning capabilities of Small Language Models (SLMs). As described in the project exposé, we investigated three distinct approaches:

1.  **Prompting and In-Context Learning:** Utilizing techniques like Chain-of-Thought (CoT) and Self-Consistency on pre-trained models without further finetuning.
2.  **Multi-Stage Finetuning:** Finetuning the base model on various instruction and reasoning datasets, inspired by the Orca 2 paper.
3.  **Hybrid Method (STaR):** A hybrid approach combining finetuning and prompting, where the model is finetuned on its own generated reasoning steps (rationales).

The core research questions were:
- Can reasoning techniques substantially improve the performance of a 1B model?
- Are these improvements sufficient to match or exceed the performance of a larger 3B model?
- Which approaches are most effective?

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

## Key Findings

The experiments revealed several key insights into how different reasoning strategies impact SLM performance.

### How did the experiments go?

Overall, the experiments were successful in demonstrating that reasoning techniques can significantly enhance the performance of a 1B parameter model. However, the effectiveness of each technique varied greatly depending on the base model's initial capabilities and the specific task.

- **Prompting:** Advanced prompting techniques like Chain-of-Thought (`cot`) were often detrimental to the performance of the base 1B model. However, when applied to the `1b-instruct` model, these same techniques yielded substantial improvements, suggesting that a model must first be properly instruction-following for complex prompting to be effective.

- **Finetuning:** Finetuning on a general instruction dataset like SlimOrca provided only marginal gains. In contrast, the `1b-sft-mixed-best` model, which was finetuned on a mix of general and task-specific data, showed remarkable performance increases across multiple benchmarks.

- **Hybrid Method (STaR):** The `star` model, after only one iteration of self-improvement, showed a massive boost in performance on math reasoning (`gsm8k`), more than doubling the score of the base 1B model. However, its performance on other benchmarks was often lower than the base model, indicating that the reasoning skills learned for one domain did not generalize well without further training.

### Comparison to Baselines

When comparing the enhanced 1B models to the simple-prompted Llama 3.2 1B and 3B models, we found:

- All specialized 1B models significantly outperformed the **base 1B model** on their respective strong suits. For example, `1b-instruct` excelled in general reasoning, `1b-sft-mixed-best` was a strong all-rounder, and `star` dominated mathematical reasoning.

- The ultimate goal was to see if a 1B model could match the **3B model**. Our results show this is indeed possible, but it requires the right approach.

### Which 1B approach could beat the 3B model?

Yes, several 1B models were able to outperform the Llama 3.2 3B model (using a simple prompt) on specific benchmarks.

- **`1b-instruct` Model:** Using Chain-of-Thought prompting with self-consistency, this model achieved an impressive **51.8%** on `gsm8k`, far surpassing the 3B model's 12.2%. It also came close to or exceeded the 3B model on `commonsense_qa`.

- **`1b-sft-mixed-best` Model:** This finetuned model beat the 3B baseline on `gsm8k` (**15.3%** vs. 12.2%) and `commonsense_qa` (**58.3%** vs. 54.4%). With self-consistency, its `gsm8k` score jumped to **29.1%**.

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
| **GSM8K**      | Cot+SC (1B-INSTRUCT)        | 51.8%      |
| **ARC**        | Simple (3B)                 | 62.9%      |
| **RACE**       | Simple (3B)                 | 68.3%      |
| **MMLU**       | Simple (3B)                 | 55.7%      |
| **DROP**       | Simple (3B)                 | 14.3%      |
| **CommonsenseQA** | Plan+SC (1B-SFT-MIXED-BEST) | 58.7%      |

### The Impact of Model Size

Increasing model size from 1B to 3B parameters generally provides a significant performance boost. The following table shows the improvement on the `gsm8k` benchmark.

| Template   | 1B Accuracy | 3B Accuracy | Improvement   |
|------------|-------------|-------------|---------------|
| Simple     | 5.5%        | 12.2%       | +6.7pp        |
| CoT        | 4.9%        | 14.3%       | +9.4pp        |

This underscores the challenge: a larger model often sets a high baseline that reasoning techniques on smaller models must overcome.

### The Effect of Self-Consistency

Self-consistency (SC) is a powerful technique, but its effectiveness varies. For math-heavy tasks like `gsm8k` and `drop`, it provides a substantial boost. However, for multiple-choice reasoning tasks like `arc` and `mmlu`, it sometimes *hurts* performance on the base 1B model, suggesting it may amplify biases if the model's underlying reasoning is not already sound.

The average effect of Self-Consistency on the base 1B model was a **+0.9pp** improvement, but this hides the high variability shown below.

| Dataset   | Template   | No SC   | With SC   | SC Effect   |
|-----------|------------|---------|-----------|-------------|
| **gsm8k** | Cot        | 4.9%    | 8.8%      | **+3.9pp**  |
| **drop**  | Simple     | 6.7%    | 9.5%      | **+2.8pp**  |
| **arc**   | Simple     | 39.8%   | 37.8%     | **-2.0pp**  |
| **mmlu**  | Simple     | 41.0%   | 38.3%     | **-2.7pp**  |

## Conclusion: Answering the Research Questions

The results of this project provide clear answers to the initial research questions:

1.  **Are performance improvements substantial?** Yes, the improvements are not just marginal but can lead to performance boosts of 2-5x on specific tasks (e.g., `gsm8k`), elevating the 1B model into a much higher capability class.

2.  **Is it just reasoning, or are there other factors?** The underlying model is critical. The `1b-instruct` model served as a much better foundation for prompting than the base model. Furthermore, the finetuning data composition was a decisive factor, as shown by the success of the `1b-sft-mixed-best` model. Therefore, performance stems from a combination of the base model's alignment, the reasoning technique applied, and the data used for finetuning.

3.  **Can these approaches work on a 1B model?** Absolutely. This project confirms that reasoning enhancement is not exclusive to large models and can be highly effective for models with as few as one billion parameters.

4.  **Can techniques be combined?** The success of the `1b-sft-mixed-best` model, which was later enhanced with `cot` prompting, shows that combining finetuning with sophisticated prompting is a powerful strategy.

In conclusion, this study demonstrates that by intelligently applying finetuning and prompting techniques, a small 1B parameter language model can achieve and even surpass the reasoning performance of a model three times its size on specific, complex tasks. The key lies in selecting the right combination of model, method, and data. 