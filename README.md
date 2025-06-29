### paper results
| Model | Method | Accuracy (%) |
|-------|--------|-------------|
| Qwen2.5-72b-instruct | prompt-based | 20.6 |
| Qwen2.5-72b-instruct | astute-rag | 20.8 |
| Qwen2.5-72b-instruct | madam-rag | 26.4 |

### Before Jun 29
Tried test time prompts (not in detail)

### Jun 29
- Cleaned Structure
- Baseline inference and evaluation pipeline
- Baseline accuracy - 0.184
- Implemented dspy baseline
- dspy baseline accuracy - 16%
- after dspy simple optimiser MIRO_v2 - 18% - few shot example generation error
- resolved error MIRO_v2 - 22% - saved as miro_v2_predict.json