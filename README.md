### paper results
| Model | Method | Accuracy (%) |
|-------|--------|-------------|
| Qwen2.5-72b-instruct | prompt-based | 20.6 |
| Qwen2.5-72b-instruct | astute-rag | 20.8 |
| Qwen2.5-72b-instruct | madam-rag | 26.4 |

### Before Jun 29
Tried test time prompts manually iterated (not in detail)

### Jun 29
- Cleaned Structure
- Baseline inference and evaluation pipeline
- Baseline accuracy - 0.184
- Implemented dspy baseline
- dspy predict baseline accuracy - 16%
- after dspy simple optimiser MIRO_v2 - 18% - few shot example generation error
- resolved error MIRO_v2 - 22% - saved as miro_v2_predict.json
- added miro_v2_cot file
- dspy cot basleine accuracy - 18%
- dspy cot optimised - 18% - not generating few shot examples - evaluation-metric should be reconsidered?
- simba predict - 22%
- simba cot - 18%
- threatened lm with miro predict - 20%
- copro predict - 18%
- copro predict with miro few shots - 18 %