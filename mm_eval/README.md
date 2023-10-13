# mm_eval Usage

```py
from mm_eval import evaluate_engine

# mmvet
metric = evaluate_engine("mmvet", model)  # model can be your model or None

# mmbench
metric = evaluate_engine("mmbench", model)  # model can be your model or None
```