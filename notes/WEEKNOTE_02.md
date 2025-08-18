# Week 2 Note

## What I tried
- added sample.py
- added config.yaml
- modified train.py

## What worked
- run train.py and sample.py without error. tho the result is a bit weird.

## What broke
- I don't know what modification I made, but the output becomes `FLxs the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the` only the first few characters may change, change random seed has no effect.
- turns out it's not a mistake. it happens because I chang sampling from `multinomial` to `argmax`.
- and because bigram LM has context length = 1 character. rediculous.
- still need to fix this in sample.py

## What's next (Week 3)
- you see there's a file utils.py where I put set_seed and save_plot? this file is basically for helpers (tho no one ever told me what a helper is, i kind of get it from its name😂) can I throw some helpers common in train.py and sample.py to utils?
- Add documentation
- Refactor data pipeline; add config-driven hyperparams.
- the loss.png is old. need update.
- tho it's deterministic (i guess). no documentation yet.
- and other tasks my boyfriend gives me.


## What I learned
- config.yaml is actually an alternative to CLI. using it, we can drop 
- temperature / top-k / top-p are hyperparameters in decoding. not in the training model.
- dim = -n nth to the last
- batch size controls how many samples are grouped together per optimization step. after one batch, update the model.
### sample 
- What's the purpose of having a "sample.py" file? I know it know load .pt. but what's its advantage? what's in sample.py that train.py cannot do?
- depends on `model` and `data`. same level with train.py

## my question
- Are these products functionally equivalent? like we only choose one?