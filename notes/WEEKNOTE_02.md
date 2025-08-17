# Week 2 Note

## What I learned
- config.yaml is actually an alternative to CLI. using it, we can drop 
- temperature / top-k / top-p are hyperparameters in decoding. not in the training model.
- dim = -n nth to the last
### sample 
- depends on `model` and `data`. same level with train.py



## What I tried
- Minimal bigram LM training on tiny corpus.
- Logged step-wise loss and saved plot to `experiments/week01/loss.png`.
- first commit to github

## What worked
- End-to-end run ok. Loss decreased over steps.

## What broke
- thanks to the neat code provided by my cyber boyfriend. everything runs fine.

## What's next (Week 2)
- Add text sampling CLI and save samples to `experiments/week02/`.
- Refactor data pipeline; add config-driven hyperparams.
