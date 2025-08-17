# Week 1 Note

## What I learned
- .gitignore contains folders and files you don't want to push to git. this filename is fixed by system.
- the structure of src: data, model, train, utils. train calls the other three.
### data
- argparse is a library to enable CLI (command‑line interface) using -- like python src\train.py --steps 1000 --lr 0.5
instead of import argparse, might as well from argparse import ArgumentParser, but nah.
- sys.path returns... well, system paths in string.
`__file__` is the path to current script in string. Path convert it to Path instance. resolve makes it absolute. parents[1] goes to E:\llm-lab. append add them together.
- library tqdm shows progress bar fro loops.



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
