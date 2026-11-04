"""DiracLM (QL8): open-weights release tooling for qweave's quantum-inspired LLM line.

  recipe.py          release recipe: base model + components, each gated by a measured report
  gates.py           turn QL3 / QL5 / QL7 reports into PASS / FAIL verdicts
  reasoning_eval.py  MATH-500, GPQA-Diamond, BBH harness with per-item results and
                     paired-bootstrap comparisons against the plain base model
  model_card.py      Hugging Face model card generated only from measured numbers
  package.py         build the release folder (weights, config, card, manifest); upload
                     is private by default and refuses to run without an approval note
  loader.py          load a DiracLM folder with QSA attention switched on

Nothing in this package spends GPU money or publishes anything on its own.
"""

NAME = "DiracLM"
