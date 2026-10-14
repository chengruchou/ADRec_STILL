"""Dataset-first training: one foreground entry point for every implemented backend.

``python -m adrec_still train`` points a model at a dataset and trains it, showing
progress in the terminal and keeping ``last.pt`` / ``best.pt`` as the only durable
model files. ``python -m adrec_still batch`` runs a model/seed list in sequence. Nothing
in this package builds a cohort index, verifies historical dataset hashes or binds
a run to the machine it started on.
"""
