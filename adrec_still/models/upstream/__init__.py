"""Upstream ADRec-fork models (KDD'25 ADRec, DiffuRec, DreamRec, SASRec+).

Ported from this repository's ``src/`` tree (the official ADRec implementation,
Chen et al., KDD 2025) so the models run inside the ``adrec_still`` package on
canonical POI data and on the bundled ``dataset.pkl`` files. Architectural
behavior is preserved; the execution fixes applied during the port (attention
key-padding mask, one-event prefixes, dynamic vocabulary sizing, path handling)
are listed in docs/MIGRATION.md.
"""
