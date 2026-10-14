"""The SASRec-H architecture trained once per observed known-target event gap."""

from adrec_still.models.baselines.sasrec import SASRecH


class SASRecEventOnly(SASRecH):
    """Matched SASRec-H parameters, encoder and scores; only training rows differ.

    The shared trainer requires the explicit event-only training profile. Neither
    silence duration nor timestamps enter the inherited encoder, and evaluation
    still scores the complete, unchanged canonical landmark cohorts.
    """

    name = "sasrec_event_only"
