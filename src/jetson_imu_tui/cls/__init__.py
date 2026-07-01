"""Real-time activity classification (CLS) — vendored LIMU-BERT + GRU inference.

Self-contained copy of just the BERT-finetune inference path from the separate
``LIMU-BERT-Public`` repo (``models.py`` + ``utils.Preprocess4Normalization``), so the
Jetson only needs ``torch`` + ``numpy`` at runtime — no scipy / argparse / JSON config
machinery. See ``cls/nn.py`` for the verbatim model classes (state_dict keys must match
the checkpoint) and ``cls/service.py`` for the streaming sampler + inference loop.
"""
