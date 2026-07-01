"""Model assets for CLS.

Drop the chosen BERT-finetune jetson_leg checkpoint (a ``.pt`` from
``LIMU-BERT-Public/saved/.../bert_classifier_base_gru_jetson_leg_10_20_both_xyz*/``) into
this directory and point ``cls.model_path`` at it. ``CLASSES`` is the label order the
checkpoint was trained with (``dataset/jetson_leg/label_map.json``, ids 0..6).
"""

CLASSES = ["stand", "walk", "turn", "jog", "rampascent", "stairascent", "stairdescent"]
