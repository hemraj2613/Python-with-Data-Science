from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

MODELS = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "valhalla/distilbart-mnli-12-3",
    "dslim/bert-base-NER",
    "distilbert-base-cased-distilled-squad",
    "t5-small",
    "distilbert-base-uncased",
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "facebook/nllb-200-distilled-600M",
]
for m in MODELS:
    print("Downloading", m)
    AutoTokenizer.from_pretrained(m)
    AutoModel.from_pretrained(m)
for st in ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]:
    print("Downloading", st)
    SentenceTransformer(st)
print("Done. All models are cached.", flush=True)
import os
os._exit(0)      # leave immediately (the program sometimes hangs while closing)
