from __future__ import annotations

import os
import time
import json
import itertools
from collections import Counter
from pathlib import Path
from typing import List, Tuple
import requests
import pandas as pd
from tqdm.auto import tqdm
from langdetect import detect, DetectorFactory
from spellchecker import SpellChecker
from newsapi import NewsApiClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re
import openai
from openai import RateLimitError, OpenAIError, OpenAI

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

### Configuration ###
TARGET_N_HEADLINES = 1500  
TRAIN_SIZE = 1000         
TEST_SIZE = 500          
SENTIMENT_CLASSES = [
    "Very Positive",
    "Positive",
    "Neutral",
    "Negative",
    "Very Negative",
]

NEWSAPI_KEY = "Mediastack API KEY"
OPENAI_API_KEY = "OPEN AI KEY"
assert NEWSAPI_KEY, "MediaStack env var not found"
assert OPENAI_API_KEY, "OPENAI_API_KEY env var not found"

newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
openai.api_key = OPENAI_API_KEY

DetectorFactory.seed = 42  
spell = SpellChecker(language="en")




# 2  data collection (mediastack)
def fetch_mediastack_headlines(total: int = 2200, page_size: int = 100) -> List[str]:
    """Fetch up to *total* latest English headlines from mediastack."""
    headlines: List[str] = []
    for offset in range(0, total, page_size):
        r = requests.get(
            "http://api.mediastack.com/v1/news",
            params={
                "access_key": "Mediastack API KEY",
                "languages": "en",
                "sort": "published_desc",
                "limit": page_size,
                "offset": offset,
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        batch = [art["title"] for art in data if art.get("title")]
        headlines.extend(batch)
        print(f"Fetched {len(batch):3} (total {len(headlines):4})")
        if len(data) < page_size:
            break  
        time.sleep(1)  
    return headlines[:total]

# 3  data cleaning


def is_english(txt: str) -> bool:
    try:
        return detect(txt) == "en"
    except Exception:
        return False


def misspell_ratio(txt: str) -> float:
    tokens = [w.strip(".,:;!?()'\"\u2019").lower() for w in txt.split() if w.isalpha()]
    if not tokens:
        return 1.0
    return len(spell.unknown(tokens)) / len(tokens)


def clean_headlines(raw: List[str]) -> List[str]:
    df = pd.DataFrame({"headline": raw})
    df.drop_duplicates(subset="headline", inplace=True)
    df["headline"] = df["headline"].str.strip()
    df = df[df["headline"].apply(is_english)]
    df = df[df["headline"].apply(lambda t: misspell_ratio(t) <= 0.25)]
    df.reset_index(drop=True, inplace=True)
    return df["headline"].tolist()

# 3 load_csvs_up

def load_or_fetch_headlines() -> List[str]:
    csv_path = Path("cleaned_headlines.csv")
    if csv_path.exists():
        print("📂  Using cleaned_headlines.csv already on disk …")
        return pd.read_csv(csv_path)["headline"].dropna().tolist()

    print("🌐  cleaned_headlines.csv not found → fetching from mediastack …")
    raw = fetch_mediastack_headlines()
    pd.DataFrame({"headline": raw}).to_csv("raw_headlines.csv", index=False)
    print("💾  Saved raw_headlines.csv")

    cleaned = clean_headlines(raw)
    pd.DataFrame({"headline": cleaned}).to_csv(csv_path, index=False)
    print("💾  Saved cleaned_headlines.csv")


# 4  GPT‑4o sentiment annotation
    

client = OpenAI(api_key=OPENAI_API_KEY)

def extract_json_block(text: str) -> str:
    """
    Strips Markdown-style code blocks (```json ... ```) from GPT responses
    so they can be parsed as proper JSON.
    """
    match = re.search(r"```(?:json)?\s*(\[\s*{.*?}\s*]\s*)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def gpt_label_batch(batch: List[str]) -> List[str]:
    system_prompt = (
        "You are a strict sentiment labeling assistant. "
        "Return ONLY a valid JSON list. Do NOT wrap it in triple backticks. "
        'Format: [{"headline": "<headline>", "label": "<class>"}]. '
        "Choose label from: Very Positive, Positive, Neutral, Negative, Very Negative."
    )

    user_prompt = f"Headlines (JSON list): {json.dumps(batch, ensure_ascii=False)}"

    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            content = resp.choices[0].message.content.strip()
            print("\n🔎 RAW GPT RESPONSE:\n", content)

            json_text = extract_json_block(content)
            parsed = json.loads(json_text)
            return [item["label"] for item in parsed]

        except (json.JSONDecodeError, KeyError):
            print("⚠️  JSON parse error – retrying…")
            continue
        except RateLimitError:
            print("⏳  Rate‑limited – sleeping 20 s…")
            time.sleep(20)
        except OpenAIError as e:
            print(f"⚠️  OpenAI error: {e} – retrying in 10 s…")
            time.sleep(10)

    raise RuntimeError("GPT labelling failed after retries")

def annotate_sentiment(headlines: List[str]) -> List[str]:
    labels: List[str] = []
    BATCH = 50
    for i in tqdm(range(0, len(headlines), BATCH), desc="Labelling w/ GPT‑4o"):
        labels.extend(gpt_label_batch(headlines[i : i + BATCH]))
    return labels

# 5  model ft

def encode(batch, tok):
    return tok(batch["headline"], padding="max_length", truncation=True, max_length=48)


def make_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, tok):
    train_ds = Dataset.from_pandas(train_df).map(lambda b: encode(b, tok), batched=True)
    test_ds = Dataset.from_pandas(test_df).map(lambda b: encode(b, tok), batched=True)
    label2id = {lbl: i for i, lbl in enumerate(SENTIMENT_CLASSES)}
    train_ds = train_ds.map(lambda b: {"label": label2id[b["label"]]})
    test_ds = test_ds.map(lambda b: {"label": label2id[b["label"]]})
    cols = ["input_ids", "attention_mask", "label"]
    train_ds.set_format(type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=cols)
    return train_ds, test_ds, label2id


# 6  main

def main():
    # fetch
    # print("📡  Fetching headlines…")
    # raw = fetch_mediastack_headlines()
    # print(f"Fetched {len(raw):,} raw headlines")

    # pd.DataFrame({"headline": raw}).to_csv("raw_headlines.csv", index=False)
    # print("💾  Saved raw_headlines.csv")

    # clean
    cleaned = load_or_fetch_headlines()
    print(f"After cleaning: {len(cleaned):,}")
    pd.DataFrame({"headline": cleaned}).to_csv("cleaned_headlines.csv", index=False)
    print("💾  Saved cleaned_headlines.csv")

    if len(cleaned) < TARGET_N_HEADLINES:
        raise RuntimeError(
            f"Need at least {TARGET_N_HEADLINES} clean headlines; got {len(cleaned)}. "
            "Consider increasing fetch limit or relaxing filters."
        )
    cleaned = cleaned[:TARGET_N_HEADLINES]

    #  annotate 
    labels = annotate_sentiment(cleaned)
    df = pd.DataFrame({"headline": cleaned, "label": labels})

    # split 
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, train_size=TRAIN_SIZE, stratify=df["label"], random_state=42
    )
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # tokeniser & datasets
    ckpt = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(ckpt)
    train_ds, test_ds, label2id = make_datasets(train_df, test_df, tok)

    #  Model & trainer -
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt,
        num_labels=len(SENTIMENT_CLASSES),
        id2label={i: l for i, l in enumerate(SENTIMENT_CLASSES)},
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir="sentiment_model",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        num_train_epochs=3,
        fp16=False,
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds)

    print("🚀  Fine‑tuning model…")
    trainer.train()

    # evaluation 
    print("🔍  Evaluating…")
    preds = trainer.predict(test_ds)
    y_pred = preds.predictions.argmax(axis=1)
    y_true = preds.label_ids
    print("\n==== Classification report ====")
    print(classification_report(y_true, y_pred, target_names=SENTIMENT_CLASSES))
    print("Accuracy:", accuracy_score(y_true, y_pred))

    # persist artefacts 
    df.to_csv("all_headlines_labelled.csv", index=False)
    train_df.to_csv("train_1000.csv", index=False)
    test_df.to_csv("test_500.csv", index=False)
    tok.save_pretrained("sentiment_model/tokenizer")

if __name__== "__main__":
    main() 
