import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml

from models.recap import build_model


POLARITY_NAMES = ("Negative", "Neutral", "Positive")
MODALITY_NAMES = ("text", "audio", "vision")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict sentiment for the unlabeled Attachment 3 PKL files."
    )
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument(
        "--output_csv", default="outputs/attachment3_predictions.csv"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected_count", type=int, default=30)
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def tokenize_text(tokenizer, text, max_length, device):
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    token_type_ids = encoded.get(
        "token_type_ids", [0] * len(encoded["input_ids"])
    )
    text_bert = np.stack(
        [
            encoded["input_ids"],
            encoded["attention_mask"],
            token_type_ids,
        ],
        axis=0,
    ).astype(np.float32)
    return torch.from_numpy(text_bert).unsqueeze(0).to(device)


def load_attachment_sample(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict) or "test" not in payload:
        raise ValueError(f"{path.name}: expected a top-level 'test' dictionary")
    sample = payload["test"]
    required = {"raw_text", "audio", "vision"}
    missing = required.difference(sample)
    if missing:
        raise ValueError(f"{path.name}: missing fields {sorted(missing)}")

    raw_text = np.asarray(sample["raw_text"]).reshape(-1)
    audio = np.asarray(sample["audio"], dtype=np.float32)
    vision = np.asarray(sample["vision"], dtype=np.float32)
    if len(raw_text) != 1 or audio.shape[0] != 1 or vision.shape[0] != 1:
        raise ValueError(f"{path.name}: each file must contain exactly one sample")
    return str(raw_text[0]), audio, vision


def zero_row_rate(features):
    rows = np.asarray(features)[0]
    return float(np.mean(np.all(rows == 0, axis=-1)))


def predict_one(model, tokenizer, path, max_text_length, device):
    raw_text, audio, vision = load_attachment_sample(path)
    text_tensor = tokenize_text(
        tokenizer, raw_text, max_text_length, device
    )
    audio_tensor = torch.from_numpy(audio).to(device)
    vision_tensor = torch.from_numpy(vision).to(device)

    complete_input = (None, None, None)
    incomplete_input = (vision_tensor, audio_tensor, text_tensor)
    with torch.no_grad():
        output = model(
            complete_input,
            incomplete_input,
            labels=None,
            mode="fusion_prediction",
        )

    intensity_raw = float(output["sentiment_preds"].reshape(-1)[0].item())
    intensity_pred = float(np.clip(intensity_raw, -3.0, 3.0))
    probabilities = torch.softmax(output["polarity_logits"], dim=-1)[0]
    class_id = int(probabilities.argmax().item())
    modality_weights = output["attention_weights"][0]

    row = {
        "sample_id": path.stem,
        "source_file": path.name,
        "raw_text": raw_text,
        "intensity_pred": intensity_pred,
        "intensity_raw": intensity_raw,
        "polarity_pred": POLARITY_NAMES[class_id],
        "prob_negative": float(probabilities[0].item()),
        "prob_neutral": float(probabilities[1].item()),
        "prob_positive": float(probabilities[2].item()),
        "text_weight": float(modality_weights[0].item()),
        "audio_weight": float(modality_weights[1].item()),
        "vision_weight": float(modality_weights[2].item()),
        "main_modality": MODALITY_NAMES[int(modality_weights.argmax().item())],
        "audio_zero_row_rate": zero_row_rate(audio),
        "vision_zero_row_rate": zero_row_rate(vision),
    }
    return row


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    config = load_config(args.config_file)
    model = build_model(config).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No PKL files found in {input_dir}")
    if args.expected_count and len(files) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} PKL files, found {len(files)}"
        )

    tokenizer = model.bertmodel.get_tokenizer()
    max_text_length = config["model"]["feature_extractor"]["input_length"][0]
    rows = []
    for index, path in enumerate(files, start=1):
        row = predict_one(model, tokenizer, path, max_text_length, device)
        rows.append(row)
        print(
            f"[{index:02d}/{len(files):02d}] {path.name}: "
            f"{row['polarity_pred']}, intensity={row['intensity_pred']:.4f}"
        )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predictions to: {output_path}")


if __name__ == "__main__":
    main()
