import argparse
import json
from collections import defaultdict
from typing import Dict, List

from predict import load_model_and_tokenizer, predict


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-annotate JSONL with NER model")
    parser.add_argument("--input", required=True, help="Input JSONL with {text, label(optional)}")
    parser.add_argument("--output", required=True, help="Output JSONL with predicted labels")
    parser.add_argument("--limit", type=int, default=0, help="Limit records (0 = no limit)")
    parser.add_argument(
        "--format",
        choices=["internal", "doccano"],
        default="internal",
        help="Output format: internal (label dict) or doccano (label list)",
    )
    return parser.parse_args()


def entities_to_internal(text: str, entities: List[Dict]) -> Dict[str, Dict[str, List[List[int]]]]:
    label = defaultdict(lambda: defaultdict(list))
    for ent in entities:
        ent_text = text[ent["start"] : ent["end"] + 1]
        label[ent["type"]][ent_text].append([ent["start"], ent["end"]])
    return {k: dict(v) for k, v in label.items()}


def entities_to_doccano(entities: List[Dict]) -> List[List]:
    out = []
    for ent in entities:
        out.append([ent["start"], ent["end"] + 1, ent["type"]])
    return out


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer()

    count = 0
    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text", "")
            if not text:
                continue

            entities = predict(text, model, tokenizer)

            if args.format == "doccano":
                record_out = {"text": text, "label": entities_to_doccano(entities)}
            else:
                record_out = {"text": text, "label": entities_to_internal(text, entities)}

            fout.write(json.dumps(record_out, ensure_ascii=False) + "\n")
            count += 1
            if args.limit and count >= args.limit:
                break

    print(f"Pre-annotated {count} records -> {args.output}")


if __name__ == "__main__":
    main()
