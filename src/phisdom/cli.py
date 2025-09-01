import argparse
import json
from .features.url import extract_url_features, vectorize_feature_dict


def main():
    parser = argparse.ArgumentParser(description="phisdom demo CLI")
    parser.add_argument("--url", type=str, required=True, help="URL to featurize")
    parser.add_argument("--ngram", type=int, default=3)
    args = parser.parse_args()

    feats = extract_url_features(args.url, ngram=args.ngram)
    keys, vals = vectorize_feature_dict(feats)
    out = {"url": args.url, "features": dict(zip(keys, vals))}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
