def feature_rows(report):
    stats = report.get("stats")
    if not stats:
        return []
    return [
        {
            "feature": row["feature"],
            "frequency": row["frequency"],
            "mean": row["mean"],
        }
        for row in stats
    ]


def token_rows(report, feature_id):
    tokens = report.get("top_tokens")
    if not tokens:
        return []
    records = tokens.get(str(feature_id))
    if not records:
        return []
    return [
        {
            "token": record["token"],
            "activation": record["activation"],
            "prompt": record["prompt"],
            "position": record["position"],
        }
        for record in records
    ]


def steering_rows(report):
    steering = report.get("steering")
    if not steering:
        return []
    return [
        {
            "coef": example["coef"],
            "text": example["text"],
        }
        for example in steering
    ]


def feature_ids(report):
    tokens = report.get("top_tokens")
    if not tokens:
        return []
    return [int(fid) for fid in tokens]
