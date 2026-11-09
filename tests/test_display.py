# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.2", "transformers>=4.44", "numpy>=1.26"]
# ///
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interp_sae.display import feature_rows, token_rows, steering_rows, feature_ids


def sample_report():
    return {
        "point": "resid",
        "layer": 1,
        "stats": [
            {"feature": 2, "frequency": 0.9, "mean": 1.5},
            {"feature": 0, "frequency": 0.4, "mean": 0.2},
        ],
        "top_tokens": {
            "2": [
                {"feature": 2, "activation": 3.0, "prompt": 0, "position": 1, "token": "cat"},
                {"feature": 2, "activation": 1.0, "prompt": 1, "position": 0, "token": "dog"},
            ],
            "0": [
                {"feature": 0, "activation": 0.5, "prompt": 0, "position": 2, "token": "sat"},
            ],
        },
        "steering": [
            {"coef": 0.0, "text": "a plain line"},
            {"coef": 4.0, "text": "a steered line"},
        ],
    }


def test_feature_rows():
    rows = feature_rows(sample_report())
    assert len(rows) == 2
    assert [row["feature"] for row in rows] == [2, 0]
    assert rows[0] == {"feature": 2, "frequency": 0.9, "mean": 1.5}
    for row in rows:
        assert set(row.keys()) == {"feature", "frequency", "mean"}


def test_token_rows():
    rows = token_rows(sample_report(), 2)
    assert len(rows) == 2
    activations = [row["activation"] for row in rows]
    assert activations == sorted(activations, reverse=True)
    assert rows[0] == {"token": "cat", "activation": 3.0, "prompt": 0, "position": 1}
    for row in rows:
        assert set(row.keys()) == {"token", "activation", "prompt", "position"}


def test_steering_rows():
    rows = steering_rows(sample_report())
    assert len(rows) == 2
    assert [row["coef"] for row in rows] == [0.0, 4.0]
    assert rows[1] == {"coef": 4.0, "text": "a steered line"}
    for row in rows:
        assert set(row.keys()) == {"coef", "text"}


def test_feature_ids():
    assert feature_ids(sample_report()) == [2, 0]


def test_missing_sections_return_empty():
    assert feature_rows({}) == []
    assert token_rows({}, 2) == []
    assert token_rows(sample_report(), 99) == []
    assert steering_rows({}) == []
    assert feature_ids({}) == []


def main():
    tests = [
        test_feature_rows,
        test_token_rows,
        test_steering_rows,
        test_feature_ids,
        test_missing_sections_return_empty,
    ]
    for test in tests:
        test()
        print(f"{test.__name__} ok")
    print("all passed")


if __name__ == "__main__":
    main()
