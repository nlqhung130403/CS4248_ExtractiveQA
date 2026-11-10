import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import importlib.util

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluation_qa", ROOT / "src/evaluation-qa.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
evaluate = module.evaluate
map_to_span = module.map_to_span
read_json = module.read_json
validate_predictions = module.validate_predictions


class EvaluationTests(unittest.TestCase):
    def test_mapping_preserves_original_offsets(self):
        context = "She studied in New  York."
        span, method, start, end = map_to_span('Answer: "new york"', context)
        self.assertEqual(span, "New  York")
        self.assertEqual(context[start:end], span)
        self.assertEqual(method, "case_whitespace")

    def test_mapping_failures_and_empty(self):
        self.assertEqual(map_to_span("Paris", "London is a city.")[1], "unmatched")
        self.assertEqual(map_to_span("", "London is a city.")[1], "empty")
        self.assertEqual(map_to_span("US", "Russia")[1], "unmatched")

    def test_prediction_bundle_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.json"
            for bundle, expected in [
                ({"q": "plain"}, {"q": "plain"}),
                ({"metrics": {}, "predictions": {"q": "clean"}}, {"q": "clean"}),
                ({"metrics": {}, "predictions": {"q": "clean"},
                  "raw_predictions": {"q": "Answer: clean"}}, {"q": "Answer: clean"}),
            ]:
                path.write_text(json.dumps(bundle))
                self.assertEqual(module.load_predictions(path)[0], expected)

    def test_validation(self):
        with self.assertRaises(ValueError):
            validate_predictions({"q1": "answer"}, {"q1": {}, "q2": {}})
        with self.assertRaises(ValueError):
            validate_predictions({"q1": None}, {"q1": {}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"q1": "a", "q1": "b"}')
            with self.assertRaises(ValueError):
                read_json(path)

    def test_official_cli_parity_and_known_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            dataset = {"version": "2.0", "data": [{"title": "Synthetic test", "paragraphs": [{
                "context": "Ada lived in New York. Bob lived in London.",
                "qas": [
                    {"id": "q1", "question": "Where did Ada live?", "answers": [{"text": "New York", "answer_start": 13}, {"text": "York", "answer_start": 17}]},
                    {"id": "q2", "question": "Who lived in London?", "answers": [{"text": "Bob", "answer_start": 23}]},
                    {"id": "q3", "question": "When was Ada born?", "answers": [], "is_impossible": True},
                    {"id": "q4", "question": "Where did Bob live?", "answers": [{"text": "London", "answer_start": 36}]},
                ]}]}]}
            predictions = {"q1": "York", "q2": 'Answer: "Bob"', "q3": "", "q4": "Atlantis"}
            data_path, predictions_path = directory / "data.json", directory / "demo.json"
            data_path.write_text(json.dumps(dataset))
            predictions_path.write_text(json.dumps(predictions))
            scorer = ROOT / "evaluate-v2.0.py"
            results = directory / "results"
            summary = evaluate(data_path, [predictions_path], scorer, results)
            self.assertEqual(summary[0]["exact"], 50.0)
            self.assertEqual(summary[1]["exact"], 75.0)
            self.assertEqual(summary[1]["mapping_failure_count"], 1)
            for variant in ["raw", "mapped"]:
                actual = json.loads(subprocess.check_output([
                    sys.executable, str(scorer), str(data_path),
                    str(results / "demo" / f"predictions.{variant}.json"),
                ], text=True))
                metrics = read_json(results / "demo" / f"metrics.{variant}.json")
                for key in ["exact", "f1", "total"]:
                    self.assertEqual(metrics[key], actual[key])
            # The prescribed v1.1 data has no unanswerable questions.
            dataset["version"] = "1.1"
            dataset["data"][0]["paragraphs"][0]["qas"].pop(2)
            del predictions["q3"]
            data_path.write_text(json.dumps(dataset))
            predictions_path.write_text(json.dumps(predictions))
            summary = evaluate(data_path, [predictions_path], scorer, results)
            self.assertAlmostEqual(summary[1]["exact"], 200 / 3)


if __name__ == "__main__":
    unittest.main()
