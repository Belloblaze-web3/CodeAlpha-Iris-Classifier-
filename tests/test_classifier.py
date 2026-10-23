import tempfile
import unittest
from pathlib import Path

from src.classifier import load_data, predict, train


class ClassifierTests(unittest.TestCase):
    def test_data_shape_and_labels(self):
        X, y, classes, features = load_data()
        self.assertEqual(X.shape, (150, 4))
        self.assertEqual(len(y.unique()), 3)
        self.assertEqual(len(classes), 3)
        self.assertEqual(len(features), 4)

    def test_train_and_predict(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.joblib"
            metrics_path = Path(directory) / "metrics.json"
            metrics = train(model_path, metrics_path)
            self.assertTrue(model_path.exists())
            self.assertTrue(metrics_path.exists())
            self.assertGreaterEqual(metrics["accuracy"], 0.0)
            self.assertLessEqual(metrics["accuracy"], 1.0)
            result = predict([5.1, 3.5, 1.4, 0.2], model_path)
            self.assertIn(result["prediction"], metrics["classes"])
            self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=3)

    def test_rejects_wrong_feature_count(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.joblib"
            train(model_path, Path(directory) / "metrics.json")
            with self.assertRaises(ValueError):
                predict([1.0, 2.0], model_path)


if __name__ == "__main__":
    unittest.main()
