"""Fast checks for the competition submission's data contract."""
import unittest

import numpy as np

from linear_baseline import consistent_intensity, features, temporal_evidence
from materialize_text import normalize_split


class CompetitionTests(unittest.TestCase):
    def test_scalar_special_test_is_batched(self):
        split = {
            'raw_text': 'a clip',
            'audio': np.ones((5, 74)),
            'vision': np.ones((5, 35)),
            'id': '01',
        }
        result = normalize_split(split)
        self.assertEqual(result['audio'].shape, (1, 5, 74))
        self.assertEqual(result['vision'].shape, (1, 5, 35))
        self.assertEqual(result['id'][0], '01')

    def test_missing_frames_do_not_dilute_mean(self):
        text = np.zeros((1, 3, 768), dtype=np.float32)
        text[0, 0] = 2
        audio = np.zeros((1, 3, 74), dtype=np.float32)
        audio[0, 1] = 4
        vision = np.zeros((1, 3, 35), dtype=np.float32)
        vision[0, 2] = -1
        split = {'text': text, 'audio': audio, 'vision': vision,
                 'text_bert': np.array([[[101, 0, 0], [1, 1, 1], [0, 0, 0]]])}
        result = features(split)[0]
        self.assertAlmostEqual(result[0], 2)
        self.assertAlmostEqual(result[1536], 4)
        self.assertAlmostEqual(result[1610], -1)
        self.assertAlmostEqual(result[1645], 1 / 3)

    def test_polarity_and_intensity_agree(self):
        raw = np.array([0.5, -0.7, 0.2, -0.3])
        classes = np.array([0, 1, 2, 2])
        result = consistent_intensity(raw, classes)
        self.assertLess(result[0], 0)
        self.assertEqual(result[1], 0)
        self.assertGreater(result[2], 0)
        self.assertGreater(result[3], 0)

    def test_window_delta_is_signed(self):
        text = np.zeros((1, 2, 768), dtype=np.float32)
        text[0, 0, 0] = 1
        text[0, 1, 0] = 3
        split = {'text': text,
                 'text_bert': np.array([[[101, 102], [1, 1], [0, 0]]])}
        model = {'cols': np.arange(768), 'mean': np.zeros(768),
                 'std': np.ones(768)}
        coef = np.zeros(768)
        coef[0] = 1
        evidence = temporal_evidence(split, 0, 'text', model, coef, top_k=2)
        self.assertIn('1-1@50.0%:delta=+1.0000', evidence)
        self.assertIn('0-0@0.0%:delta=-1.0000', evidence)


if __name__ == '__main__':
    unittest.main()
