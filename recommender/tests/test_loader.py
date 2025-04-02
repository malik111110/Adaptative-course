import unittest
from recommender.data.loader import DataLoader

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.loader = DataLoader()

    def test_load_data(self):
        data = self.loader.load_data('path/to/dataset.csv')
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)

    def test_load_invalid_data(self):
        with self.assertRaises(FileNotFoundError):
            self.loader.load_data('path/to/nonexistent.csv')

    def test_data_format(self):
        data = self.loader.load_data('path/to/dataset.csv')
        self.assertTrue(isinstance(data, list))
        self.assertTrue(all(isinstance(item, dict) for item in data))

if __name__ == '__main__':
    unittest.main()