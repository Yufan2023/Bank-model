import unittest
from src.data_loader import load_data

class TestDataLoader(unittest.TestCase):
    def test_load_data_structure(self):
        data = load_data('2023-01-01', '2024-01-01')
        self.assertTrue(isinstance(data, pd.DataFrame))
        self.assertTrue('TD_Close' in data.columns)

if __name__ == '__main__':
    unittest.main()
