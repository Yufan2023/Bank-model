import unittest
from src.collect_future_data import collect_data

class TestCollectFutureData(unittest.TestCase):
    def test_future_data_collection(self):
        data = collect_data('2024-01-02', '2024-12-31')
        self.assertFalse(data.empty)
        self.assertTrue('TD_Close' in data.columns)

if __name__ == '__main__':
    unittest.main()
