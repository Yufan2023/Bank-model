import unittest
from src.preprocessing import preprocess_data

class TestPreprocessing(unittest.TestCase):
    def test_preprocessing_output(self):
        data = pd.DataFrame(...)  # Sample input data
        processed_data = preprocess_data(data)
        self.assertFalse(processed_data.isnull().values.any())

if __name__ == '__main__':
    unittest.main()
