import unittest
from src.model import create_model

class TestModel(unittest.TestCase):
    def test_model_creation(self):
        model = create_model(input_shape=(60, 10))  # Example input shape
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
