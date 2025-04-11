import unittest
import pandas as pd

import sys
sys.path.insert(0, '../src/')

from data_preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
 
    def setUp(self):

        data = pd.DataFrame({
            'threshold': ['1-5', 'unknown', '10-20', '5-10'],
            'units': ['ppm', 'µg/kg', 'ng/g', 'mg/kg'],
            'values': [1, 2, 3, 4]
        })
        self.preprocessor = DataPreprocessor(data)
 
    def test_clean_unknow_threshold(self):
        result = self.preprocessor.clean_unknow_threshold('threshold')
        expected_data = pd.DataFrame({
            'threshold': ['1-5', '10-20', '5-10'],
            'units': ['ppm', 'ng/g', 'mg/kg'],
            'values': [1, 3, 4]
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_data)
 
    def test_calculate_mean_from_range(self):
        result = self.preprocessor.calculate_mean_from_range('threshold')
        expected_data = pd.DataFrame({
            'threshold': ['1-5', 'unknown', '10-20', '5-10'],
            'units': ['ppm', 'µg/kg', 'ng/g', 'mg/kg'],
            'values': [1, 2, 3, 4],
            'processed_threshold': [3.0, float('nan'), 15.0, 7.5]
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_data)
 
    def test_convert_units_ppm(self):
        result = self.preprocessor.convert_units_ppm('values', 'units')
        expected_data = pd.DataFrame({
            'threshold': ['1-5', 'unknown', '10-20', '5-10'],
            'units': ['ppm', 'µg/kg', 'ng/g', 'mg/kg'],
            'values': [1, 2, 3, 4],
            'Threshold_ppm': [1, 2, 0.003, 4000]
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_data)
 
if __name__ == '__main__':
    unittest.main()