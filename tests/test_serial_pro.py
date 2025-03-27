import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import time
import matplotlib.pyplot as plt
from serialPro import init, update

class TestSerialPro(unittest.TestCase):

    def setUp(self):
        self.mock_lines = {
            'Temperature (C)': MagicMock(),
            'Voltage (V)': MagicMock(),
            'Current (mA)': MagicMock(),
            'Power (mW)': MagicMock(),
            'Speed (km/h)': MagicMock(),
            'SoC (%)': MagicMock(),
            'Predicted Range (Km)': MagicMock()
        }
        for line in self.mock_lines.values():
            line.set_data = MagicMock()

    def test_init_function(self):
        result = init()
        self.assertEqual(len(result), 7)
        for line in self.mock_lines.values():
            line.set_data.assert_called_with([], [])

    @patch('serialPro.ser')
    def test_update_valid_data(self, mock_ser):
        mock_ser.in_waiting = True
        mock_ser.readline.return_value = b"25.5,12.0,100,1200,30,80,150\n"
        
        result = update(0)
        self.assertEqual(len(result), 7)

    @patch('serialPro.ser')
    def test_update_invalid_data_format(self, mock_ser):
        mock_ser.in_waiting = True
        mock_ser.readline.return_value = b"invalid,data\n"
        
        result = update(0)
        self.assertEqual(len(result), 7)

    @patch('serialPro.ser')
    def test_update_empty_serial_buffer(self, mock_ser):
        mock_ser.in_waiting = False
        
        result = update(0)
        self.assertEqual(len(result), 7)

    @patch('serialPro.ser')
    def test_update_max_points_limit(self, mock_ser):
        mock_ser.in_waiting = True
        mock_ser.readline.return_value = b"25.5,12.0,100,1200,30,80,150\n"
        
        for _ in range(105):  # Exceed max_points (100)
            update(0)
            
        self.assertTrue(len(time_data) <= 100)

    @patch('serialPro.ser')
    def test_update_decode_error(self, mock_ser):
        mock_ser.in_waiting = True
        mock_ser.readline.return_value = b"\xff\xfe invalid bytes"
        
        result = update(0)
        self.assertEqual(len(result), 7)

if __name__ == '__main__':
    unittest.main()
