
# Running the App on Windows 11 PowerShell

## Setting Up the Virtual Environment
1. Create a virtual environment:
    ```powershell
    python -m venv venv
    ```
2. Activate the virtual environment:
    ```powershell
    .\venv\Scripts\activate
    ```

## Installing Dependencies
3. Install the required packages:
    ```powershell
    pip install -r requirements.txt
    ```

## Tests
4. Unnitests:
    ```powershell
    python -m unittest discover -s tests
    ```

## Model LSTM layers
5. Run the Architecture:
    ```powershell
    python model\lstm.py
    ```

## Data Preprocessing and Preparation
6. Cleaning:
    ```powershell
    python -m Preprocessing.preprocessing
    ```

Thanks for your help!
