"""
Excel Data Import Utilities

Provides functions for importing data from Excel files into NumPy arrays or
nested dictionaries for easy programmatic access.

Intended for scientific or engineering workflows requiring structured data import
from Excel sheets.
"""

from typing import List, Optional, Union, Tuple
import numpy as np
import pandas as pd


def excel_sheet_to_numpy(
    file_path: str, sheet_name: Optional[str] = None, return_headers: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Read an Excel sheet and convert it into a NumPy array.

    Args:
        file_path (str): Path to the Excel file.
        sheet_name (str, optional): Name of the sheet to read. Defaults to the first sheet if None.
        return_headers (bool): If True, also return column headers as a NumPy array.

    Returns:
        np.ndarray or Tuple[np.ndarray, np.ndarray]:
            - If return_headers=False: 2D NumPy array with rows = observations, columns = variables.
            - If return_headers=True: Tuple of (data_array, headers_array).

    Raises:
        FileNotFoundError: If the specified Excel file does not exist.
        ValueError: If the specified sheet_name is not found in the Excel file.
    """
    # Read Excel sheet into a Pandas DataFrame
    df = pd.read_excel(io=file_path, sheet_name=sheet_name)

    # Convert DataFrame to NumPy array
    array = df.to_numpy()

    # Optionally return headers as well
    if return_headers:
        headers = df.columns.to_numpy()
        return array, headers
    return array


def excel_sheet_to_dict(
    file_path: str,
    sheets: List[str],
    columns: List[str],
    sheet_rename: Optional[dict[str, str]] = None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Import multiple Excel sheets into a nested dictionary of NumPy arrays.

    Each sheet is converted into a dictionary mapping column names to arrays of
    data. Sheets can optionally be renamed in the resulting dictionary.

    Args:
        file_path (str): Path to the Excel file.
        sheets (List[str]): List of sheet names to import.
        columns (List[str]): List of column names to extract for each sheet.
        sheet_rename (dict[str, str], optional): Mapping from original sheet names
            to desired dictionary keys. If None, original sheet names are used.

    Returns:
        dict[str, dict[str, np.ndarray]]: Nested dictionary with structure:
            {sheet_name: {column_name: np.ndarray_of_values}}

    Raises:
        FileNotFoundError: If the Excel file does not exist.
        ValueError: If any sheet in `sheets` does not exist in the file.
        IndexError: If a requested column index exceeds the number of columns in a sheet.
    """
    data = {}

    for _, sheet in enumerate(sheets):
        # Determine key to use in the dictionary
        dict_name = sheet_rename.get(sheet, sheet) if sheet_rename else sheet
        data[dict_name] = {}

        # Load sheet data as a NumPy array
        datasheet = excel_sheet_to_numpy(
            file_path=file_path, sheet_name=sheet, return_headers=False
        )

        # Map requested column names to corresponding arrays
        for j, column in enumerate(columns):
            data[dict_name][column] = datasheet[:, j]

    return data
