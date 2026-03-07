import csv
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pills_core.ingestion.csv import CSVDataSource, CSVOptions


@pytest.fixture
def csv_dir():
    with tempfile.TemporaryDirectory(prefix="pills_csv_tests_") as tmpdir:
        yield Path(tmpdir)


def make_csv(
    base_dir: Path,
    rows: list[list],
    filename: str = "test.csv",
    sep: str = ",",
    encoding: str = "utf-8"
):
    path = base_dir / filename
    with open(path, "w", newline="", encoding=encoding) as f:
        writer = csv.writer(f, delimiter=sep)
        writer.writerows(rows)
    return str(path)


def make_raw_file(base_dir: Path, content: bytes, filename: str = "test.csv") -> str:
    path = base_dir / filename
    path.write_bytes(content)
    return str(path)


class TestValidate:

    @pytest.mark.negative
    def test_file_not_found_raises(self):
        src = CSVDataSource(CSVOptions(path="no-found-path"))
        with pytest.raises(FileNotFoundError, match="File not found"):
            src._validate()
    

    @pytest.mark.negative
    def test_empty_file_raises(self, csv_dir):
        path = make_csv(csv_dir, rows=[], filename="empty.csv")
        src = CSVDataSource(CSVOptions(path=path))
        with pytest.raises(ValueError, match="File is empty: "):
            src._validate()
    
    @pytest.mark.positive
    def test_valid_file_passes(self, csv_dir):
        path = make_csv(csv_dir, [["a", "b"], [1, 2]], filename="validate_ok.csv")
        src = CSVDataSource(CSVOptions(path=path))
        src._validate()


class TestReadRaw:

    @pytest.mark.positive
    def test_reads_plain_utf8(self, csv_dir):
        path = make_csv(csv_dir, [["name", "age"], ["Alice", 30]])
        src = CSVDataSource(CSVOptions(path=path))
        content = src._read_raw()
        assert "name" in content
        assert "age" in content
        assert "Alice" in content
        assert "30" in content        
    
    @pytest.mark.positive
    def test_strips_bom_prefix(self, csv_dir):
        raw = b"\xef\xbb\xbfa,b\n1,2\n"
        path = make_raw_file(csv_dir, raw, filename="test_raw_bom.csv")
        src = CSVDataSource(CSVOptions(path=path))
        content = src._read_raw()
        assert not content.startswith("\ufeff")
        assert content.startswith("a,b")
    
    @pytest.mark.positive
    def test_reads_cp1251_encoding(self, csv_dir):
        path = make_csv(csv_dir, [["name", "age"], ["Alice", 30]], encoding="cp1251")
        src = CSVDataSource(CSVOptions(path=path))
        content = src._read_raw()
        assert "name" in content
        assert "age" in content
        assert "Alice" in content
        assert "30" in content  



class TestDetectSeparator:

    @pytest.mark.positive
    def test_explicit_sep_returned_as_is(self, csv_dir):
        src = CSVDataSource(CSVOptions(path="x.csv", sep=";"))
        assert src._detect_separator("a;b\n1;2") == ";"
    
    @pytest.mark.positive
    def test_sniff_detects_tab_separator(self):
        src = CSVDataSource(CSVOptions(path="x.csv", sep=""))
        content = "a\tb\tc\n1\t2\t3\n"
        assert src._detect_separator(content) == "\t"
    
    @pytest.mark.negative
    def test_sniffer_error_fallsback(self):
        src = CSVDataSource(CSVOptions(path="x.csv", sep=""))
        with patch("csv.Sniffer.sniff", side_effect=csv.Error("cannot sniff")):
            result = src._detect_separator("garbage data")
        assert result == ","



class TestCoerceNumeric:

    @pytest.mark.positive
    def test_plain_numeric_strings_converted(self):
        src = CSVDataSource(CSVOptions(path="x.csv"))
        s = pd.Series(["1.0", "2.5", "3.14"])
        result = src._coerce_numeric(s)
        assert result.dtype == np.float64
    
    @pytest.mark.positive
    def test_currency_symbols_stripped(self):
        src = CSVDataSource(CSVOptions(path="x.csv"))
        s = pd.Series(["$100", "$200", "$300"])
        result = src._coerce_numeric(s)
        assert result.dtype == np.float64
        assert result.iloc[0] == pytest.approx(100.0)
    
    @pytest.mark.positive
    def test_percent_symbols_stripped(self):
        src = CSVDataSource(CSVOptions(path="x.csv"))
        s = pd.Series(["10%", "20%", "30%"])
        result = src._coerce_numeric(s)
        assert result.dtype == np.float64
    

    @pytest.mark.positive
    def test_custom_decimal_comma(self):
        src = CSVDataSource(CSVOptions(path="x.csv", decimal=","))
        s = pd.Series(["1,5", "2,0", "3,70"])
        result = src._coerce_numeric(s)
        assert result.dtype == np.float64
        assert result.iloc[0] == pytest.approx(1.5)
        

    @pytest.mark.negative
    def test_mostly_text_slays_string(self):
        src = CSVDataSource(CSVOptions(path="x.csv"))  
        s = pd.Series(["foo", "bar", "baz", "wex", "1"])
        result = src._coerce_numeric(s)
        assert not pd.api.types.is_float_dtype(result)
        assert pd.api.types.is_string_dtype(result)
    
    @pytest.mark.positive
    def test_already_numeric_series_unchanged(self):
       src = CSVDataSource(CSVOptions(path="x.csv"))
       s = pd.Series([1.0, 2.0, 3.0], dtype=np.float64)
       result = src._coerce_numeric(s)
       assert result.dtype == np.float64

    @pytest.mark.edge_case
    def test_empty_series_no_crash(self):
        src = CSVDataSource(CSVOptions(path="x.csv"))
        s = pd.Series([], dtype=object)
        result = src._coerce_numeric(s) 
        assert len(result) == 0


class TestHeaderHandling:

    @pytest.mark.positive
    def test_headers_lowercased_and_stripped(self):
        src = CSVDataSource(CSVOptions(path="x.csv"))
        df = pd.DataFrame(columns=["  Name ", "AGE", "  City"])
        result = src._header_handling(df)
        assert list(result.columns) == ["name", "age", "city"]
    
    @pytest.mark.negative
    def test_duplicate_columns_raise_value_error(self):
        src = CSVDataSource(CSVOptions(path="x.csv"))
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "a"])
        with pytest.raises(ValueError, match="Duplicate column names found: "):
            src._header_handling(df)

    @pytest.mark.edge_case
    def test_single_column(self):
        src = CSVDataSource(CSVOptions(path="x.csv"))
        df = pd.DataFrame(columns=[" Onecolumn "])
        result = src._header_handling(df)
        assert list(result.columns) == ["onecolumn"]



class TestLoad:

    @pytest.mark.positive
    def test_basic_numeric_csv_loads(self, csv_dir):
        path = make_csv(
            csv_dir,
            [["name", "age"], ["Alice", 30.5], ["Bob", 25.0]]
        )
        src = CSVDataSource(CSVOptions(path=path))
        df = src.load()

        assert list(df.columns) == ["name", "age"]
        assert len(df) == 2
        assert df["age"].dtype == np.float64
    
    @pytest.mark.positive
    def test_semicolon_separator(self, csv_dir):
        path = make_csv(
            csv_dir,
            [["a", "b"], [1, 2], [3, 4]],
            sep=";"
        )
        src = CSVDataSource(CSVOptions(path=path, sep=";"))
        df = src.load()
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2
    
    @pytest.mark.positive
    def test_skiprows_skips_metadata_row(self, csv_dir):
        path = make_csv(
            csv_dir,
            [["### metadata"], ["col_a", "col_b"], [1, 2], [3, 4]],
        )
        src = CSVDataSource(CSVOptions(path=path, skip_rows=1))
        df = src.load()
        assert list(df.columns) == ["col_a", "col_b"]
        assert len(df) == 2
    
    @pytest.mark.positive
    def test_na_values_become_nan(self, csv_dir):
        path = make_csv(
            csv_dir,
            [["val"], ["1"], ["N/A"], ["3"]],
        )

        src = CSVDataSource(CSVOptions(path=path, na_values=["N/A"]))
        df = src.load()
        assert df["val"].isna().sum() == 1
    
    @pytest.mark.positive
    def test_thousand_separator_parsed(self, csv_dir):
        path = make_csv(
            csv_dir,
            [["price"], ["1,000"], ["2,500"]]
        )
        src = CSVDataSource(CSVOptions(path=path, thousands=","))
        df = src.load()
        assert df["price"].iloc[0] == pytest.approx(1000.0)
        assert df["price"].iloc[1] == pytest.approx(2500.0)
    
    @pytest.mark.negative
    def test_load_missing_file_raises(self):
        src = CSVDataSource(CSVOptions(path="/no/file.csv"))
        with pytest.raises(FileNotFoundError):
            src.load()

    @pytest.mark.edge_case
    def test_csv_with_only_header_returns_empty_df(self, csv_dir):
        path = make_csv(
            csv_dir,
            [["col_a", "col_b"]]
        )

        src = CSVDataSource(CSVOptions(path=path))
        df = src.load()
        assert list(df.columns) == ["col_a", "col_b"]
        assert len(df) == 0
    
    @pytest.mark.edge_case
    def test_bom_csv_loads_without_garbage_column(self, csv_dir):
        raw = "name,age\nAlice,30\n".encode("utf-8-sig")
        path = make_raw_file(csv_dir, raw, filename="bom_load.csv")
        src = CSVDataSource(CSVOptions(path=path))
        df = src.load()
        assert "name" in df.columns
        assert not any(col.startswith("\ufeff") for col in df.columns)