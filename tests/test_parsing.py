import pytest
from abc import ABC
from adaptive_chunking.parsing import BaseParser, AzureDIParser, DoclingParser, PyMuPDFParser, ExcelParser


class TestBaseParser:
    def test_is_abstract(self):
        assert issubclass(BaseParser, ABC)
        with pytest.raises(TypeError):
            BaseParser()

    def test_azure_di_parser_is_subclass(self):
        assert issubclass(AzureDIParser, BaseParser)

    def test_docling_parser_is_subclass(self):
        assert issubclass(DoclingParser, BaseParser)

    def test_pymupdf_parser_is_subclass(self):
        assert issubclass(PyMuPDFParser, BaseParser)

    def test_excel_parser_is_subclass(self):
        assert issubclass(ExcelParser, BaseParser)

    def test_azure_di_missing_dep_message(self):
        """AzureDIParser should give a clear error if Azure SDK is not installed."""
        try:
            parser = AzureDIParser(endpoint="https://fake", key="fake")
        except ImportError as e:
            assert "adaptive-chunking[parsing]" in str(e)
        except Exception:
            # If Azure SDK IS installed, it may fail with a connection error — that's fine
            pass

    def test_docling_missing_dep_message(self):
        """DoclingParser should give a clear error if docling is not installed."""
        try:
            parser = DoclingParser()
        except ImportError as e:
            assert "adaptive-chunking[parsing]" in str(e)
        except Exception:
            # If docling IS installed, constructor succeeds — that's fine
            pass

    def test_pymupdf_missing_dep_message(self):
        """PyMuPDFParser should give a clear error if pymupdf4llm is not installed."""
        try:
            parser = PyMuPDFParser()
        except ImportError as e:
            assert "adaptive-chunking[parsing]" in str(e)
        except Exception:
            # If pymupdf4llm IS installed, constructor succeeds — that's fine
            pass


class TestDoclingParserHelpers:
    def test_split_table_markdown_small(self):
        """Small tables should not be split."""
        try:
            parser = DoclingParser()
        except ImportError:
            pytest.skip("docling not installed")

        table_md = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = parser._split_table_markdown(table_md)
        assert len(result) == 1
        assert result[0] == table_md
