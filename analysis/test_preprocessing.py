import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Support imports regardless of whether tests are run from repository root or analysis dir
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
for p in (str(project_root), str(current_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from analysis.preprocessing import clean_text, load_dataset
except ImportError:
    from preprocessing import clean_text, load_dataset


def _call_load_dataset(filepath):
    """Helper to invoke load_dataset with positional or keyword argument."""
    try:
        return load_dataset(str(filepath))
    except TypeError:
        return load_dataset(filepath=str(filepath))


class TestCleanTextMissingValues:
    """Tests for clean_text handling of missing, empty, and non-string values."""

    def test_clean_text_none(self):
        result = clean_text(None)
        assert isinstance(result, str)
        assert result == ""

    def test_clean_text_nan(self):
        result = clean_text(float("nan"))
        assert isinstance(result, str)
        assert result == ""

    def test_clean_text_numpy_nan(self):
        result = clean_text(np.nan)
        assert isinstance(result, str)
        assert result == ""

    def test_clean_text_empty_string(self):
        assert clean_text("") == ""

    def test_clean_text_whitespace_only(self):
        assert clean_text("   \t\n\r  ") == ""

    def test_clean_text_numeric_and_bool_typecasting(self):
        # Non-string scalar types should be cast without throwing TypeError
        res_int = clean_text(12345)
        assert isinstance(res_int, str)
        assert "12345" in res_int

        res_float = clean_text(99.9)
        assert isinstance(res_float, str)
        assert "99" in res_float

        res_bool = clean_text(False)
        assert isinstance(res_bool, str)


class TestCleanTextForeignCharacters:
    """Tests for clean_text handling of international scripts, accents, and emojis."""

    def test_clean_text_accented_latin_characters(self):
        text = "Café crème brûlée résumé naïve façade"
        result = clean_text(text)
        assert isinstance(result, str)
        assert len(result) > 0
        # Valid content must not be discarded
        assert any(term in result.lower() for term in ["cafe", "café", "creme", "crème", "facade", "façade"])

    def test_clean_text_cyrillic_script(self):
        text = "Срочно! Вы выиграли приз в лотерею. Заберите выигрыш."
        result = clean_text(text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_clean_text_cjk_characters(self):
        text = "恭喜您获得特等奖，请立即点击链接领取奖励"
        result = clean_text(text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_clean_text_arabic_script(self):
        text = "رسالة عاجلة: لقد فزت بجائزة مالية كبرى اتصل الآن"
        result = clean_text(text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_clean_text_emojis_and_currency_symbols(self):
        text = "WINNER! You won £1000 or 500€ cash now! 💰🎁 Call 09061743810!"
        result = clean_text(text)
        assert isinstance(result, str)
        # Core message content must be retained
        lower_res = result.lower()
        assert "winner" in lower_res
        assert "cash" in lower_res
        assert "call" in lower_res


class TestCleanTextCorruptedEmailFormats:
    """Tests for clean_text handling of corrupted email headers, HTML markup, and malformed text."""

    def test_clean_text_email_headers_retains_payload(self):
        raw_email = (
            "From: promo@spam-domain.com\n"
            "To: recipient@example.com\n"
            "Subject: Urgent Notification!\n"
            "Date: Wed, 21 Oct 2020 07:28:00 -0000\n"
            "Content-Type: text/plain; charset=utf-8\n\n"
            "Dear user, your account requires immediate attention. Click here to verify."
        )
        result = clean_text(raw_email)
        assert isinstance(result, str)
        lower_res = result.lower()
        # Valid message payload should not be dropped
        assert "account" in lower_res
        assert "verify" in lower_res or "attention" in lower_res

    def test_clean_text_malformed_html(self):
        malformed_html = "<div class='ad'><p>Special <b>Discount</b>! Claim <a href='http://bad.link'>NOW</div>"
        result = clean_text(malformed_html)
        assert isinstance(result, str)
        lower_res = result.lower()
        assert "discount" in lower_res
        assert "claim" in lower_res
        assert "<div" not in lower_res
        assert "</div>" not in lower_res

    def test_clean_text_html_entities(self):
        text = "Save 50&percnt; &amp; get free shipping &lt;limited offer&gt; &quot;deal&quot;"
        result = clean_text(text)
        assert isinstance(result, str)
        lower_res = result.lower()
        assert "save" in lower_res
        assert "shipping" in lower_res
        assert "offer" in lower_res

    def test_clean_text_control_characters_and_irregular_spacing(self):
        text = "Important\r\n\t\t\tSecurity   Update\x00\x01\x02\x1fPlease review your settings."
        result = clean_text(text)
        assert isinstance(result, str)
        lower_res = result.lower()
        assert "important" in lower_res
        assert "security" in lower_res
        assert "review" in lower_res
        assert "\x00" not in result

    def test_clean_text_broken_urls(self):
        text = "Visit http://example.com/verify?id=123&token=abc&&&amp;ref=mail for details"
        result = clean_text(text)
        assert isinstance(result, str)
        lower_res = result.lower()
        assert "visit" in lower_res
        assert "details" in lower_res


class TestLoadDataset:
    """Tests for load_dataset handling fallback encodings, missing values, and corrupted data."""

    def test_load_dataset_utf8(self, tmp_path):
        csv_content = "label,text\nspam,Congratulations you won a car\nham,Are we still meeting at 5?\n"
        file_path = tmp_path / "utf8_dataset.csv"
        file_path.write_text(csv_content, encoding="utf-8")

        data = _call_load_dataset(file_path)
        assert data is not None
        if isinstance(data, pd.DataFrame):
            assert len(data) == 2
        elif isinstance(data, tuple):
            assert len(data[0]) == 2

    def test_load_dataset_fallback_latin1_encoding(self, tmp_path):
        # Latin-1 characters such as £ (pound symbol) commonly present in SMS spam datasets
        csv_content = (
            "v1,v2,label,text\n"
            "spam,Win a £1000 cash prize at the café!,spam,Win a £1000 cash prize at the café!\n"
            "ham,Meeting at the café tomorrow morning,ham,Meeting at the café tomorrow morning\n"
        )
        file_path = tmp_path / "latin1_dataset.csv"
        file_path.write_bytes(csv_content.encode("latin-1"))

        data = _call_load_dataset(file_path)
        assert data is not None
        if isinstance(data, pd.DataFrame):
            assert len(data) == 2
            # Verify valid data loaded without dropping valid rows
            assert not data.empty

    def test_load_dataset_fallback_cp1252_encoding(self, tmp_path):
        # Windows-1252 specific byte sequence (e.g. smart quote or euro symbol \x80)
        csv_content = (
            "label,text\n"
            "spam,Special €500 bonus for you!\n"
            "ham,See you soon!\n"
        )
        file_path = tmp_path / "cp1252_dataset.csv"
        file_path.write_bytes(csv_content.encode("cp1252"))

        data = _call_load_dataset(file_path)
        assert data is not None
        if isinstance(data, pd.DataFrame):
            assert len(data) >= 1

    def test_load_dataset_missing_values_and_type_casting(self, tmp_path):
        # Dataset with empty/missing message fields and valid rows
        csv_content = (
            "label,text\n"
            "spam,Claim your free coupon now\n"
            "ham,\n"
            "spam,Urgent message regarding your account\n"
        )
        file_path = tmp_path / "missing_values.csv"
        file_path.write_text(csv_content, encoding="utf-8")

        data = _call_load_dataset(file_path)
        assert data is not None
        if isinstance(data, pd.DataFrame):
            text_col = "text" if "text" in data.columns else data.columns[1]
            # Type-casting logic should ensure texts are strings and valid rows not dropped
            for val in data[text_col]:
                assert isinstance(val, str) or pd.isna(val)
            # Valid records must be preserved
            assert len(data) >= 2

    def test_load_dataset_corrupted_rows_preserves_valid_data(self, tmp_path):
        csv_content = (
            "label,text\n\n"
            "spam,Special cash rewards await\n\n"
            "ham,Can you send the document?\n\n"
        )
        file_path = tmp_path / "corrupted_blank_lines.csv"
        file_path.write_text(csv_content, encoding="utf-8")

        data = _call_load_dataset(file_path)
        assert data is not None
        if isinstance(data, pd.DataFrame):
            assert len(data) >= 2
