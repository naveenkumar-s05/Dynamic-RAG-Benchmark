"""CSV parsing using pandas.

Each row is serialized as a self-describing ``column: value`` record so that no
value loses its column header. Row numbers are preserved as metadata so records
remain identifiable.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd  # type: ignore

from .base import PARSER_VERSION, new_block
from .models import ContentType, ParsedBlock, ParsedDocument

PARSER_NAME = "csv_parser"
MAX_ROWS_WARN = 20000


def _serialize_record(columns: List[str], values: List[str], row_number: int) -> str:
    pairs = [f"{col}: {val}" for col, val in zip(columns, values)]
    return f"row {row_number} -> " + " | ".join(pairs)


def parse(
    path: str | Path,
    *,
    source_id: str,
    source_name: str,
    document_version: str = "1",
    enable_ocr: bool = False,
) -> ParsedDocument:
    path = Path(path)
    warnings: List[str] = []
    blocks: List[ParsedBlock] = []
    index = 0

    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:  # noqa: BLE001 - surface clear error upstream
        raise ValueError(f"Failed to read CSV '{source_name}': {exc}") from exc

    columns = [str(c) for c in frame.columns]

    # A schema/caption block makes the column layout explicit.
    blocks.append(
        new_block(
            source_id=source_id,
            source_name=source_name,
            source_type="csv",
            content_type=ContentType.CAPTION,
            text="Columns: " + ", ".join(columns),
            index=index,
            parser_name=PARSER_NAME,
            section_path=[source_name],
            document_version=document_version,
            meta={"columns": columns},
        )
    )
    index += 1

    if len(frame) > MAX_ROWS_WARN:
        warnings.append(
            f"CSV has {len(frame)} rows; the first {MAX_ROWS_WARN} are indexed."
        )
        frame = frame.head(MAX_ROWS_WARN)

    for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
        values = [str(row[col]) for col in columns]
        blocks.append(
            new_block(
                source_id=source_id,
                source_name=source_name,
                source_type="csv",
                content_type=ContentType.RECORD,
                text=_serialize_record(columns, values, row_number),
                index=index,
                parser_name=PARSER_NAME,
                section_path=[source_name],
                row_number=row_number,
                document_version=document_version,
                meta={"columns": columns, "values": values},
            )
        )
        index += 1

    if frame.empty:
        warnings.append("The CSV file has a header but no data rows.")

    return ParsedDocument(
        source_id=source_id,
        source_name=source_name,
        source_type="csv",
        blocks=blocks,
        parser_name=PARSER_NAME,
        parser_version=PARSER_VERSION,
        document_version=document_version,
        source_path=str(path),
        warnings=warnings,
        metadata={"columns": columns, "row_count": int(len(frame))},
    )
