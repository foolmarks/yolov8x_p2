"""
**************************************************************************
||                        SiMa.ai CONFIDENTIAL                          ||
||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
**************************************************************************
 NOTICE:  All information contained herein is, and remains the property of
 SiMa.ai. The intellectual and technical concepts contained herein are
 proprietary to SiMa and may be covered by U.S. and Foreign Patents,
 patents in process, and are protected by trade secret or copyright law.

 Dissemination of this information or reproduction of this material is
 strictly forbidden unless prior written permission is obtained from
 SiMa.ai.  Access to the source code contained herein is hereby forbidden
 to anyone except current SiMa.ai employees, managers or contractors who
 have executed Confidentiality and Non-disclosure agreements explicitly
 covering such access.

 The copyright notice above does not evidence any actual or intended
 publication or disclosure  of  this source code, which includes information
 that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.

 ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
 DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
 CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
 LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
 CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
 REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
 SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.

**************************************************************************
"""

# Generate a tabular layer-error report from a SiMa ModelSDK JSON export.
# Created: 2026-03-25

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_JSON_PATH = PROJECT_DIR / "build" / "yolov8x-p2_opt_4o" / "yolov8x-p2_opt_4o.sima.json"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "build" / "yolov8x-p2_opt_4o" / "yolov8x-p2_opt_4o_layer_errors.txt"
TABLE_HEADERS = ("Operator", "name", "error_value")
LayerRow = tuple[str, str, str]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the error analysis utility."""

    parser = argparse.ArgumentParser(
        description="Extract layer names and error values from a SiMa JSON export."
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help=f"Path to the input SiMa JSON file (default: {DEFAULT_JSON_PATH}).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to the output text report (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--sort-by-error-value",
        action="store_true",
        help="Sort rows by error_value from greatest to smallest. Missing values are listed last.",
    )
    return parser.parse_args()


def load_layers(json_path: Path) -> list[dict[str, Any]]:
    """Load the SiMa JSON file and return its layer list."""

    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    try:
        layers = payload["modelTopology"]["model_config"]["config"]["layers"]
    except KeyError as exc:
        raise SystemExit(
            f"Could not find the layer list in {json_path}. Missing key: {exc}"
        ) from exc

    if not isinstance(layers, list):
        raise SystemExit(f"Expected 'layers' to be a list in {json_path}.")

    return layers


def extract_layer_rows(layers: list[dict[str, Any]]) -> list[LayerRow]:
    """Extract class name, layer name, and optional error value for each layer."""

    rows: list[LayerRow] = []

    for layer in layers:
        class_name = str(layer.get("class_name", ""))
        name = str(layer.get("name", ""))
        config = layer.get("config", {})
        layer_stats = config.get("Layer Statistics", {}) if isinstance(config, dict) else {}
        error_value = layer_stats.get("error_value") if isinstance(layer_stats, dict) else None
        error_text = "" if error_value is None else str(error_value)
        rows.append((class_name, name, error_text))

    return rows


def sort_rows_by_error_value(rows: list[LayerRow]) -> list[LayerRow]:
    """Sort rows by descending error value, placing missing values at the end."""

    def sort_key(row: LayerRow) -> tuple[int, float]:
        """Build a stable sort key for descending error-value ordering."""

        error_text = row[2]
        if not error_text:
            return (1, 0.0)

        try:
            return (0, -float(error_text))
        except ValueError:
            return (1, 0.0)

    return sorted(rows, key=sort_key)


def format_table(rows: list[LayerRow]) -> str:
    """Format layer rows as a fixed-width text table."""

    widths = [len(header) for header in TABLE_HEADERS]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    line_format = f"{{:<{widths[0]}}}  {{:<{widths[1]}}}  {{:<{widths[2]}}}"
    lines = [
        line_format.format(*TABLE_HEADERS),
        line_format.format(*("-" * widths[0], "-" * widths[1], "-" * widths[2])),
    ]
    lines.extend(line_format.format(*row) for row in rows)
    return "\n".join(lines) + "\n"


def write_report(output_path: Path, report_text: str) -> None:
    """Write the formatted report to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")


def main() -> None:
    """Generate the layer error report from the requested SiMa JSON file."""

    args = parse_args()
    layers = load_layers(args.json_path)
    rows = extract_layer_rows(layers)
    if args.sort_by_error_value:
        rows = sort_rows_by_error_value(rows)

    report_text = format_table(rows)
    write_report(args.output_path, report_text)
    print(f"Wrote {len(layers)} layer rows to {args.output_path}")


if __name__ == "__main__":
    main()
