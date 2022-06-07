"""Merge two GTSFM reports for side-by-side comparison.

Authors: John Lambert
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tabulate import tabulate

SINGLE_REPORT_TABLES = Dict[str, Dict[str, Any]]
# contains info about each metric, from two separate reports
MERGED_REPORT_TABLES = Dict[str, List[Tuple[str, Any, Any]]]


def extract_tables_from_report(report_fpath: str) -> SINGLE_REPORT_TABLES:
    """Given an HTML text file containing HTML table info (and Plotly rendering code), rip out the table information
    for each table.

    Args:
        report_fpath: file path to a GTSFM HTML report.

    Returns:
        table_dict: Dictionary mapping the names of GTSFM modules to their associated table information.
            Each table is represented as a map from metric names to their associated values.
    """
    if not Path(report_fpath).exists():
        raise FileNotFoundError(f"HTML report missing. File path '{report_fpath}' does not exist.")

    with open(report_fpath, "r") as f:
        lines = f.readlines()

    start_token_tab_name = "<h2>"
    end_token_tab_name = "</h2>"

    old_start_token_tab_name = 'font-size:25px;font-family:Arial">'
    old_end_token_tab_name = "</p><table>"

    start_token_submetric = "<tr><td>"
    end_token_submetric = "</td></tr>"

    metric_html_boilerplate = '</td><td style="text-align: right;">'

    table_dict = defaultdict(dict)

    for line in lines:
        # check to see if the start of a new table
        s = line.find(start_token_tab_name)
        if s != -1:
            e = line.find(end_token_tab_name)
            curr_tab_name = line[s + len(start_token_tab_name) : e]
        else:
            s = line.find(old_start_token_tab_name)
            if s != -1:
                e = line.find(old_end_token_tab_name)
                curr_tab_name = line[s + len(old_start_token_tab_name) : e]

        # check to see if is a row of a table, containing info about one metric.
        s = line.find(start_token_submetric)
        if s != -1:
            e = line.find(end_token_submetric)
            metric_info = line[s + len(start_token_submetric) : e]

            # break apart the metric name and the associated value
            metric_name, metric_val = metric_info.split(metric_html_boilerplate)
            # strip whitepace
            metric_name = metric_name.strip()
            metric_val = metric_val.strip()
            table_dict[curr_tab_name][metric_name] = metric_val

    return table_dict


def merge_tables(tables_dict1: SINGLE_REPORT_TABLES, tables_dict2: SINGLE_REPORT_TABLES) -> MERGED_REPORT_TABLES:
    """Combine 2-column tables from single reports, to 3-column tables, with columns corresponding to each report."""

    new_table_dict = defaultdict(list)

    for tab_name, tab_metrics1 in tables_dict1.items():
        for metric_name, tab_val1 in tab_metrics1.items():
            tab_val2 = tables_dict2[tab_name].get(metric_name, None)
            new_table_dict[tab_name] += [(metric_name, tab_val1, tab_val2)]

    return new_table_dict


def print_tables_tabulate(merged_tables_dict: MERGED_REPORT_TABLES, table_format: str) -> None:
    """Dump tables to stdout in a Markdown or HTML format."""
    for tab_name, tab_metrics in merged_tables_dict.items():

        headers = [f"{tab_name}", "Report 1", "Report 2"]
        table = tab_metrics
        print(tabulate(table, headers, tablefmt=table_format))
        # add newline between tables
        print()


def merge_reports(report1_fpath: str, report2_fpath: str, output_format: str) -> None:
    """Combine all tables from two reports into a single table, for side-by-side comparisons."""
    tables_dict1 = extract_tables_from_report(report1_fpath)
    tables_dict2 = extract_tables_from_report(report2_fpath)
    merged_tables_dict = merge_tables(tables_dict1, tables_dict2)
    print_tables_tabulate(merged_tables_dict, output_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report1_fpath", required=True, help="Path to previous report.")
    parser.add_argument("--report2_fpath", required=True, help="Path to new report.")
    parser.add_argument(
        "--table_format",
        default="github",
        choices=["github", "html"],
        help="Output format for exported table (markdown or HTML).",
    )
    args = parser.parse_args()
    merge_reports(args.report1_fpath, args.report2_fpath, args.table_format)
