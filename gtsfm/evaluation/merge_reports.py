"""Merge two GTSFM reports for side-by-side comparison.

Authors: John Lambert
"""

import argparse
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from tabulate import tabulate

GTSFM_REPORT_TABLES = Dict[str, Dict[str, Any]]
# contains info about each metric, from two separate reports
GTSFM_MERGED_REPORT_TABLES = Dict[str, List[Tuple[str, Any, Any]]]


def extract_tables_from_report(report_fpath: str) -> GTSFM_REPORT_TABLES:
    """ """
    with open(report_fpath, "r") as f:
        lines = f.readlines()

    start_token_tab_name = 'font-size:25px;font-family:Arial">'
    end_token_tab_name = "</p><table>"

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


def merge_tables(table_dict1: GTSFM_REPORT_TABLES, table_dict2: GTSFM_REPORT_TABLES) -> GTSFM_MERGED_REPORT_TABLES:
    """Combine 2-column tables from single reports, to 3-column tables, with columns corresponding to each report."""

    new_table_dict = defaultdict(list)

    for tab_name, tab_metrics1 in table_dict1.items():
        for tab_name1, tab_val1 in tab_metrics1.items():
            tab_val2 = table_dict2[tab_name].get(tab_name1, None)
            new_table_dict[tab_name] += [(tab_name1, tab_val1, tab_val2)]

    return new_table_dict


def print_tables_tabulate(merged_table_dict: GTSFM_MERGED_REPORT_TABLES, table_format: str) -> None:
    """Dump to stdout Markdown-formatted tables."""
    for tab_name, tab_metrics in merged_table_dict.items():

        headers = [f"{tab_name}", "Report 1", "Report 2"]
        table = tab_metrics
        print(tabulate(table, headers, tablefmt=table_format))
        # add newline between tables
        print()


def merge_reports(report_fpath1: str, report_fpath2: str, output_format: str) -> None:
    """Combine all tables from two reports into a single table, for side-by-side comparisons."""
    table_dict1 = extract_tables_from_report(report_fpath1)
    table_dict2 = extract_tables_from_report(report_fpath2)
    merged_table_dict = merge_tables(table_dict1, table_dict2)
    print_tables_tabulate(merged_table_dict, output_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_fpath1", required=True, help="Path to previous report.")
    parser.add_argument("--report_fpath2", required=True, help="Path to new report.")
    parser.add_argument("--table_format", default="github", choices=["github", "html"], help="Output format for exported table (markdown or HTML).")
    args = parser.parse_args()
    merge_reports(args.report_fpath1, args.report_fpath2, args.table_format)
