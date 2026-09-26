"""Collects data-quality decisions and merge row counts for the data_quality_log / merge_row_counts sheets."""
import pandas as pd


class QualityLog:
    def __init__(self, verbose=True):
        self.logs, self.counts, self.verbose = [], [], verbose

    def log(self, step, issue, action, rows_affected=None, detail=""):
        self.logs.append(dict(step=step, issue=issue, action=action,
                              rows_affected=rows_affected, detail=detail))

    def count(self, step, before, after, note=""):
        self.counts.append(dict(step=step, rows_before=before, rows_after=after,
                                delta=after - before if before is not None else None, note=note))
        if self.verbose:
            print(f"  [{step}] {before} -> {after}  {note}")

    def log_frame(self):
        return pd.DataFrame(self.logs)

    def count_frame(self):
        return pd.DataFrame(self.counts)
