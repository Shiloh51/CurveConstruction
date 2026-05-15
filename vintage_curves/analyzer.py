"""Public façade: VintageAnalyzer and SegmentResult."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

from vintage_curves.aggregate import Weighting, aggregate_curves
from vintage_curves.default_policy import DefaultPolicy
from vintage_curves.metrics import VintageMetrics
from vintage_curves.mob import VintageFreq
from vintage_curves.output import to_csv_dir, to_excel
from vintage_curves.plotting import plot_curves
from vintage_curves.rolls import WeightMode, transition_matrix
from vintage_curves.schema import SchemaMapper
from vintage_curves.segment import Segment
from vintage_curves.summary import vintage_summary
from vintage_curves.tape import LoanTape


class VintageAnalyzer:
    """Top-level entry point. Holds a tape, default policy, and vintage frequency."""

    def __init__(
        self,
        tape: LoanTape,
        vintage_freq: VintageFreq = "Q",
        default_policy: DefaultPolicy | None = None,
    ) -> None:
        self.tape = tape
        self.vintage_freq = vintage_freq
        self.default_policy = default_policy or DefaultPolicy.charge_off_field()

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        schema: str | Path | SchemaMapper,
        vintage_freq: VintageFreq = "Q",
        default_policy: DefaultPolicy | None = None,
        **read_csv_kwargs,
    ) -> "VintageAnalyzer":
        tape = LoanTape.from_csv(path, schema=schema, **read_csv_kwargs)
        return cls(tape, vintage_freq=vintage_freq, default_policy=default_policy)

    def segment(self, query: str | None, label: str = "segment") -> "SegmentResult":
        seg = Segment.from_tape(self.tape, query=query, label=label)
        return SegmentResult(
            segment=seg,
            default_policy=self.default_policy,
            vintage_freq=self.vintage_freq,
        )

    def all(self, label: str = "all") -> "SegmentResult":
        """Convenience: segment with no filter."""
        return self.segment(query=None, label=label)


class SegmentResult:
    """Curves, aggregates, summary, rolls, plotting, and export for one segment."""

    def __init__(
        self,
        segment: Segment,
        default_policy: DefaultPolicy,
        vintage_freq: VintageFreq,
    ) -> None:
        self.segment = segment
        self.metrics = VintageMetrics.from_segment(
            segment, default_policy=default_policy, vintage_freq=vintage_freq
        )

    @property
    def label(self) -> str:
        return self.segment.label

    # ----- curves --------------------------------------------------------

    def curve(self, name: str) -> pd.DataFrame:
        return self.metrics.curve(name)

    def aggregate(
        self,
        metrics: Iterable[str],
        weighting: Weighting = "original",
    ) -> pd.DataFrame:
        return aggregate_curves(self.metrics, metrics, weighting=weighting)

    def summary(self) -> pd.DataFrame:
        return vintage_summary(self.metrics)

    def roll_rates(
        self,
        by_mob: bool = False,
        weight: WeightMode = "count",
    ) -> pd.DataFrame:
        return transition_matrix(
            self.metrics.perf_flagged, weight=weight, per_mob=by_mob
        )

    # ----- plotting ------------------------------------------------------

    def plot(
        self,
        name: str,
        include_agg: bool = False,
        weighting: Weighting = "original",
    ):
        per_v = self.curve(name)
        agg = None
        if include_agg:
            agg = self.aggregate([name], weighting=weighting)[name]
        return plot_curves(
            per_v,
            title=f"{name} — {self.label}",
            ylabel=name,
            aggregate=agg,
            aggregate_label=f"agg ({weighting})",
        )

    # ----- export --------------------------------------------------------

    def export_excel(
        self,
        path: str | Path,
        metrics: Iterable[str] | None = None,
        include_aggregates: bool = True,
        include_rolls: bool = True,
    ) -> Path:
        names = list(metrics) if metrics is not None else self.metrics.available_metrics()
        curves = {n: self.curve(n) for n in names}
        aggs = None
        if include_aggregates:
            aggs = {
                w: self.aggregate(names, weighting=w)
                for w in ("original", "beginning", "equal")
            }
        rolls = self.roll_rates() if include_rolls else None
        return to_excel(
            path,
            curves=curves,
            summary=self.summary(),
            rolls=rolls,
            aggregates=aggs,
        )

    def export_csv_dir(
        self,
        out_dir: str | Path,
        metrics: Iterable[str] | None = None,
        include_aggregates: bool = True,
        include_rolls: bool = True,
    ) -> Path:
        names = list(metrics) if metrics is not None else self.metrics.available_metrics()
        curves = {n: self.curve(n) for n in names}
        aggs = None
        if include_aggregates:
            aggs = {
                w: self.aggregate(names, weighting=w)
                for w in ("original", "beginning", "equal")
            }
        rolls = self.roll_rates() if include_rolls else None
        return to_csv_dir(
            out_dir,
            curves=curves,
            summary=self.summary(),
            rolls=rolls,
            aggregates=aggs,
        )
