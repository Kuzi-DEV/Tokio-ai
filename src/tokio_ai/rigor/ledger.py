"""Session-level bookkeeping so an agent conversation can't p-hack itself.

Every hypothesis test run in a session gets recorded here. Verdicts are
issued relative to everything tested so far in the SAME session -- the 5th
signal tested in one sitting needs a stronger raw p-value than the 1st to
count as significant, which is exactly the discipline a naive
"just check p<0.05 every time" agent would skip.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .provenance import stamp
from .stats import PermutationResult, benjamini_hochberg


@dataclass
class RecordedTest:
    name: str
    result: PermutationResult


@dataclass
class TestLedger:
    __test__ = False  # not a pytest test class despite the name

    tests: list[RecordedTest] = field(default_factory=list)

    def record(self, name: str, result: PermutationResult) -> None:
        self.tests.append(RecordedTest(name, result))

    def significant_at(self, index: int, alpha: float = 0.05) -> bool:
        """BH-corrected significance of the test at `index`, against every
        test recorded so far. A later test can change the answer -- the
        correction only knows about the tests it has already seen."""
        return benjamini_hochberg([t.result.p_value for t in self.tests], alpha=alpha)[index]

    def verdict(self, name: str, alpha: float = 0.05) -> str:
        p_values = [t.result.p_value for t in self.tests]
        sig = benjamini_hochberg(p_values, alpha=alpha)
        # Walk backwards so a reused name reports its MOST RECENT test, not
        # a stale first result -- every entry (including earlier duplicates)
        # still counts toward the correction denominator either way.
        for i in range(len(self.tests) - 1, -1, -1):
            t = self.tests[i]
            if t.name != name:
                continue
            if not t.result.meets_min_sample:
                return (
                    f"NOT REPORTABLE: n={t.result.n_a}/{t.result.n_b} is below the "
                    f"MIN_SAMPLE floor -- sample too small to say anything."
                )
            corrected = "SIGNIFICANT" if sig[i] else "NOT SIGNIFICANT"
            # The method is part of the verdict, not a footnote: a p-value
            # from a studentized rotation test and one from a naive shuffled
            # difference are not comparable numbers, and the agent quoting
            # this line should be able to say which it has.
            detail = (
                f"{corrected} after correcting for {len(self.tests)} hypothesis "
                f"test(s) run this session (raw p={t.result.p_value:.4f}, "
                f"gap={t.result.observed_gap:+.4%}, "
                f"method={t.result.statistic}, "
                f"reproducible with seed={t.result.seed} iters={t.result.iters})"
            )
            if t.result.variance_ratio is not None and (
                t.result.variance_ratio > 1.5 or t.result.variance_ratio < 1 / 1.5
            ):
                # Direction matters for the explanation: a ratio below 1 means
                # the condition picks CALM days, and calling those "volatile"
                # would hand the agent a false sentence to repeat.
                # Likewise the bias: pooling overstates significance only when
                # the smaller group is the noisier one.
                vr = t.result.variance_ratio
                selects = "volatile" if vr > 1 else "calm"
                smaller_is_noisier = (t.result.n_a < t.result.n_b) == (vr > 1)
                bias = "overstated" if smaller_is_noisier else "understated"
                detail += (
                    f". Note: the two groups have quite different variances "
                    f"(ratio {vr:.2f}x) -- the condition is selecting unusually "
                    f"{selects} days, so a test on the raw mean difference would "
                    f"have {bias} this result"
                )
            return detail + "."
        raise KeyError(f"no test recorded under name {name!r}")

    def summary(self) -> str:
        if not self.tests:
            return "No hypotheses tested yet."
        p_values = [t.result.p_value for t in self.tests]
        sig = benjamini_hochberg(p_values, alpha=0.05)
        lines = [f"{len(self.tests)} hypothesis test(s) this session (BH-corrected, alpha=0.05):"]
        for t, s in zip(self.tests, sig):
            flag = "SIGNIFICANT" if s else "not significant"
            lines.append(f"  {t.name}: p={t.result.p_value:.4f} gap={t.result.observed_gap:+.4%} -> {flag}")
        lines.append(f"[{stamp(agent=True)}]")
        return "\n".join(lines)
