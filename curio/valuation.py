"""Explainable price estimates.

Every estimate is built from the price observations in the pack and comes with the full working:
which observations were used, how each was adjusted for grade and age, what weight it got, and
which were thrown out and why. Nothing is fetched from the network and there is no hidden model.

Method
  1. Keep observations in the pack's currency (no silent FX conversion).
  2. Grade-adjust each price to the target grade: price x mult[target] / mult[observed grade].
  3. Weight = kind weight (realised sale > guide > asking price) x recency (half-life decay)
     x penalty if the grade could not be adjusted.
  4. Drop outliers more than 3 robust standard deviations (MAD, on log price) from the weighted median.
  5. Estimate = weighted median; range = weighted 20th-80th percentile.
  6. Confidence from effective sample size, spread, and age of the newest realised sale.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np

from .pack import Item, Pack

KIND_WEIGHT = {"sale": 1.0, "guide": 0.5, "ask": 0.35}
KIND_LABEL = {"sale": "realised sale", "guide": "catalog/guide value", "ask": "asking price"}
UNADJUSTED_GRADE_PENALTY = 0.3
MIN_LOG_SPREAD = 0.1


@dataclass
class CompLine:
    source: str
    kind: str
    date: str | None
    grade: str | None
    price: float
    grade_factor: float | None
    age_years: float | None
    weight: float
    adjusted_price: float
    used: bool
    note: str = ""
    url: str = ""


@dataclass
class Valuation:
    item_key: str
    currency: str
    target_grade: str | None
    estimate: float | None
    low: float | None
    high: float | None
    confidence: str  # high / medium / low / none
    effective_samples: float
    comps: list[CompLine] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return float(np.interp(q, cw, v))


def _money(x: float, cur: str) -> str:
    return f"{x:,.2f} {cur}" if x < 100 else f"{x:,.0f} {cur}"


def estimate_value(item: Item, pack: Pack, target_grade: str | None = None, as_of: date | None = None,
                   half_life_years: float = 1.5) -> Valuation:
    as_of = as_of or date.today()
    cur = pack.currency
    mults = pack.grade_multipliers
    if target_grade is not None and pack.grades and target_grade not in pack.grades:
        raise ValueError(f"unknown grade '{target_grade}' for pack {pack.id}")

    explanation = [
        f"Based on {len(item.prices)} price observation(s) recorded in the '{pack.name}' pack "
        f"(v{pack.version}); no online lookup or hidden model is involved."
    ]
    if target_grade:
        explanation.append(f"Prices are adjusted to grade {target_grade} using the pack's grade multipliers.")
    else:
        explanation.append("No grade chosen, so prices are not grade-adjusted; the range reflects mixed grades. "
                           "Pick a grade for a tighter estimate.")

    comps: list[CompLine] = []
    for p in item.prices:
        d = _parse_date(p.date)
        age = (as_of - d).days / 365.25 if d else None
        line = CompLine(source=p.source, kind=p.kind, date=p.date, grade=p.grade, price=p.value,
                        grade_factor=None, age_years=round(age, 2) if age is not None else None,
                        weight=0.0, adjusted_price=p.value, used=True, url=p.url)
        if p.currency != cur:
            line.used, line.note = False, f"in {p.currency}, pack currency is {cur} (no FX conversion offline)"
            comps.append(line)
            continue

        w = KIND_WEIGHT[p.kind]
        notes = []
        if target_grade:
            if p.grade == target_grade:
                line.grade_factor = 1.0
            elif p.grade in mults and target_grade in mults:
                line.grade_factor = mults[target_grade] / mults[p.grade]
                line.adjusted_price = p.value * line.grade_factor
                notes.append(f"x{line.grade_factor:.2f} for {p.grade} -> {target_grade}")
            else:
                w *= UNADJUSTED_GRADE_PENALTY
                notes.append("grade unknown or no multiplier; not adjusted, down-weighted")
        if age is None:
            w *= 0.5
            notes.append("undated; down-weighted")
        else:
            w *= 0.5 ** (max(age, 0.0) / half_life_years)
        line.weight = round(w, 4)
        line.note = "; ".join(notes)
        comps.append(line)

    usable = [c for c in comps if c.used and c.weight > 0]
    if not usable:
        explanation.append("There are no usable price observations for this item yet, so no estimate is given. "
                           "Contribute sold prices to the pack to enable one.")
        return Valuation(item.key, cur, target_grade, None, None, None, "none", 0.0, comps, explanation)

    # Outlier rejection on log price.
    logp = np.log([c.adjusted_price for c in usable])
    wts = np.array([c.weight for c in usable])
    if len(usable) >= 4:
        med = _weighted_quantile(logp, wts, 0.5)
        # Floor at 0.1 (~10%) so a tight cluster of sales doesn't make ordinary variation look extreme.
        mad = max(float(np.median(np.abs(logp - med))) * 1.4826, MIN_LOG_SPREAD)
        if mad > 0:
            for c, lp in zip(usable, logp):
                z = abs(lp - med) / mad
                if z > 3:
                    c.used = False
                    c.note = (c.note + "; " if c.note else "") + f"outlier ({z:.1f} robust SDs from median), excluded"
        usable = [c for c in usable if c.used]
        logp = np.log([c.adjusted_price for c in usable])
        wts = np.array([c.weight for c in usable])

    prices = np.exp(logp)
    est = _weighted_quantile(prices, wts, 0.5)
    low, high = _weighted_quantile(prices, wts, 0.2), _weighted_quantile(prices, wts, 0.8)
    n_eff = float(wts.sum() ** 2 / (wts ** 2).sum())
    spread = high / low if low > 0 else math.inf
    sale_ages = [c.age_years for c in usable if c.kind == "sale" and c.age_years is not None]
    newest_sale = min(sale_ages) if sale_ages else None

    if n_eff >= 4 and spread < 1.8 and newest_sale is not None and newest_sale < 2:
        confidence = "high"
    elif n_eff >= 2 and newest_sale is not None:
        confidence = "medium"
    else:
        confidence = "low"

    kinds = {k: sum(1 for c in usable if c.kind == k) for k in KIND_WEIGHT}
    excluded = [c for c in comps if not c.used]
    explanation.append(
        "Used " + ", ".join(f"{n} {KIND_LABEL[k]}{'s' if n != 1 else ''}" for k, n in kinds.items() if n)
        + f". Realised sales count fully, guide values {KIND_WEIGHT['guide']:.0%}, asking prices "
          f"{KIND_WEIGHT['ask']:.0%} (asks tend to overstate what things sell for)."
    )
    explanation.append(f"Older observations count less: weight halves every {half_life_years:g} years.")
    if excluded:
        explanation.append(f"Excluded {len(excluded)} observation(s); see the notes in the table.")
    explanation.append(
        f"Estimate is the weighted median ({_money(est, cur)}); the range is the weighted 20th-80th "
        f"percentile ({_money(low, cur)} - {_money(high, cur)})."
    )
    reason = [f"effective sample size {n_eff:.1f}", f"high/low ratio {spread:.2f}"]
    reason.append(f"newest realised sale {newest_sale:.1f} years old" if newest_sale is not None
                  else "no dated realised sales")
    explanation.append(f"Confidence is {confidence}: " + ", ".join(reason) + ".")

    return Valuation(item.key, cur, target_grade, round(est, 2), round(low, 2), round(high, 2),
                     confidence, round(n_eff, 2), comps, explanation)


def value_by_grade(item: Item, pack: Pack, as_of: date | None = None) -> list[dict]:
    """Estimate at every grade on the pack's scale (only grades with multipliers can be adjusted to)."""
    rows = []
    for g in pack.grades:
        if g not in pack.grade_multipliers:
            continue
        v = estimate_value(item, pack, g, as_of)
        rows.append({"grade": g, "estimate": v.estimate, "low": v.low, "high": v.high, "confidence": v.confidence})
    return rows
