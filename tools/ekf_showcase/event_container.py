"""Event-level casualty container with child-location conservation.

Disaster reporting attaches figures to an EVENT first and to sub-regional places
second. Early and headline reports routinely omit the place ("at least 44 killed
in the storm", "quake death toll rises to 109"), while multi-region summaries
name several places in one passage so an extractor can bind the event-wide total
to one child state ("250 dead across six states ... in Tennessee").

Keying tracks purely by `location` creates two symmetric failures:
  1. Unlocated (`unknown`) event totals sit in a dead key, leaving multi-state
     events with `natl = 0` so the scope decode cannot catch regional totals
     mis-keyed to a child state.
  2. A hardcoded `part_ratio = 2.0` (`child <= Event / 2`) cannibalizes single-
     and dominant-location events where one place IS 88-98% of the event.

`EventContainer` models the event `E` as the top-level container over declared
child locations `C = states`, processing each casualty role (`dead`, `injured`,
`missing`) independently so injured counts never set the scale for deaths:
  - Unlocated (`unknown`) and explicit aggregate (`__aggregate__`) readings
    stage on the container and are cleaned via exact global DP.
  - Within a single article, observations for a key are grouped into consistent
    value clusters; one article can assert at most one `OWN` toll per place.
  - Multi-child conservation: when the container has independent observations
    and multiple children are active, child `k` is bounded by `E(t) - sum_{j!=k} m_j(t)`;
    readings exceeding child `k`'s room that match the container bracket promote
    to `E`.
  - Single-child reconciliation: when only one child location in `C` is active
    across the event, validated unlocated container readings reconcile with that
    sole child rather than fragmenting one stream in two.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Iterable, Optional

RISE_ROLES = ("dead", "injured")
UNRESOLVED_KEYS = frozenset({"unknown", "all", ""})

_MONTH_LEFT_RE = re.compile(
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+$",
    re.I,
)
_YEAR_RIGHT_RE = re.compile(r"(?:st|nd|rd|th)?\s*,\s*(?:19|20)\d{2}\b", re.I)


def span_boundary_defect(
    full_text: str,
    span: str,
    start: Optional[int],
    end: Optional[int],
) -> Optional[str]:
    """Return a defect reason when `[start, end]` is a truncated number or calendar day.

    Whitespace tokenization splits `'35,418'` into `'35'`, `','`, `'418'` and
    `'1.2 million'` into `'1'`, `'.'`, `'2'`, `'million'`. A candidate anchor on
    `'35'`, `'418'`, or `'1'` is half of a number token, not a standalone count.
    Likewise `'1'` in `'Oct. 1, 2024'` is a calendar day in a caption or dateline.
    """
    if start is None or end is None or not (0 <= start < end <= len(full_text)):
        return None
    s_clean = span.strip()
    if not re.fullmatch(r"[\d,]+", s_clean):
        return None
    if start > 0 and full_text[start - 1].isdigit():
        return "partial_left_digit"
    if end < len(full_text) and full_text[end].isdigit():
        return "partial_right_digit"
    if start >= 2 and full_text[start - 1] in ",.:/" and full_text[start - 2].isdigit():
        return "partial_left_sep"
    if end + 1 < len(full_text) and full_text[end] in ",.:/" and full_text[end + 1].isdigit():
        return "partial_right_sep"
    left_ctx = full_text[max(0, start - 16): start]
    right_ctx = full_text[end: min(len(full_text), end + 12)]
    if _MONTH_LEFT_RE.search(left_ctx) and _YEAR_RIGHT_RE.match(right_ctx):
        return "calendar_date"
    return None


def _article_key(obs: dict):
    if "_article_id" in obs and obs["_article_id"] is not None:
        return obs["_article_id"]
    return round(float(obs["t_hours"]), 4)


def dedup_within_article(observations: list[dict]) -> tuple[list[dict], list[dict]]:
    """Collapse exact `(article, role, event_key, value)` chunk repeats to one report."""
    best: dict[tuple, dict] = {}
    order: list[tuple] = []
    for o in observations:
        sig = (_article_key(o), o.get("role"), str(o.get("event_key", "")), float(o["value"]))
        if sig not in best:
            best[sig] = o
            order.append(sig)
        elif float(o.get("confidence") or 0.0) > float(best[sig].get("confidence") or 0.0):
            best[sig] = o
    kept = [best[s] for s in order]
    kept_ids = {id(x) for x in kept}
    dropped = [dict(o, _from=str(o.get("event_key", "")), _dedup=True)
               for o in observations if id(o) not in kept_ids]
    return kept, dropped


def _cluster_article_obs(obs_in_article: list[dict], cluster_tol_log: float = 0.10) -> list[list[dict]]:
    """Group same-article observations whose values agree within `cluster_tol_log`."""
    ordered = sorted(obs_in_article, key=lambda o: float(o["value"]))
    clusters: list[list[dict]] = []
    for o in ordered:
        lv = math.log(max(float(o["value"]), 1.0))
        if clusters:
            ref_lv = math.log(max(float(clusters[-1][0]["value"]), 1.0))
            if abs(lv - ref_lv) <= cluster_tol_log:
                clusters[-1].append(o)
                continue
        clusters.append([o])
    return clusters


def _running_series(items: list[dict]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    run = 0.0
    for x in sorted(items, key=lambda z: float(z["t_hours"])):
        run = max(run, float(x["value"]))
        out.append((float(x["t_hours"]), run))
    return out


def _level_at(series: list[tuple[float, float]], t: float) -> float:
    best = 0.0
    for tt, val in series:
        if tt <= t:
            best = val
        else:
            break
    return best


def _next_level_at(series: list[tuple[float, float]], t: float) -> float:
    for tt, val in series:
        if tt >= t:
            return val
    return series[-1][1] if series else 0.0


def exact_article_dp(
    obs: list[dict],
    *,
    rising: bool = True,
    sigma: float = 0.3,
    down_sigma: float = 0.12,
    reject_cost: float = 4.0,
    step_weight: float = 0.05,
    ceiling_fn: Optional[Callable[[float, float], Optional[float]]] = None,
    agg_fn: Optional[Callable[[float, float, float], float]] = None,
    sibling_claimed: Optional[set[tuple]] = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Exact sequence decode grouped by article, choosing at most one OWN cluster per article.

    Within an article `a` having clusters `C_{a,0}, ..., C_{a,M-1}`, at most one cluster
    may be assigned `OWN`; every non-OWN cluster in `a` takes the better of `PROMOTE`
    (`agg_fn`) or `REJECT` (`-reject_cost * len(cluster)`). State `r` tracks the index
    of the running-maximum `OWN` cluster so rejecting an outlier leaves the running
    level untouched for all later articles without hard-EM local minima.
    """
    if not obs:
        return [], [], []

    # Group by article in chronological order
    by_art: list[tuple[object, float, list[list[dict]]]] = []
    art_buckets: dict[object, list[dict]] = {}
    art_order: list[object] = []
    for o in sorted(obs, key=lambda x: (float(x["t_hours"]), float(x["value"]))):
        ak = _article_key(o)
        if ak not in art_buckets:
            art_buckets[ak] = []
            art_order.append(ak)
        art_buckets[ak].append(o)
    for ak in art_order:
        grp = art_buckets[ak]
        t_art = float(grp[0]["t_hours"])
        by_art.append((ak, t_art, _cluster_article_obs(grp)))

    # Enumerate all candidate OWN cluster representatives to index DP states 1..K
    all_clusters: list[tuple[float, list[dict]]] = []
    art_cluster_ids: list[list[int]] = []
    for _ak, _t, clist in by_art:
        ids = []
        for cl in clist:
            v_rep = max(float(x["value"]) for x in cl)
            all_clusters.append((max(v_rep, 1.0), cl))
            ids.append(len(all_clusters))  # 1-based state index
        art_cluster_ids.append(ids)

    def non_own_score_and_act(t: float, cl: list[dict], m_own: float) -> tuple[float, str]:
        rej_sc = -reject_cost * len(cl)
        if agg_fn is None:
            return rej_sc, "REJ"
        v_rep = max(float(x["value"]) for x in cl)
        em_a = agg_fn(t, max(v_rep, 1.0), m_own) * len(cl)
        if em_a > rej_sc:
            return em_a, "AGG"
        return rej_sc, "REJ"

    NEG_INF = -1e18
    dp: dict[int, float] = {0: 0.0}
    # back[step][next_r] = (prev_r, chosen_own_local_idx_or_None, per_cluster_actions)
    back: list[dict[int, tuple[int, Optional[int], list[str]]]] = []

    for step_idx, (ak, t, clist) in enumerate(by_art):
        cids = art_cluster_ids[step_idx]
        m_cnt = len(clist)
        ndp: dict[int, float] = {}
        nbk: dict[int, tuple[int, Optional[int], list[str]]] = {}

        for r, sc in dp.items():
            m_own = 0.0 if r == 0 else all_clusters[r - 1][0]
            non_own = [non_own_score_and_act(t, clist[idx], m_own) for idx in range(m_cnt)]
            base_non_own_sum = sum(x[0] for x in non_own)

            # Option A: assign NO cluster in this article to OWN
            acts_none = [x[1] for x in non_own]
            cand_none = sc + base_non_own_sum
            if cand_none > ndp.get(r, NEG_INF):
                ndp[r] = cand_none
                nbk[r] = (r, None, acts_none)

            # Option B: assign exactly one cluster `j` in this article to OWN
            for j in range(m_cnt):
                cl = clist[j]
                cid = cids[j]
                v = all_clusters[cid - 1][0]
                lv = math.log(v)
                mult = len(cl)

                pen_low = 0.0
                pen_step = 0.0
                if rising and m_own > 0:
                    lo = math.log(m_own)
                    if lv < lo:
                        pen_low = lo - lv
                    else:
                        pen_step = lv - lo

                pen_high = 0.0
                if ceiling_fn is not None:
                    cap = ceiling_fn(t, m_own)
                    if cap is not None and cap > 0:
                        pen_high = max(0.0, lv - math.log(max(cap, 1.0)))

                pen_spike = 0.0
                if m_own >= 10.0 and lv - math.log(m_own) > math.log(25.0):
                    pen_spike = lv - math.log(m_own) - math.log(25.0)

                sib_pen = 0.0
                if sibling_claimed and any((ak, round(float(x["value"]), 4)) in sibling_claimed for x in cl):
                    sib_pen = reject_cost

                em_single = (
                    -(pen_low ** 2) / (2.0 * down_sigma ** 2)
                    - (pen_high ** 2 + pen_spike ** 2) / (2.0 * sigma ** 2)
                    - step_weight * (pen_step ** 2)
                    - sib_pen
                )
                own_sc = em_single * mult
                total_sc = sc + (base_non_own_sum - non_own[j][0]) + own_sc
                nr = cid if (not rising or r == 0 or v >= m_own) else r
                if total_sc > ndp.get(nr, NEG_INF):
                    acts = [x[1] for x in non_own]
                    acts[j] = "OWN"
                    ndp[nr] = total_sc
                    nbk[nr] = (r, j, acts)

        dp = ndp
        back.append(nbk)

    best_r = max(dp, key=lambda k: dp[k])
    own_out: list[dict] = []
    agg_out: list[dict] = []
    rej_out: list[dict] = []

    cur_r = best_r
    chosen_per_step: list[list[str]] = [[] for _ in by_art]
    for step_idx in range(len(by_art) - 1, -1, -1):
        prev_r, _j, acts = back[step_idx][cur_r]
        chosen_per_step[step_idx] = acts
        cur_r = prev_r

    for (_ak, _t, clist), acts in zip(by_art, chosen_per_step):
        for cl, act in zip(clist, acts):
            if act == "OWN":
                own_out.extend(cl)
            elif act == "AGG":
                agg_out.extend(cl)
            else:
                rej_out.extend(cl)

    return own_out, agg_out, rej_out


class EventContainer:
    """Top-level event container resolving unlocated and child-location casualty streams."""

    def __init__(
        self,
        states: Iterable[str],
        aggregate_key: str = "__aggregate__",
        sigma: float = 0.3,
        down_sigma: float = 0.12,
        reject_cost: float = 4.0,
        outer_iters: int = 4,
    ) -> None:
        self.states = set(states)
        self.aggregate_key = aggregate_key
        self.sigma = sigma
        self.down_sigma = down_sigma
        self.reject_cost = reject_cost
        self.outer_iters = outer_iters

    def resolve_role(
        self,
        role_obs: list[dict],
        role: str,
    ) -> tuple[dict[str, list[dict]], list[dict], list[dict]]:
        """Resolve one role (`dead`, `injured`, or `missing`) across container and children."""
        rising = role in RISE_ROLES
        deduped, dedup_dropped = dedup_within_article(role_obs)

        base_container_raw = [
            o for o in deduped
            if str(o.get("event_key", "")) == self.aggregate_key
            or str(o.get("event_key", "")).lower() in UNRESOLVED_KEYS
        ]
        has_container_seed = len(base_container_raw) > 0

        child_raw: dict[str, list[dict]] = {k: [] for k in self.states}
        passthrough: dict[str, list[dict]] = {}
        for o in deduped:
            k = str(o.get("event_key", ""))
            if k == self.aggregate_key or k.lower() in UNRESOLVED_KEYS:
                continue
            if k in self.states:
                child_raw[k].append(o)
            else:
                passthrough.setdefault(k, []).append(o)

        child_own: dict[str, list[dict]] = {}
        child_prom: dict[str, list[dict]] = {k: [] for k in self.states}
        child_rej: dict[str, list[dict]] = {k: [] for k in self.states}
        for k, obs_k in child_raw.items():
            own_k, _, rej_k = exact_article_dp(
                obs_k,
                rising=rising,
                sigma=self.sigma,
                down_sigma=self.down_sigma,
                reject_cost=self.reject_cost,
            )
            child_own[k] = own_k
            child_rej[k] = rej_k

        active_init = [k for k in self.states if child_own[k]]
        if len(active_init) > 1:
            # Order children largest-peak-first so dominant siblings establish claims first
            child_order = sorted(
                self.states,
                key=lambda k: max((float(x["value"]) for x in child_own[k]), default=0.0),
                reverse=True,
            )
            for _ in range(self.outer_iters):
                child_series = {k: _running_series(child_own[k]) for k in self.states}

                changed = False
                for k in child_order:
                    obs_k = child_raw[k]
                    if not obs_k:
                        continue
                    other_prom = [o for j in self.states if j != k for o in child_prom[j]]
                    cont_k, _, _ = exact_article_dp(
                        base_container_raw + other_prom,
                        rising=rising,
                        sigma=self.sigma,
                        down_sigma=self.down_sigma,
                        reject_cost=self.reject_cost,
                    )
                    cont_series = _running_series(cont_k)

                    my_peak = max((float(x["value"]) for x in child_own[k]), default=0.0)
                    my_art_conf: dict[tuple, float] = {}
                    my_art_count: dict[object, int] = {}
                    for x in obs_k:
                        ak = _article_key(x)
                        vk = round(float(x["value"]), 4)
                        my_art_conf[(ak, vk)] = max(my_art_conf.get((ak, vk), 0.0), float(x.get("confidence") or 0.5))
                        my_art_count[ak] = my_art_count.get(ak, 0) + 1

                    sib_claims: set[tuple] = set()
                    for j in self.states:
                        if j == k:
                            continue
                        j_peak = max((float(x["value"]) for x in child_own[j]), default=0.0)
                        j_art_count: dict[object, int] = {}
                        for x in child_raw[j]:
                            ak = _article_key(x)
                            j_art_count[ak] = j_art_count.get(ak, 0) + 1
                        for x in child_own[j]:
                            ak = _article_key(x)
                            vk = round(float(x["value"]), 4)
                            if (ak, vk) not in my_art_conf:
                                continue
                            # Sibling `j` owns `(ak, vk)` when `j` has no alternative in `ak` while `k` does,
                            # or when both have alternatives and `j` is the larger stream.
                            j_sole = j_art_count.get(ak, 0) == 1 and my_art_count.get(ak, 0) > 1
                            k_sole = my_art_count.get(ak, 0) == 1 and j_art_count.get(ak, 0) > 1
                            if k_sole:
                                continue
                            if j_sole or (j_peak, j) > (my_peak, k):
                                sib_claims.add((ak, vk))

                    if has_container_seed and rising:
                        def ceiling_fn(t: float, m_own: float, _k: str = k):
                            other_sum = sum(_level_at(child_series[j], t) for j in self.states if j != _k)
                            c_now = _level_at(cont_series, t)
                            if c_now <= 0:
                                return None
                            return max(1.0, c_now - other_sum)

                        def agg_fn(t: float, v: float, m_own: float, _k: str = k) -> float:
                            other_sum = sum(_level_at(child_series[j], t) for j in self.states if j != _k)
                            c_now = _level_at(cont_series, t)
                            c_nxt = _next_level_at(cont_series, t)
                            if c_now <= 0 and c_nxt <= 0:
                                return -1e12
                            e_low = max(c_now, other_sum + m_own)
                            e_high = max(c_nxt, e_low)
                            room = max(m_own * 1.5, (c_now - other_sum) if c_now > 0 else m_own * 1.5)
                            if v <= room:
                                return -1e12
                            lv = math.log(max(v, 1.0))
                            lo_b = math.log(max(e_low, 1.0)) if e_low > 0 else lv
                            hi_b = math.log(max(e_high, 1.0))
                            if lo_b <= lv <= hi_b:
                                dist = 0.0
                            elif lv < lo_b:
                                dist = lo_b - lv
                            else:
                                dist = lv - hi_b
                            return -(dist ** 2) / (2.0 * self.sigma ** 2) - 0.25
                    else:
                        ceiling_fn = None
                        agg_fn = None

                    new_own, new_prom, new_rej = exact_article_dp(
                        obs_k,
                        rising=rising,
                        sigma=self.sigma,
                        down_sigma=self.down_sigma,
                        reject_cost=self.reject_cost,
                        ceiling_fn=ceiling_fn,
                        agg_fn=agg_fn,
                        sibling_claimed=sib_claims,
                    )
                    if (
                        [id(x) for x in new_own] != [id(x) for x in child_own[k]]
                        or [id(x) for x in new_prom] != [id(x) for x in child_prom[k]]
                    ):
                        changed = True
                    child_own[k] = new_own
                    child_prom[k] = new_prom
                    child_rej[k] = new_rej
                    child_series[k] = _running_series(new_own)

                if not changed:
                    break

        all_prom = [o for k in self.states for o in child_prom[k]]
        cont_kept, _, cont_rej = exact_article_dp(
            base_container_raw + all_prom,
            rising=rising,
            sigma=self.sigma,
            down_sigma=self.down_sigma,
            reject_cost=self.reject_cost,
        )

        active_children = [k for k in self.states if child_own[k]]
        moved: list[dict] = []
        dropped: list[dict] = list(dedup_dropped) + [
            dict(o, _from=str(o.get("event_key", "")), _container_reject=True)
            for o in cont_rej
        ]
        for k in self.states:
            dropped.extend(dict(o, _from=k, _container_reject=True) for o in child_rej[k])

        kept: dict[str, list[dict]] = dict(passthrough)
        for k in self.states:
            if child_own[k]:
                kept[k] = list(child_own[k])

        if len(active_children) == 1:
            sole = active_children[0]
            reconciled = []
            for o in cont_kept:
                orig_k = str(o.get("event_key", ""))
                if orig_k.lower() in UNRESOLVED_KEYS:
                    reconciled.append(dict(o, event_key=sole, _from=orig_k))
                else:
                    kept.setdefault(self.aggregate_key, []).append(o)
            if reconciled:
                merged_sole, _, rej_sole = exact_article_dp(
                    kept.get(sole, []) + reconciled,
                    rising=rising,
                    sigma=self.sigma,
                    down_sigma=self.down_sigma,
                    reject_cost=self.reject_cost,
                )
                rec_ids = {id(x) for x in reconciled}
                kept_rec_ids = {id(x) for x in merged_sole if id(x) in rec_ids}
                moved.extend(x for x in reconciled if id(x) in kept_rec_ids)
                kept[sole] = merged_sole
                for o in rej_sole:
                    src = o.get("_from", sole)
                    dropped.append(dict(o, _from=src, _container_reject=True))
        else:
            final_agg = []
            for o in cont_kept:
                orig_k = str(o.get("event_key", ""))
                if orig_k != self.aggregate_key:
                    cp = dict(o, event_key=self.aggregate_key, _from=orig_k)
                    moved.append(cp)
                    final_agg.append(cp)
                else:
                    final_agg.append(o)
            if final_agg:
                kept[self.aggregate_key] = final_agg

        return kept, moved, dropped


def container_gate(
    observations: list[dict],
    states: dict,
    aggregate_key: str = "__aggregate__",
    sigma: float = 0.3,
    down_sigma: float = 0.12,
    reject_cost: float = 4.0,
) -> tuple[dict[str, list[dict]], list[dict], list[dict]]:
    """Same `(kept, moved, dropped)` contract as `scope_gate.viterbi_gate`, run per role."""
    container = EventContainer(
        states=states.keys(),
        aggregate_key=aggregate_key,
        sigma=sigma,
        down_sigma=down_sigma,
        reject_cost=reject_cost,
    )
    by_role: dict[str, list[dict]] = {}
    role_order: list[str] = []
    for o in observations:
        r = str(o.get("role") or "dead")
        if r not in by_role:
            by_role[r] = []
            role_order.append(r)
        by_role[r].append(o)

    kept_all: dict[str, list[dict]] = {}
    moved_all: list[dict] = []
    dropped_all: list[dict] = []
    for r in role_order:
        k_r, m_r, d_r = container.resolve_role(by_role[r], r)
        for key, lst in k_r.items():
            kept_all.setdefault(key, []).extend(lst)
        moved_all.extend(m_r)
        dropped_all.extend(d_r)
    return kept_all, moved_all, dropped_all
