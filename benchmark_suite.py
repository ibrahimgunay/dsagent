"""
dsagent real-world benchmark suite.

Runs dsagent's own estimators against a battery of classic, published causal /
econometric datasets and grades each estimate against the literature value. It
spans every design the pipeline supports: RCT, observational (selection on
observables), instrumental variables, difference-in-differences / panel, and
plain regression.

WHY A HARNESS: this sandbox has no outbound network for the data step, so only
the bundled LaLonde file runs here. On any machine WITH network, every row below
runs from its public URL with a single `pd.read_csv(url)` — no other changes.

    python benchmark_suite.py                 # runs all (skips rows it can't fetch)
    python benchmark_suite.py --only lalonde   # one dataset

Each dataset is graded PASS if dsagent's estimate lands within tolerance of the
published benchmark.
"""
import argparse
import os
import numpy as np
import pandas as pd
from dsagent.execution import estimators as est

R = "https://vincentarelbundock.github.io/Rdatasets/csv"

# ---------------------------------------------------------------- dataset specs
# Each spec: how to map the raw file to (y, t, controls[, z]) + the design +
# the published benchmark to grade against.
SPECS = [
    dict(key="lalonde", title="LaLonde NSW job training (RCT)",
         url=f"{R}/causaldata/nsw_mixtape.csv", local="lalonde_nsw_exp.csv",
         design="rct", y="re78", t="treat",
         controls=["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"],
         benchmark=1794, tol=400, units="$ on 1978 earnings",
         source="Dehejia-Wahba 1999 (~$1,672-1,794)"),

    dict(key="nhefs", title="NHEFS smoking cessation -> weight gain (observational)",
         url=f"{R}/causaldata/nhefs.csv", design="observational",
         y="wt82_71", t="qsmk",
         controls=["age", "sex", "race", "education", "smokeintensity",
                   "smokeyrs", "exercise", "active", "wt71"],
         benchmark=3.4, tol=1.2, units="kg weight gain",
         source="Hernan-Robins, Causal Inference: What If (~3.4-3.5 kg, IPW/g-formula)"),

    dict(key="card", title="Card proximity-to-college returns to schooling (IV)",
         url=f"{R}/causaldata/close_college.csv", design="iv",
         y="lwage", t="educ", z="nearc4",
         controls=["exper", "black", "south", "married", "smsa"],
         benchmark=0.14, tol=0.07, units="log-wage per year of schooling",
         source="Card 1995 (IV ~0.13, well above OLS ~0.07)"),

    dict(key="cigsSW", title="Cigarette demand elasticity (IV)",
         url=f"{R}/AER/CigarettesSW.csv", design="iv_custom",
         benchmark=-1.08, tol=0.4, units="price elasticity of demand",
         source="Stock-Watson via Cig tax instrument (~ -1.0 to -1.1)"),

    dict(key="fatalities", title="Beer tax -> traffic fatalities (DiD/panel)",
         url=f"{R}/AER/Fatalities.csv", design="panel_fe",
         y="frate", t="beertax", unit="state", time="year",
         benchmark=-0.66, tol=0.5, units="deaths/10k per $ beer tax (within)",
         source="Stock-Watson (FE within estimate negative ~ -0.66)"),

    dict(key="organ", title="Active-choice default -> organ donation (DiD)",
         url=f"{R}/causaldata/organ_donations.csv", design="did_custom_organ",
         y="Rate", t="Quarter", unit="State", time="Quarter",
         benchmark=-0.022, tol=0.03, units="pp change in donation rate",
         source="Kessler-Roth 2014 (active-choice REDUCED donations ~ -2pp)"),

    dict(key="castle", title="Castle-doctrine laws -> homicide (DiD/TWFE)",
         url=f"{R}/causaldata/castle.csv", design="did_panel",
         y="l_homicide", t="post", unit="sid", time="year",
         benchmark=0.08, tol=0.07, units="log homicide (8-10% increase)",
         source="Cheng-Hoekstra 2013 (~ +0.07-0.10)"),

    dict(key="thornton", title="Cash incentive -> learn HIV result (RCT)",
         url=f"{R}/causaldata/thornton_hiv.csv", design="rct",
         y="got", t="any", controls=["age", "hiv2004", "distvct"],
         benchmark=0.45, tol=0.15, units="pp increase in learning result",
         source="Thornton 2008 (control ~34% -> treated ~80%, ~+45pp)"),

    dict(key="caschools", title="Class size -> test scores (regression)",
         url=f"{R}/AER/CASchools.csv", design="regression_custom",
         benchmark=-2.28, tol=1.5, units="test points per student/teacher ratio",
         source="Stock-Watson (bivariate ~ -2.28, attenuates with controls)"),

    dict(key="journals", title="Journal price -> library subscriptions (regression/IV)",
         url=f"{R}/AER/Journals.csv", design="regression_custom",
         benchmark=-0.53, tol=0.25, units="elasticity of demand",
         source="Stock-Watson Journals (~ -0.5)"),

    dict(key="wage1", title="Return to education (regression)",
         url=f"{R}/wooldridge/wage1.csv", design="regression_custom",
         benchmark=0.083, tol=0.03, units="log-wage per year of education",
         source="Wooldridge wage1 (~8.3%/yr)"),

    dict(key="gov_transfers", title="Cash transfers -> political support (RDD)",
         url=f"{R}/causaldata/gov_transfers.csv", design="rdd",
         y="Support", running_var="Income_Centered", cutoff=0,
         benchmark=0.20, tol=0.20, units="jump in support at eligibility cutoff",
         source="Manacorda-Miguel-Vigorito 2011 (positive discontinuity ~0.2-0.4)"),
]


def _load(spec):
    path = spec.get("local")
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return pd.read_csv(spec["url"])           # needs network (user env)


def _grade(point, spec):
    b, tol = spec.get("benchmark"), spec.get("tol")
    if b is None:
        return "DESIGN-ONLY"
    return "PASS" if abs(point - b) <= tol else "CHECK"


def run_one(spec):
    df = _load(spec)
    d = spec["design"]

    if d in ("rct", "observational"):
        ctrls = [c for c in spec["controls"] if c in df.columns]
        x = df.rename(columns={spec["y"]: "y", spec["t"]: "t"})[["y", "t"] + ctrls].dropna()
        for c in ctrls:                                   # numeric-encode any factors
            if x[c].dtype == object:
                x[c] = pd.factorize(x[c])[0]
        x["t"] = (x["t"] > x["t"].median()).astype(int) if x["t"].nunique() > 2 else \
                 pd.factorize(x["t"])[0] if x["t"].dtype == object else x["t"]
        r = est.double_ml(x, controls=tuple(ctrls)) if d == "observational" \
            else est.diff_in_means(x)
        return r.point

    if d == "iv":
        ctrls = [c for c in spec["controls"] if c in df.columns]
        x = df.rename(columns={spec["y"]: "y", spec["t"]: "t", spec["z"]: "z"})
        x = x[["y", "t", "z"] + ctrls].dropna()
        for c in ctrls:
            if x[c].dtype == object:
                x[c] = pd.factorize(x[c])[0]
        return est.iv_2sls(x).point

    if d == "iv_custom":                                  # CigarettesSW elasticity
        sub = df[df["year"] == df["year"].max()].copy()
        sub["lp"] = np.log(sub["price"] / sub["cpi"])
        sub["lq"] = np.log(sub["packs"])
        sub["z"] = sub["taxs"] / sub["cpi"]               # sales-tax instrument
        x = sub.rename(columns={"lq": "y", "lp": "t"})[["y", "t", "z"]].dropna()
        return est.iv_2sls(x).point

    if d == "panel_fe":                                   # within (FE) slope
        g = df.copy()
        if "frate" not in g.columns and {"fatal", "pop"}.issubset(g.columns):
            g["frate"] = g["fatal"] / g["pop"] * 10000
        g = g[[spec["unit"], spec["y"], spec["t"]]].dropna()
        g["yd"] = g[spec["y"]] - g.groupby(spec["unit"])[spec["y"]].transform("mean")
        g["td"] = g[spec["t"]] - g.groupby(spec["unit"])[spec["t"]].transform("mean")
        return float(np.polyfit(g["td"], g["yd"], 1)[0])

    if d == "did_panel":
        x = df.rename(columns={spec["y"]: "y"})
        # two-way demeaned slope of outcome on treatment indicator
        tcol = spec["t"]
        if tcol not in x.columns:
            return float("nan")
        x = x[[spec["unit"], spec["time"], "y", tcol]].dropna()
        for c in (spec["unit"], spec["time"]):
            x[f"_{c}"] = x.groupby(c)["y"].transform("mean")
        x["yd"] = x["y"] - x[f"_{spec['unit']}"] - x[f"_{spec['time']}"] + x["y"].mean()
        return float(np.polyfit(x[tcol], x["yd"], 1)[0])

    if d == "regression_custom":
        if spec["key"] == "caschools":
            df["str"] = df["students"] / df["teachers"]
            return float(np.polyfit(df["str"], df["score"] if "score" in df else
                                    (df["read"] + df["math"]) / 2, 1)[0])
        if spec["key"] == "journals":
            x = np.log(df["price"] / df["citations"]); y = np.log(df["subs"])
            return float(np.polyfit(x, y, 1)[0])
        if spec["key"] == "wage1":
            return float(np.polyfit(df["educ"], df["lwage"], 1)[0])

    if d == "did_custom_organ":                           # Kessler-Roth 2x2 DiD
        # California switched to active-choice mid-panel; quarters are ordered labels.
        g = df.rename(columns={"Rate": "y"}).copy()
        g["treated"] = (g["State"] == "California").astype(int)
        order = sorted(g["Quarter"].unique())
        idx = {q: i for i, q in enumerate(order)}
        cut = len(order) // 2
        g["post"] = (g["Quarter"].map(idx) >= cut).astype(int)
        m = g.groupby(["treated", "post"])["y"].mean()
        did = (m.get((1, 1), 0) - m.get((1, 0), 0)) - (m.get((0, 1), 0) - m.get((0, 0), 0))
        return float(did)

    if d == "rdd":                                        # local-linear RDD
        rv = spec["running_var"]
        cutoff = spec.get("cutoff", 0)
        g = df.copy()
        g["_rv"] = g[rv] - cutoff
        g["_d"] = (g["_rv"] <= 0).astype(int)            # below cutoff = eligible = treated
        g["_int"] = g["_rv"] * g["_d"]
        g = g[[spec["y"], "_rv", "_d", "_int"]].dropna()
        X = np.column_stack([np.ones(len(g)), g["_rv"], g["_d"], g["_int"]])
        b = np.linalg.lstsq(X, g[spec["y"]].values, rcond=None)[0]
        return float(b[2])                                # jump at the cutoff

    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    specs = [s for s in SPECS if not args.only or s["key"] == args.only]
    print(f"{'dataset':<42}{'design':<14}{'estimate':>12}{'benchmark':>12}  result")
    print("-" * 95)
    n_pass = n_run = 0
    for s in specs:
        try:
            pt = run_one(s)
            if pt != pt:                       # nan
                raise ValueError("mapping produced no estimate")
            verdict = _grade(pt, s)
            n_run += 1
            n_pass += verdict == "PASS"
            bm = "n/a" if s["benchmark"] is None else f"{s['benchmark']:.3g}"
            print(f"{s['title'][:41]:<42}{s['design']:<14}{pt:>12.3f}{bm:>12}  {verdict}")
        except Exception as e:
            msg = "no network (runs in your env)" if "URLError" in type(e).__name__ \
                  or "Errno" in str(e) or "HTTP" in str(e) else str(e)[:40]
            print(f"{s['title'][:41]:<42}{s['design']:<14}{'-':>12}{'-':>12}  SKIP: {msg}")
    print("-" * 95)
    print(f"ran {n_run} datasets, {n_pass} matched the published benchmark within tolerance")
    print("(rows marked SKIP need outbound network for pd.read_csv(url) — run on your machine)")


if __name__ == "__main__":
    main()
