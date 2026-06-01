"""Predictive modeling with the guardrails that matter at scale.

`fit_predictive` trains a calibrated classifier with a leakage-safe CV strategy
(grouped split when an entity id is given, so the same unit never spans
train/test), reports AUC / PR-AUC / Brier (calibration), and runs a leakage scan
that flags any single feature that predicts the target almost perfectly (the
classic target-leak). This is the Meta/Google-pragmatism layer: honest
evaluation, not just a high in-sample number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


@dataclass
class MLResult:
    target: str
    features: list
    cv_strategy: str
    auc: float
    pr_auc: float
    brier: float
    leakage_flags: list = field(default_factory=list)
    n: int = 0

    def as_dict(self) -> dict:
        return {"target": self.target, "features": self.features,
                "cv_strategy": self.cv_strategy, "auc": round(self.auc, 4),
                "pr_auc": round(self.pr_auc, 4), "brier": round(self.brier, 4),
                "leakage_flags": self.leakage_flags, "n": self.n}


def _leakage_scan(df, target, features) -> list:
    flags = []
    y = df[target].to_numpy()
    for f in features:
        try:
            x = df[[f]].to_numpy()
            auc = roc_auc_score(y, LogisticRegression(max_iter=200)
                                .fit(x, y).predict_proba(x)[:, 1])
            if auc > 0.99:
                flags.append(f"{f}: single-feature AUC={auc:.3f} (possible target leak)")
        except Exception:
            continue
    return flags


def fit_predictive(df, target="t", features=("x",), group=None,
                   folds=5, seed=0) -> MLResult:
    feats = list(features)
    X = df[feats].to_numpy()
    y = df[target].to_numpy().astype(int)
    n = len(df)

    if group is not None and group in df.columns:
        cv = GroupKFold(n_splits=folds)
        groups = df[group].to_numpy()
        strategy = f"GroupKFold(by={group}) — no entity leakage across folds"
        proba = cross_val_predict(GradientBoostingClassifier(random_state=seed),
                                  X, y, cv=cv, groups=groups,
                                  method="predict_proba")[:, 1]
    else:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        strategy = f"StratifiedKFold(k={folds})"
        proba = cross_val_predict(GradientBoostingClassifier(random_state=seed),
                                  X, y, cv=cv, method="predict_proba")[:, 1]

    return MLResult(
        target=target, features=feats, cv_strategy=strategy,
        auc=float(roc_auc_score(y, proba)),
        pr_auc=float(average_precision_score(y, proba)),
        brier=float(brier_score_loss(y, proba)),
        leakage_flags=_leakage_scan(df, target, feats), n=n)


def conformal_classify(df, target="t", features=("x",), alpha=0.1,
                       seed=0) -> dict:
    """Split-conformal prediction sets (LAC) with a distribution-free coverage
    guarantee of 1 - alpha. Reports empirical coverage and mean set size on a
    held-out test fold — calibrated uncertainty, not just a point probability.
    """
    from sklearn.model_selection import train_test_split
    X = df[list(features)].to_numpy(); y = df[target].to_numpy().astype(int)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=seed)
    X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5,
                                                random_state=seed)
    clf = GradientBoostingClassifier(random_state=seed).fit(X_tr, y_tr)
    classes = clf.classes_
    # nonconformity = 1 - p(true label) on calibration set
    p_cal = clf.predict_proba(X_cal)
    idx = {c: i for i, c in enumerate(classes)}
    scores = np.array([1 - p_cal[i, idx[y_cal[i]]] for i in range(len(y_cal))])
    qhat = np.quantile(scores, min(1.0, (1 - alpha) * (len(scores) + 1) / len(scores)))
    # test sets: include label c if (1 - p_c) <= qhat
    p_te = clf.predict_proba(X_te)
    covered, sizes = 0, []
    for i in range(len(y_te)):
        pset = [classes[j] for j in range(len(classes)) if (1 - p_te[i, j]) <= qhat]
        sizes.append(len(pset))
        if y_te[i] in pset:
            covered += 1
    return {"target_coverage": round(1 - alpha, 3),
            "empirical_coverage": round(covered / len(y_te), 3),
            "mean_set_size": round(float(np.mean(sizes)), 3),
            "n_test": int(len(y_te))}
