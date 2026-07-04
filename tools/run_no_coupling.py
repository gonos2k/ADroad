"""Reproducible no-coupling driver for RoadSurf-Python (P0 §13).

Replicates RoadSurf-Python `main.py`'s loop with use_coupling=False and no
matplotlib, so it is importable and instrumentable. Optional snapshot hooks
capture function-boundary state via monkeypatch WITHOUT touching physics
(deep-copy only) — see P0 §13 hook purity contract.

Usage:
    python tools/run_no_coupling.py                 # write fixture CSV
    python tools/run_no_coupling.py --snapshot 0    # + dump step-0 snapshots
"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RSP_SRC = REPO / "reference" / "RoadSurf-Python" / "src"
DATA = REPO / "reference" / "RoadSurf-Python" / "example_data" / "test_input.csv"


def _import_rsp():
    sys.path.insert(0, str(RSP_SRC))
    import readInputcsv, Initialization, InputOutput, BalanceModel  # noqa
    import Storage, Cond, ModRadiation, WearingFactors, writecsv, BoundaryLayer  # noqa
    return dict(
        readInputcsv=readInputcsv, Initialization=Initialization,
        InputOutput=InputOutput, BalanceModel=BalanceModel, Storage=Storage,
        Cond=Cond, ModRadiation=ModRadiation, WearingFactors=WearingFactors,
        writecsv=writecsv, BoundaryLayer=BoundaryLayer,
    )


# --- snapshot hooks (deep-copy only; do not mutate model state) ------------

class _Capture:
    def __init__(self, target_step):
        self.target = target_step
        self.current = None
        self.data = {}

    def take(self, name, **state):
        if self.current == self.target:
            self.data[name] = {k: copy.deepcopy(v) for k, v in state.items()}


def _install_hooks(m, cap: _Capture):
    """Monkeypatch boundary functions to record state after they return."""
    BalanceModel, BoundaryLayer = m["BalanceModel"], sys.modules["BoundaryLayer"]

    o_blc = BoundaryLayer.CalcBLCondAndLE
    def blc(surf, DtSecs, SrfWat, phy, atm):
        r = o_blc(surf, DtSecs, SrfWat, phy, atm)
        cap.take("after_boundary_layer", BLCond=atm.BLCond, LE_Flux=atm.LE_Flux,
                 EvapmmTS=surf.EvapmmTS)
        return r
    BoundaryLayer.CalcBLCondAndLE = blc

    o_rnet = BalanceModel.CalcRNet
    def rnet(*a):
        r = o_rnet(*a)
        cap.take("after_calc_rnet", RNet=r)
        return r
    BalanceModel.CalcRNet = rnet

    o_prof = BalanceModel.calcProfile
    def prof(NLayers, DTSecs, TrfFric, ground, atm):
        r = o_prof(NLayers, DTSecs, TrfFric, ground, atm)
        cap.take("after_calc_profile", TmpNw=ground.TmpNw, GroundFlux=ground.GroundFlux)
        return r
    BalanceModel.calcProfile = prof


# --- driver (mirrors main.py, coupling OFF) --------------------------------

def build_model():
    """Initialize RoadSurf-Python state (coupling OFF), no loop. Returns
    (modules, objects-tuple). Used by parity harnesses."""
    m = _import_rsp()
    init_length, forecast_length = 48, 60
    forecast_start = datetime(2021, 3, 4, 3, 0, 0)
    start_time = forecast_start - timedelta(hours=init_length)
    end_time = forecast_start + timedelta(hours=forecast_length)
    timeStep, outputStep = 30.0, 15
    use_coupling = False
    lat, lon = 62.246, 25.769

    csv_data = m["readInputcsv"].read_csv_data(str(DATA))
    csv_data["timestamp"] = [t.timestamp() for t in csv_data["time"]]

    objs = m["Initialization"].initialize_model(
        timeStep, use_coupling, outputStep, csv_data, lat, lon,
        forecast_start, start_time, end_time, init_length)
    return m, objs


def run(output_path, snapshot_step=None, snapshots_out=None):
    m, objs = build_model()
    (modelInput, modelOutput, phy, ground, surf, atm, coupling, settings,
     condParam, localParam) = objs
    outputStep = settings.outputStep

    cap = _Capture(snapshot_step) if snapshot_step is not None else None
    if cap is not None:
        _install_hooks(m, cap)

    def one_step(i):
        m["Storage"].PrecipitationToStorage(
            settings, condParam, modelInput.PrecPhase[i], atm, surf)
        if -0.01 < localParam.sky_view < 1.0:
            m["ModRadiation"].ModRadiationBySurroundings(modelInput, ground, localParam, i)
        m["BalanceModel"].BalanceModelOneStep(
            modelInput.SW[i], modelInput.LW[i], phy, ground, surf, atm,
            settings, coupling, modelInput, i, condParam)
        wearF = m["WearingFactors"].WearingFactors()
        m["Cond"].WearFactors(condParam, settings.Tph, surf, wearF)
        m["Cond"].RoadCond(phy.MaxPormms, surf, atm, settings, condParam, wearF)
        ground.Albedo = m["Cond"].CalcAlbedo(surf, condParam)

    i = 0
    while i < settings.SimLen - 1 and not settings.simulation_failed:
        m["InputOutput"].CheckValues(modelInput, i, settings, surf, localParam)
        m["InputOutput"].SetCurrentValues(i, modelInput, atm, settings, surf, coupling, ground)
        if cap is not None:
            cap.current = i
        one_step(i)
        m["InputOutput"].SaveOutput(modelOutput, i, surf)
        i += 1

    if not settings.simulation_failed:
        m["InputOutput"].lastValues(modelInput, atm, settings, ground, surf)
        if cap is not None:
            cap.current = i
        one_step(i)
        m["InputOutput"].SaveOutput(modelOutput, i, surf)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    m["writecsv"].write_to_csv(modelOutput, modelInput, str(output_path), outputStep)

    if cap is not None and snapshots_out is not None:
        Path(snapshots_out).parent.mkdir(parents=True, exist_ok=True)
        with open(snapshots_out, "wb") as f:
            pickle.dump({"step": cap.target, "boundaries": cap.data}, f)
    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "fixtures" / "no_coupling" / "testi_output.csv"))
    ap.add_argument("--snapshot", type=int, default=None)
    ap.add_argument("--snapshots-out", default=str(REPO / "fixtures" / "no_coupling" / "snapshots_step.pkl"))
    a = ap.parse_args()
    run(a.out, a.snapshot, a.snapshots_out if a.snapshot is not None else None)
    print("wrote", a.out)
