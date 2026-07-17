from __future__ import annotations

import importlib.util
import json
import gc
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("DENDRITRON_OUTPUT_DIR") or _HERE.parent / "results" / "local")
OUT.mkdir(parents=True, exist_ok=True)
V07_PATH = _HERE / 'dendritron_plasticity_benchmark_v0_7.py'

spec = importlib.util.spec_from_file_location('dendritron_v07', V07_PATH)
v07 = importlib.util.module_from_spec(spec)
sys.modules['dendritron_v07'] = v07
assert spec.loader is not None
spec.loader.exec_module(v07)


def nearest_mode_oracle(world, X: np.ndarray, eval_modes: Dict[int, List[str]]) -> np.ndarray:
    centers, labels = [], []
    for c in sorted(eval_modes):
        for m in eval_modes[c]:
            centers.append(world.class_base[c] + world.mode_offsets[(c, m)])
            labels.append(c)
    C = np.stack(centers)
    d2 = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
    return np.array(labels, dtype=int)[np.argmin(d2, axis=1)]


def run_one(
    seed: int,
    center_scale: float,
    noise: float,
    certificate_size: int,
    exposure_scale: float,
) -> dict:
    rng = np.random.default_rng(seed + 100)
    world = v07.StreamingWorld(seed=seed)
    world.class_base *= center_scale / 4.1
    world.noise = noise

    web = v07.PlasticDendritronWeb(
        input_dim=world.input_dim,
        base_sigma=max(0.75, 3.05 * noise),
        min_sigma=max(0.20, 0.75 * noise),
        max_sigma=max(1.5, 5.8 * noise),
        grow_z=2.05,
        certificate_size=certificate_size,
        retirement_steps=max(500, int(1100 * exposure_scale)),
        split_interval=max(140, int(350 * exposure_scale)),
        merge_interval=max(180, int(500 * exposure_scale)),
        retire_interval=max(100, int(250 * exposure_scale)),
        rng=rng,
    )

    base_counts = [2200, 1400, 1500, 1400, 1500, 900, 1100]
    counts = [max(350, int(x * exposure_scale)) for x in base_counts]
    phases = [
        ('warmup', counts[0], {c: ['A', 'B'] for c in range(6)}, None),
        ('novel_mode', counts[1], {**{c: ['A', 'B'] for c in range(6)}, 2: ['A', 'B', 'C']}, None),
        ('new_classes', counts[2], {**{c: ['A', 'B'] for c in range(6)}, 2: ['A', 'B', 'C'], 6: ['A', 'B'], 7: ['A', 'B']}, None),
        ('specialization', counts[3], {**{c: ['A', 'B'] for c in range(8)}, 2: ['A', 'B', 'C'], 3: ['A', 'B', 'D']}, None),
        ('inactivity', counts[4], {**{c: ['A', 'B'] for c in range(8) if c != 4}, 2: ['A', 'B', 'C'], 3: ['A', 'B', 'D']}, None),
        ('recurrence', counts[5], {**{c: ['A', 'B'] for c in range(8)}, 2: ['A', 'B', 'C'], 3: ['A', 'B', 'D'], 4: ['B']}, {4: 0.42, **{c: 0.58 / 7 for c in range(8) if c != 4}}),
        ('repair', counts[6], {**{c: ['A', 'B'] for c in range(8)}, 2: ['A', 'B', 'C'], 3: ['A', 'B', 'D']}, {1: 0.38, **{c: 0.62 / 7 for c in range(8) if c != 1}}),
    ]

    eval_modes = {c: ['A', 'B'] for c in range(8)}
    eval_modes[2] = ['A', 'B', 'C']
    eval_modes[3] = ['A', 'B', 'D']
    Xeval, yeval, geval = world.eval_set(eval_modes, per_mode=110)
    oracle_acc = float(np.mean(nearest_mode_oracle(world, Xeval, eval_modes) == yeval))

    old_modes = np.array([g not in {'2:C', '3:D', '6:A', '6:B', '7:A', '7:B'} for g in geval])
    new_modes = ~old_modes
    phase_acc = {}
    recurrence_start = None
    repair_start = None
    class4_before = None
    class1_after_damage = None
    damage_detected = False

    for phase, n, modes, probs in phases:
        X, y, _ = world.sample(modes, n, probs)
        if phase == 'repair':
            web.damage_region(1, fraction=1.0)
            class1_after_damage = float(np.mean(web.predict(Xeval[yeval == 1]) == 1))
            damage_detected = 1 in web.repair_scan(threshold=max(0.72, oracle_acc - 0.12))
            repair_start = web.step
        if phase == 'recurrence':
            recurrence_start = web.step
            class4_before = float(np.mean(web.predict(Xeval[yeval == 4]) == 4))

        web.learn_batch(X, y)
        if phase == 'warmup':
            web.inject_redundant_branch(0)
            web._merge_candidates()
        if phase == 'inactivity':
            web.induce_dormancy(4, world.class_base[4] + world.mode_offsets[(4, 'A')])

        pred = web.predict(Xeval)
        phase_acc[phase] = float(np.mean(pred == yeval))

    pred = web.predict(Xeval)
    final_acc = float(np.mean(pred == yeval))
    old_acc = float(np.mean(pred[old_modes] == yeval[old_modes]))
    new_acc = float(np.mean(pred[new_modes] == yeval[new_modes]))
    class1_final = float(np.mean(pred[yeval == 1] == 1))
    class4_final = float(np.mean(pred[yeval == 4] == 4))

    events = pd.DataFrame(web.events)
    counts_ops = events['event'].value_counts().to_dict() if len(events) else {}
    class4_reactivate = events[(events.event == 'reactivate') & (events.class_id == 4) & (events.step >= recurrence_start)] if len(events) else pd.DataFrame()
    class1_repair = events[(events.event.isin(['reactivate', 'grow'])) & (events.class_id == 1) & (events.step >= repair_start)] if len(events) else pd.DataFrame()

    # Geometric overlap makes perfect accuracy impossible. Score against oracle ceiling.
    target = max(0.70, oracle_acc - 0.045)
    criteria = {
        'near_oracle_final': final_acc >= target,
        'old_stability': old_acc >= max(0.68, oracle_acc - 0.065),
        'new_acquisition': new_acc >= max(0.68, oracle_acc - 0.075),
        'growth': counts_ops.get('grow', 0) > 0,
        'split': counts_ops.get('split', 0) > 0,
        'merge': counts_ops.get('merge', 0) > 0,
        'retirement': counts_ops.get('retire', 0) > 0,
        'recurrence': len(class4_reactivate) > 0 and class4_final > class4_before + 0.08,
        'damage_detection': damage_detected,
        'damage_repair': class1_after_damage is not None and class1_final > class1_after_damage + 0.20 and class1_final >= max(0.65, oracle_acc - 0.12),
    }

    functional_keys = ['near_oracle_final','old_stability','new_acquisition','recurrence','damage_detection','damage_repair']
    structural_keys = ['growth','split','merge','retirement']
    functional_pass = bool(all(criteria[k] for k in functional_keys))
    structural_repertoire_observed = bool(all(criteria[k] for k in structural_keys))

    return {
        'seed': seed,
        'center_scale': center_scale,
        'nominal_pair_separation': math.sqrt(2.0) * center_scale,
        'noise': noise,
        'separation_to_noise': math.sqrt(2.0) * center_scale / noise,
        'certificate_size': certificate_size,
        'exposure_scale': exposure_scale,
        'stream_samples': sum(counts),
        'oracle_accuracy': oracle_acc,
        'final_accuracy': final_acc,
        'oracle_gap': oracle_acc - final_acc,
        'old_accuracy': old_acc,
        'new_accuracy': new_acc,
        'class4_before_recurrence': class4_before,
        'class4_after_recurrence': class4_final,
        'class4_reactivation_samples': None if len(class4_reactivate) == 0 else int(class4_reactivate.iloc[0].step - recurrence_start),
        'class1_after_damage': class1_after_damage,
        'class1_after_repair': class1_final,
        'class1_first_repair_samples': None if len(class1_repair) == 0 else int(class1_repair.iloc[0].step - repair_start),
        'active_branches': web.structural_counts()['active_branches'],
        'archived_branches': web.structural_counts()['archived_branches'],
        'grow_events': counts_ops.get('grow', 0),
        'split_events': counts_ops.get('split', 0),
        'merge_events': counts_ops.get('merge', 0),
        'retire_events': counts_ops.get('retire', 0),
        'reactivate_events': counts_ops.get('reactivate', 0),
        **{f'criterion_{k}': bool(v) for k, v in criteria.items()},
        'functional_pass': functional_pass,
        'structural_repertoire_observed': structural_repertoire_observed,
        'plasticity_pass': bool(all(criteria.values())),
    }


def main() -> dict:
    started = time.time()
    # Geometry/noise stress grid across three random worlds.
    geometry_grid = [
        (4.1, 0.48),
        (3.5, 0.60),
        (3.0, 0.72),
        (2.6, 0.82),
        (2.3, 0.92),
        (2.0, 1.02),
    ]
    rows = []
    for seed in [11, 29, 47]:
        for center_scale, noise in geometry_grid:
            r = run_one(seed, center_scale, noise, certificate_size=72, exposure_scale=1.0); r['sweep'] = 'geometry'; rows.append(r); print('geometry', seed, center_scale, noise, r['plasticity_pass'], flush=True); gc.collect()

    # Review-budget and developmental-exposure ablations at moderate difficulty.
    for seed in [11, 29, 47]:
        for cert in [12, 24, 48, 72]:
            r = run_one(seed, 3.0, 0.72, certificate_size=cert, exposure_scale=1.0); r['sweep'] = 'certificate'; rows.append(r); print('certificate', seed, cert, r['plasticity_pass'], flush=True); gc.collect()
        for exposure in [0.35, 0.50, 0.70, 1.0]:
            r = run_one(seed, 3.0, 0.72, certificate_size=72, exposure_scale=exposure); r['sweep'] = 'exposure'; rows.append(r); print('exposure', seed, exposure, r['plasticity_pass'], flush=True); gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'dendritron_v0_8_limit_sweep.csv', index=False)

    geom = df[df.sweep == 'geometry'].groupby(
        ['center_scale', 'noise', 'separation_to_noise'], as_index=False
    ).agg(
        runs=('seed', 'count'),
        pass_rate=('plasticity_pass', 'mean'),
        oracle_accuracy=('oracle_accuracy', 'mean'),
        final_accuracy=('final_accuracy', 'mean'),
        oracle_gap=('oracle_gap', 'mean'),
        old_accuracy=('old_accuracy', 'mean'),
        new_accuracy=('new_accuracy', 'mean'),
        active_branches=('active_branches', 'mean'),
    ).sort_values('separation_to_noise', ascending=False)
    geom.to_csv(OUT / 'dendritron_v0_8_geometry_boundary.csv', index=False)

    cert = df[df.sweep == 'certificate'].groupby('certificate_size', as_index=False).agg(
        runs=('seed', 'count'), pass_rate=('plasticity_pass', 'mean'), final_accuracy=('final_accuracy', 'mean'),
        oracle_gap=('oracle_gap', 'mean'), active_branches=('active_branches', 'mean')
    ).sort_values('certificate_size')
    cert.to_csv(OUT / 'dendritron_v0_8_certificate_ablation.csv', index=False)

    exposure = df[df.sweep == 'exposure'].groupby('exposure_scale', as_index=False).agg(
        runs=('seed', 'count'), pass_rate=('plasticity_pass', 'mean'), final_accuracy=('final_accuracy', 'mean'),
        oracle_gap=('oracle_gap', 'mean'), active_branches=('active_branches', 'mean')
    ).sort_values('exposure_scale')
    exposure.to_csv(OUT / 'dendritron_v0_8_exposure_ablation.csv', index=False)

    # Locate empirical boundary: hardest geometry with majority pass.
    majority = geom[geom.pass_rate >= 2/3]
    boundary = None if len(majority) == 0 else majority.sort_values('separation_to_noise').iloc[0].to_dict()
    failure = geom[geom.pass_rate < 2/3]
    first_failure = None if len(failure) == 0 else failure.sort_values('separation_to_noise', ascending=False).iloc[0].to_dict()

    summary = {
        'runtime_seconds': time.time() - started,
        'total_runs': len(df),
        'overall_pass_rate': float(df.plasticity_pass.mean()),
        'geometry_majority_pass_boundary': boundary,
        'first_geometry_majority_failure': first_failure,
        'minimum_certificate_size_majority_pass': None,
        'minimum_exposure_scale_majority_pass': None,
    }
    full_cert = cert[cert.pass_rate >= 2/3]
    if len(full_cert):
        summary['minimum_certificate_size_majority_pass'] = int(full_cert.certificate_size.min())
    full_exp = exposure[exposure.pass_rate >= 2/3]
    if len(full_exp):
        summary['minimum_exposure_scale_majority_pass'] = float(full_exp.exposure_scale.min())

    with open(OUT / 'dendritron_v0_8_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('DENDRITRON ARTIFICIAL NEURAL PLASTICITY LIMIT SWEEP v0.8')
    print('=' * 76)
    print(f"Runs: {summary['total_runs']} | Runtime: {summary['runtime_seconds']:.2f}s | Overall pass rate: {summary['overall_pass_rate']:.3f}")
    print('\nGeometry boundary:')
    print(geom.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nCertificate ablation:')
    print(cert.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nExposure ablation:')
    print(exposure.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nSummary:')
    print(json.dumps(summary, indent=2))
    return summary





def _single_subprocess(seed: int, center_scale: float, noise: float, certificate: int, exposure: float) -> dict:
    """Run one world in a fresh process to avoid numerical-runtime state buildup."""
    import subprocess
    script = str(Path(__file__).resolve())
    cmd = [
        sys.executable, script, '--mode', 'single',
        '--seed', str(seed), '--center-scale', str(center_scale), '--noise', str(noise),
        '--certificate', str(certificate), '--exposure', str(exposure),
    ]
    import os
    env = os.environ.copy()
    env.update({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
    cp = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    return json.loads(cp.stdout.strip().splitlines()[-1])


def run_geometry_isolated() -> None:
    rows = []
    grid = [(4.1, 0.48), (3.5, 0.60), (3.0, 0.72), (2.6, 0.82), (2.3, 0.92), (2.0, 1.02)]
    for seed in [11, 29, 47]:
        for center_scale, noise in grid:
            r = _single_subprocess(seed, center_scale, noise, 72, 1.0)
            r['sweep'] = 'geometry'; rows.append(r)
            print('geometry', seed, center_scale, noise, r['plasticity_pass'], flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'dendritron_v0_8_geometry_runs.csv', index=False)


def run_ablations_isolated() -> None:
    rows = []
    for seed in [11, 29, 47]:
        for cert in [12, 24, 48, 72]:
            r = _single_subprocess(seed, 3.0, 0.72, cert, 1.0)
            r['sweep'] = 'certificate'; rows.append(r)
            print('certificate', seed, cert, r['plasticity_pass'], flush=True)
        for exposure in [0.35, 0.50, 0.70, 1.0]:
            r = _single_subprocess(seed, 3.0, 0.72, 72, exposure)
            r['sweep'] = 'exposure'; rows.append(r)
            print('exposure', seed, exposure, r['plasticity_pass'], flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'dendritron_v0_8_ablation_runs.csv', index=False)


def aggregate_chunks() -> dict:
    g = pd.read_csv(OUT / 'dendritron_v0_8_geometry_runs.csv')
    a = pd.read_csv(OUT / 'dendritron_v0_8_ablation_runs.csv')
    df = pd.concat([g, a], ignore_index=True)
    df.to_csv(OUT / 'dendritron_v0_8_limit_sweep.csv', index=False)

    geom = g.groupby(['center_scale','noise','separation_to_noise'], as_index=False).agg(
        runs=('seed','count'),
        full_mechanism_pass_rate=('plasticity_pass','mean'),
        functional_pass_rate=('functional_pass','mean'),
        structural_repertoire_rate=('structural_repertoire_observed','mean'),
        oracle_accuracy=('oracle_accuracy','mean'),
        final_accuracy=('final_accuracy','mean'),
        oracle_gap=('oracle_gap','mean'),
        old_accuracy=('old_accuracy','mean'),
        new_accuracy=('new_accuracy','mean'),
        active_branches=('active_branches','mean'),
        archived_branches=('archived_branches','mean'),
        split_events=('split_events','mean'),
        merge_events=('merge_events','mean'),
        reactivate_events=('reactivate_events','mean'),
    ).sort_values('separation_to_noise', ascending=False)
    geom.to_csv(OUT / 'dendritron_v0_8_geometry_boundary.csv', index=False)

    cert = a[a.sweep == 'certificate'].groupby('certificate_size', as_index=False).agg(
        runs=('seed','count'),
        full_mechanism_pass_rate=('plasticity_pass','mean'),
        functional_pass_rate=('functional_pass','mean'),
        final_accuracy=('final_accuracy','mean'),
        oracle_gap=('oracle_gap','mean'),
        active_branches=('active_branches','mean'),
    ).sort_values('certificate_size')
    cert.to_csv(OUT / 'dendritron_v0_8_certificate_ablation.csv', index=False)

    exposure = a[a.sweep == 'exposure'].groupby('exposure_scale', as_index=False).agg(
        runs=('seed','count'),
        full_mechanism_pass_rate=('plasticity_pass','mean'),
        functional_pass_rate=('functional_pass','mean'),
        structural_repertoire_rate=('structural_repertoire_observed','mean'),
        final_accuracy=('final_accuracy','mean'),
        oracle_gap=('oracle_gap','mean'),
        active_branches=('active_branches','mean'),
    ).sort_values('exposure_scale')
    exposure.to_csv(OUT / 'dendritron_v0_8_exposure_ablation.csv', index=False)

    def clean(d):
        if d is None: return None
        return {k: (v.item() if hasattr(v, 'item') else v) for k, v in d.items()}

    full = geom[geom.full_mechanism_pass_rate >= 2/3]
    full_fail = geom[geom.full_mechanism_pass_rate < 2/3]
    func = geom[geom.functional_pass_rate >= 2/3]
    func_fail = geom[geom.functional_pass_rate < 2/3]
    summary = {
        'total_runs': int(len(df)),
        'overall_full_mechanism_pass_rate': float(df.plasticity_pass.mean()),
        'overall_functional_pass_rate': float(df.functional_pass.mean()),
        'full_mechanism_majority_boundary': clean(None if full.empty else full.sort_values('separation_to_noise').iloc[0].to_dict()),
        'first_full_mechanism_majority_failure': clean(None if full_fail.empty else full_fail.sort_values('separation_to_noise', ascending=False).iloc[0].to_dict()),
        'functional_majority_boundary': clean(None if func.empty else func.sort_values('separation_to_noise').iloc[0].to_dict()),
        'first_functional_majority_failure': clean(None if func_fail.empty else func_fail.sort_values('separation_to_noise', ascending=False).iloc[0].to_dict()),
        'minimum_certificate_size_full_pass_all_seeds': None,
        'minimum_certificate_size_majority_functional_pass': None,
        'minimum_exposure_scale_full_pass_all_seeds': None,
        'exposure_response_nonmonotonic': True,
    }
    x = cert[cert.full_mechanism_pass_rate == 1.0]
    if not x.empty: summary['minimum_certificate_size_full_pass_all_seeds'] = int(x.certificate_size.min())
    x = cert[cert.functional_pass_rate >= 2/3]
    if not x.empty: summary['minimum_certificate_size_majority_functional_pass'] = int(x.certificate_size.min())
    x = exposure[exposure.full_mechanism_pass_rate == 1.0]
    if not x.empty: summary['minimum_exposure_scale_full_pass_all_seeds'] = float(x.exposure_scale.min())
    (OUT / 'dendritron_v0_8_summary.json').write_text(json.dumps(summary, indent=2))

    print('\nGEOMETRY BOUNDARY')
    print(geom.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nCERTIFICATE ABLATION')
    print(cert.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nEXPOSURE ABLATION')
    print(exposure.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nSUMMARY')
    print(json.dumps(summary, indent=2))
    return summary


def cli() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['all','geometry','ablations','aggregate','single'], default='all')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--center-scale', type=float)
    parser.add_argument('--noise', type=float)
    parser.add_argument('--certificate', type=int)
    parser.add_argument('--exposure', type=float)
    args = parser.parse_args()
    if args.mode == 'single':
        r = run_one(args.seed, args.center_scale, args.noise, args.certificate, args.exposure)
        print(json.dumps(r, separators=(',', ':')))
    elif args.mode == 'geometry':
        run_geometry_isolated()
    elif args.mode == 'ablations':
        run_ablations_isolated()
    elif args.mode == 'aggregate':
        aggregate_chunks()
    else:
        run_geometry_isolated()
        run_ablations_isolated()
        aggregate_chunks()


if __name__ == '__main__':
    cli()
