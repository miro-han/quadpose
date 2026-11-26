"""Independent autonomous rollouts and label-only scoring for Phase IV."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter_ns

from evaluation.pose_benchmark.benchmark_data import CONFIG_PATH, config, np, trial
from evaluation.layout_pose import benchmark as reference
from pose.fusion import (
    EDGES, AssociationLabels, analyze_frame, associate_view, build_problem, choose_result,
    oracle_pairs, solve_pose, view_match,
)

EMPTY = tuple(np.empty((0, 2), dtype=int) for _ in range(4))
STAGES = ('association', 'hypothesis', 'verification', 'refinement', 'observability')
RECOVERY_REGIMES = ('reacquisition', 'rapid_rotation', 'single_camera_fault', 'multiple_camera_faults', 'combined')


@dataclass
class RunResult:
    pose: object
    measured: bool
    pairs: tuple
    status: str
    reason: str
    stage_ms: dict
    iterations: int = 0
    rank: int | None = None
    min_eigenvalue: float | None = None
    condition: float | None = None
    correspondences_changed: float | None = None
    objective_first: float | None = None
    objective_last: float | None = None
    converged: bool | None = None
    details: dict | None = None


def ms(start):
    return (perf_counter_ns() - start) / 1e6


def getter(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def native_result(result, *, supplemental=None):
    diagnostics = getter(result, 'diagnostics') or {}
    stages = dict(getter(result, 'stage_ms') or getter(diagnostics, 'stage_ms') or getter(diagnostics, 'timings_ms') or {})
    stages.update({key: getter(diagnostics, f'{key}_ms') for key in STAGES if getter(diagnostics, f'{key}_ms') is not None})
    selection = getter(diagnostics, 'selection_ms')
    if selection is not None and 'verification' not in stages:
        stages['verification'] = max(0., selection - stages.get('refinement', 0.))
    for standard, aliases in {'hypothesis': ('hypotheses', 'hypothesis_generation'),
                              'refinement': ('fit', 'refine'), 'verification': ('selection', 'verify')}.items():
        if standard not in stages:
            for alias in aliases:
                if alias in stages:
                    stages[standard] = stages[alias]
                    break
    if supplemental:
        stages.update({key: value for key, value in supplemental.items() if key not in stages})
    history = getter(result, 'history') or getter(diagnostics, 'history') or ()
    pairs = getter(result, 'segment_pairs')
    if pairs is None:
        pairs = getter(result, 'edge_pairs', EMPTY)
    measured = bool(result.valid)
    status = getter(result, 'status', 'measured' if measured else 'lost')
    if not measured:
        pairs = EMPTY
    support = getter(result, 'support')
    if support is not None and len(support) == len(pairs):
        pairs = tuple(pair if flag else pair[:0] for flag, pair in zip(support, pairs, strict=True))
    reason = str(getter(result, 'reason') or '')
    details = {key: getter(diagnostics, key) for key in ('omitted', 'held_out_residual_px', 'held_out_supported',
                                                       'refinement_status', 'recovery_attempts', 'pnp_proposals', 'prior_source')}
    details['reinitialized'] = getter(result, 'reacquired', False)
    details['cycle'] = reason == 'cycle' or details['refinement_status'] == 'cycle'
    converged = getter(diagnostics, 'converged')
    if converged is None and len(history) > 1:
        converged = reason == 'converged'
    return RunResult(result.pose, measured, pairs, str(status), reason, stages,
                     int(getter(diagnostics, 'iterations', len(history))), getter(diagnostics, 'rank'),
                     getter(diagnostics, 'min_eigenvalue'), getter(diagnostics, 'condition'),
                     max((float(getter(step, 'correspondence_changes', 0)) for step in history[1:]), default=None),
                     getter(history[0], 'objective') if history else None,
                     getter(history[-1], 'objective') if history else None,
                     converged, details)


def unchanged_phase2(name, prior, cameras, data, samples):
    if name == 'phase2_best':
        start = perf_counter_ns()
        evidence = analyze_frame(prior, cameras, data, reference.BOX)
        fit_time = ms(start)
        select = perf_counter_ns()
        chosen = choose_result(evidence, 'best')
        pairs = tuple(match.segment_pairs if keep else match.segment_pairs[:0]
                      for keep, match in zip(chosen.support, evidence.matches, strict=True))
        return RunResult(chosen.pose, chosen.valid, pairs if chosen.valid else EMPTY,
                         'measured' if chosen.valid else 'lost', '', {'hypothesis': fit_time, 'verification': ms(select)})

    start = perf_counter_ns()
    if name == 'oracle_all':
        claims = [oracle_pairs(AssociationLabels(sample.corner_labels, sample.edge_labels)) for sample in samples]
    else:
        claims = [associate_view(prior, camera, measured, reference.BOX)
                  for camera, measured in zip(cameras, data, strict=True)]
    matches = tuple(view_match(camera, measured, *pairs) for camera, measured, pairs in zip(cameras, data, claims, strict=True))
    association = ms(start)
    start = perf_counter_ns()
    fit = solve_pose(prior, build_problem(reference.BOX, matches))
    return RunResult(fit.pose if fit.valid else prior, fit.valid, tuple(match.segment_pairs for match in matches) if fit.valid else EMPTY,
                     'measured' if fit.valid else 'lost', '', {'association': association, 'refinement': ms(start)})


def run_method(name, prior, cameras, samples, tracker, stamp):
    data = tuple(sample.detections for sample in samples)
    if name.startswith(('phase2_', 'oracle_')):
        return unchanged_phase2(name, prior, cameras, data, samples)
    if tracker is not None:
        return native_result(tracker.step(stamp, cameras, data))
    if name.startswith('cross'):
        from pose.hypotheses import evaluate_hypotheses, select_pose
        start = perf_counter_ns()
        matrix = evaluate_hypotheses(prior, cameras, data, reference.BOX)
        generated_ms = ms(start)
        selected = select_pose(matrix, policy=name)
        return native_result(selected, supplemental={'hypothesis': generated_ms})

    from pose.refinement import Config, refine_frame
    stage, assignment, edge_mode, topology, fragments = {
        'single_legacy': ('baseline', 'greedy', 'legacy', False, False),
        'single_gated': ('baseline', 'greedy', 'gated', False, False),
        'single_hungarian': ('baseline', 'hungarian', 'gated', False, False),
        'single_topology': ('baseline', 'hungarian', 'gated', True, False),
        'single_full': ('baseline', 'hungarian', 'gated', True, True),
        'remove': ('removal', 'hungarian', 'gated', True, True),
        'reassign': ('reassignment', 'hungarian', 'gated', True, True),
        'alternate': ('alternating', 'hungarian', 'gated', True, True),
    }[name]
    options = Config(edge_mode=edge_mode, topology=topology, fragments=fragments)
    return native_result(refine_frame(prior, cameras, data, reference.BOX, method=stage, assignment=assignment,
                                      last_valid=prior, config=options))


def feature_counts(samples, pairs, lost, previous):
    detected = np.zeros((4, len(EDGES)), dtype=bool)
    assigned = np.zeros_like(detected)
    correct = np.zeros_like(detected)
    current = defaultdict(set)
    fragment_claims = correct_fragments = 0
    for view, (sample, assignments) in enumerate(zip(samples, pairs, strict=True)):
        labels = sample.edge_labels
        detected[view, labels[labels >= 0]] = True
        claims = defaultdict(list)
        for identity, observation in np.asarray(assignments).reshape(-1, 2):
            identity, observation = int(identity), int(observation)
            if not 0 <= identity < len(EDGES) or not 0 <= observation < len(labels):
                raise RuntimeError('Matcher returned an ID outside the original input arrays')
            physical = int(labels[observation])
            fragment_claims += 1
            correct_fragments += physical == identity
            claims[identity].append(physical)
            if physical >= 0:
                current[view, physical].add(identity)
        for identity, parts in claims.items():
            assigned[view, identity] = True
            correct[view, identity] = all(part == identity for part in parts)

    pair_claims = pair_correct = pair_available = 0
    for a, b in combinations(range(4), 2):
        pair_claims += np.count_nonzero(assigned[a] & assigned[b])
        pair_correct += np.count_nonzero(correct[a] & correct[b])
        pair_available += np.count_nonzero(detected[a] & detected[b])
    common = set(previous) & set(current)
    rescued = sum(np.count_nonzero(lost[view] & np.delete(correct, view, axis=0).any(axis=0)) for view in range(4))
    counts = dict(edge_assigned=int(assigned.sum()), edge_correct=int(correct.sum()), edge_detected=int(detected.sum()),
                  fragment_claims=fragment_claims, correct_fragments=correct_fragments,
                  pair_claims=int(pair_claims), pair_correct=int(pair_correct), pair_available=int(pair_available),
                  rescue_lost=int(lost.sum()), rescue_correct=int(rescued), stable_claims=sum(previous[key] == current[key] for key in common),
                  stable_eligible=len(common))
    return counts, current


def run_trial(cameras, initial, frames, settings, method, *, tracker=None):
    rows = []
    prior = initial
    previous = {}
    for index, frame in enumerate(frames):
        start = perf_counter_ns()
        result = run_method(method, prior, cameras, frame.samples, tracker, index / settings['cadence_hz'])
        elapsed = ms(start)
        if result.measured:
            prior = result.pose
        mm, deg = reference.errors(prior if result.pose is None else result.pose, frame.truth)
        accurate = mm <= settings['success_translation_mm'] and deg <= settings['success_rotation_deg']
        counts, previous = feature_counts(frame.samples, result.pairs, frame.lost_by_occlusion, previous)
        detail = result.details or {}
        rows.append(dict(frame=index, active=frame.active, measured=result.measured, status=result.status,
                         reason=result.reason, success=result.measured and accurate, false_valid=result.measured and not accurate,
                         translation_mm=mm, rotation_deg=deg, latency_ms=elapsed, iterations=result.iterations,
                         rank=result.rank, min_eigenvalue=result.min_eigenvalue, condition=result.condition,
                         correspondences_changed=result.correspondences_changed, objective_first=result.objective_first,
                         objective_last=result.objective_last, converged=result.converged,
                         omitted_camera=detail.get('omitted'), held_out_residual_px=detail.get('held_out_residual_px'),
                         held_out_supported=detail.get('held_out_supported'), refinement_status=detail.get('refinement_status'),
                         recovery_attempts=' '.join(detail.get('recovery_attempts') or ()), pnp_proposals=detail.get('pnp_proposals'),
                         prior_source=detail.get('prior_source'), reinitialized=detail.get('reinitialized', False), cycle=detail.get('cycle', False),
                         **{f'{name}_ms': result.stage_ms.get(name) for name in STAGES}, **counts))
    return rows


def recovery_onset(rows, settings):
    first = settings['fault_end_exclusive']
    run = settings['recovery_consecutive_measured']
    for index in range(first, len(rows) - run + 1):
        if all(row['success'] for row in rows[index:index + run]):
            return index - first
    return None


def rate(numerator, denominator):
    return 100. * numerator / denominator if denominator else None


def metrics(rows):
    sums = {key: int(sum(item[key] for item in rows)) for key in (
        'edge_assigned', 'edge_correct', 'edge_detected', 'fragment_claims', 'correct_fragments', 'pair_claims', 'pair_correct',
        'pair_available', 'rescue_lost', 'rescue_correct', 'stable_claims', 'stable_eligible', 'measured', 'success', 'false_valid')}
    result = dict(frames=len(rows), **sums, tracking_success_pct=rate(sums['success'], len(rows)),
                  false_valid_pct=rate(sums['false_valid'], sums['measured']), false_valid_per_frame_pct=rate(sums['false_valid'], len(rows)),
                  edge_precision_pct=rate(sums['edge_correct'], sums['edge_assigned']),
                  edge_recall_pct=rate(sums['edge_correct'], sums['edge_detected']),
                  edge_f1_pct=rate(2 * sums['edge_correct'], sums['edge_assigned'] + sums['edge_detected']),
                  incorrect_association_pct=rate(sums['edge_assigned'] - sums['edge_correct'], sums['edge_assigned']),
                  pair_accuracy_pct=rate(sums['pair_correct'], sums['pair_claims']),
                  pair_recall_pct=rate(sums['pair_correct'], sums['pair_available']),
                  rescue_pct=rate(sums['rescue_correct'], sums['rescue_lost']),
                  stability_pct=rate(sums['stable_claims'], sums['stable_eligible']))
    fields = ('translation_mm', 'rotation_deg', 'latency_ms', 'iterations', 'correspondences_changed',
              'objective_first', 'objective_last', 'rank', 'min_eigenvalue', 'condition', *(f'{name}_ms' for name in STAGES))
    for key in fields:
        data = np.array([row[key] for row in rows if row[key] is not None and np.isfinite(row[key])])
        result[f'{key}_p50'] = float(np.quantile(data, .50)) if len(data) else None
        result[f'{key}_p95'] = float(np.quantile(data, .95)) if len(data) else None
        if key in ('translation_mm', 'rotation_deg'):
            result[f'{key}_rmse'] = float(np.sqrt(np.mean(data ** 2)))
    known = [row['converged'] for row in rows if row['converged'] is not None]
    result['converged_pct'] = rate(sum(known), len(known))
    return result


def summary(rows):
    groups = defaultdict(list)
    for row in rows:
        for layout, regime in ((row['layout'], row['regime']), ('pooled', row['regime']), ('pooled', 'pooled')):
            groups[row['method'], layout, regime, 'all_frames'].append(row)
            if row['active']:
                groups[row['method'], layout, regime, 'active'].append(row)
    return [dict(method=method, layout=layout, regime=regime, window=window, **metrics(group))
            for (method, layout, regime, window), group in groups.items()]


def configurations(settings, suite):
    if suite == 'primary':
        return settings['layouts'], settings['regimes'], settings['test_seeds'], settings['primary_methods'], False
    section = settings[f'{suite}_diagnostic']
    modes = section.get('modes')
    return section['layouts'], section['regimes'], settings['development_seeds'], section.get('methods', modes), modes is not None


def process_peak_rss():
    if platform.system() == 'Windows':
        import ctypes as ct
        from ctypes import wintypes as wt

        class MemoryCounters(ct.Structure):
            _fields_ = [('cb', wt.DWORD), ('page_faults', wt.DWORD), *[(key, ct.c_size_t) for key in (
                'peak_working_set', 'working_set', 'peak_paged_pool', 'paged_pool',
                'peak_nonpaged_pool', 'nonpaged_pool', 'pagefile', 'peak_pagefile')]]

        current = ct.WinDLL('kernel32').GetCurrentProcess
        current.restype = wt.HANDLE
        query = ct.WinDLL('psapi').GetProcessMemoryInfo
        query.argtypes = [wt.HANDLE, ct.POINTER(MemoryCounters), wt.DWORD]
        counters = MemoryCounters()
        counters.cb = ct.sizeof(counters)
        return counters.peak_working_set / 2**20 if query(current(), ct.byref(counters), counters.cb) else None
    import resource
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (2**20 if platform.system() == 'Darwin' else 2**10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('primary', 'selection', 'assignment', 'recovery', 'observability'), required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true', help='Run one layout, one seed and one regime for each method')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    settings = config()
    layouts, regimes, seeds, methods, use_tracker = configurations(settings, args.suite)
    if args.smoke:
        layouts, regimes, seeds = list(layouts)[:1], list(regimes)[:1], seeds[:1]
    modules = ('benchmark.py', 'benchmark_data.py', 'benchmark_config.json', 'hypotheses.py', 'refinement.py', 'recovery.py')
    source_hashes = {file: hashlib.sha256(Path(__file__).with_name(file).read_bytes()).hexdigest() for file in modules}
    rows, recovery_rows = [], []
    for test, (layout, regime, seed) in enumerate(product(layouts, regimes, seeds)):
        cameras, initial, frames = trial(seed, regime, settings['layouts'][layout], settings)
        for method in methods:
            tracker = None
            if test == 0:
                if use_tracker:
                    from pose.recovery import RecoveryTracker
                    warmup_tracker = RecoveryTracker(reference.BOX, initial, policy='cross', mode=method)
                else:
                    warmup_tracker = None
                run_method(method, initial, cameras, frames[0].samples, warmup_tracker, 0.)
            if use_tracker:
                from pose.recovery import RecoveryTracker
                tracker = RecoveryTracker(reference.BOX, initial, policy='cross', mode=method)
            frame_rows = run_trial(cameras, initial, frames, settings, method, tracker=tracker)
            labels = dict(method=method, layout=layout, regime=regime, seed=seed)
            rows.extend(dict(labels, **row) for row in frame_rows)
            if regime in RECOVERY_REGIMES:
                onset = recovery_onset(frame_rows, settings)
                recovery_rows.append(dict(labels, recovered=onset is not None, onset_frames=onset,
                                          onset_ms=None if onset is None else 1000 * onset / settings['cadence_hz']))
        print(f'{args.suite}: {layout} {regime} seed {seed}', flush=True)
    args.output.mkdir(parents=True)
    reference.write_csv(args.output / 'frames.csv', rows)
    reference.write_csv(args.output / 'summary.csv', summary(rows))
    if recovery_rows:
        reference.write_csv(args.output / 'recovery.csv', recovery_rows)
    seen_hashes = {file: hashlib.sha256(Path(__file__).with_name(file).read_bytes()).hexdigest() for file in modules}
    if source_hashes != seen_hashes:
        raise RuntimeError('A benchmark source file changed during this run')
    manifest = dict(config_file=CONFIG_PATH.name, config=settings, suite=args.suite, smoke=args.smoke,
                    layouts=list(layouts), regimes=list(regimes), seeds=list(seeds), methods=methods, rows=len(rows),
                    source_sha256=source_hashes, peak_process_rss_mib=process_peak_rss(), warmup_frames_per_method=1,
                    pipeline='CPU NumPy, host wall clock; unisolated host loads',
                    platform=platform.platform(), python=platform.python_version(), numpy=np.__version__)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
