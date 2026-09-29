"""Physical and numerical self-checks plus a small-scale reproduction of the experiment driver."""
import json, numpy as np, pytest
from cn0sampling import simulation as sim
from cn0sampling import run_experiments as exp

def test_self_checks():
    sim.self_checks()

def test_horizontal_ground_null_is_exact():
    s = sim.make_scene(2000, 7)
    d = sim.signal(s, (0.5, 0, 0)) - sim.signal(s)
    assert np.max(np.abs(d[s['y'] == 1])) < 1e-12          # MPG
    assert np.max(np.abs(d[np.isin(s['y'], [0, 3])])) < 1e-12  # LOS, NLOS

def test_vertical_ground_phase_step():
    el = np.deg2rad(30.0); u = np.array([[0, np.cos(el), np.sin(el)]])
    step = sim.phase_step((0, 0, 0.10), u, np.array([0, 0, 1.0]))
    assert np.isclose(abs(step[0]), 4*np.pi*0.10*np.sin(el)/sim.LAM_L1)

def test_bayes_error_below_chance_when_D_below_one():
    D = sim.rms_delta_db(0.05, np.pi) / np.sqrt(2*0.5**2 + 0.25**2/6)
    assert D < 1
    assert sim.bayes_error(0.05, np.pi, 0.5, 0.25) < 0.5

def test_small_run(tmp_path):
    R = exp.run_all(n=400, seeds=(0,), output_dir=tmp_path)
    assert (tmp_path / 'results.json').exists()
    assert set(R['E3']) == set(exp.CONFIGS)
    assert all((tmp_path/'tables'/f).exists() for f in ['table_e2.tex','table_e3.tex','table_e5.tex','table_recall.tex'])

def test_full_results_match_archived(tmp_path):
    """Regenerates the archived results with the archived protocol and compares (slow: ~15 s)."""
    ref = json.load(open('results/results.json'))
    R = exp.run_all(n=ref['protocol']['n_train'], seeds=tuple(ref['protocol']['seeds']), output_dir=tmp_path)
    for name in ref['E3']:
        for tag in ref['E3'][name]:
            assert abs(R['E3'][name][tag]['mean'] - ref['E3'][name][tag]['mean']) < 1e-9
    for a, b in zip(R['E4'], ref['E4']):
        assert abs(a['bayes_error'] - b['bayes_error']) < 1e-9
