import importlib.util
import math
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "exp18", Path(__file__).resolve().parents[1] / "examples" / "18_relativity_at_infinity" / "relativity_at_infinity.py")
exp18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp18)
EQUAL_TAU = exp18.EQUAL_TAU


def test_special_functions():
    assert abs(exp18.bessel_i(0, 3) - 4.880792585865024) < 1e-12
    assert abs(exp18.bessel_i(1, 3) - 3.953370217402609) < 1e-12
    assert abs(exp18.bessel_j0(3) - -0.2600519549019334) < 1e-12
    assert abs(exp18.erfc(1) - math.erfc(1)) < 2e-7


def test_clock_converges_like_one_over_n():
    def err(n):
        return max(abs(k["E"] - exp18.clock_prediction(exp18.A_RATE, k["tau"])) for k in exp18.evolve(n, EQUAL_TAU)["marks"])
    e80, e320 = err(80), err(320)
    assert e320 < 0.01
    assert 3.5 < e80 / e320 < 5


def test_time_dilation():
    marks = exp18.evolve(1280, [{"t": 1, "x": 0.6}, {"t": 1, "x": 0.8}])["marks"]
    assert abs(exp18.proper_time_from_clock(exp18.A_RATE, marks[0]["E"]) - 0.8) < 1e-3
    assert abs(exp18.proper_time_from_clock(exp18.A_RATE, marks[1]["E"]) - 0.6) < 1e-3


def test_preferred_frame_and_unitarity():
    r = exp18.evolve(1280, EQUAL_TAU)
    for k in r["marks"]:
        assert abs(k["P"] - exp18.density_prediction(exp18.A_RATE, k["t"], k["tau"])) < 0.01
        assert abs(k["Q"] - exp18.amplitude_prediction(exp18.MASS, k["tau"])) < 0.02
        assert k["Qre"] == 0
    assert r["marks"][0]["P"] / r["marks"][2]["P"] > 5
    assert abs(r["norm"] - 1) < 1e-12


def test_memoryless_escapes_cone():
    m = exp18.memoryless_outside_cone(1280)
    assert m["max_speed"] > 15
    assert abs(m["outside"] - exp18.erfc(math.sqrt(exp18.A_RATE / 2))) < 0.003
