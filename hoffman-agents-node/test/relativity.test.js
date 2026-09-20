const test = require('node:test');
const assert = require('node:assert');
const exp18 = require('../examples/18_relativity_at_infinity/relativity_at_infinity');

const equalTau = [{ t: 0.6, x: 0 }, { t: 0.75, x: 0.45 }, { t: 1, x: 0.8 }];

test('special functions: Bessel and erfc reference values', () => {
  assert.ok(Math.abs(exp18.besselI(0, 3) - 4.880792585865024) < 1e-12);
  assert.ok(Math.abs(exp18.besselI(1, 3) - 3.953370217402609) < 1e-12);
  assert.ok(Math.abs(exp18.besselJ0(3) - -0.2600519549019334) < 1e-12);
  assert.ok(Math.abs(exp18.erfc(1) - 0.157299207050285) < 2e-7);
});

test('clock converges to a function of proper time, error ~ 1/n', () => {
  const err = (n) => Math.max(...exp18.evolve(n, equalTau).marks.map(k => Math.abs(k.E - exp18.clockPrediction(exp18.A_RATE, k.tau))));
  const e80 = err(80), e320 = err(320);
  assert.ok(e320 < 0.01);
  assert.ok(e80 / e320 > 3.5 && e80 / e320 < 5);
});

test('time dilation: clock readings at speeds 0.6 and 0.8 give proper time sqrt(1 - v^2)', () => {
  const marks = exp18.evolve(1280, [{ t: 1, x: 0.6 }, { t: 1, x: 0.8 }]).marks;
  assert.ok(Math.abs(exp18.properTimeFromClock(exp18.A_RATE, marks[0].E) - 0.8) < 1e-3);
  assert.ok(Math.abs(exp18.properTimeFromClock(exp18.A_RATE, marks[1].E) - 0.6) < 1e-3);
});

test('classical density keeps a preferred frame; quantum amplitude is a function of tau; unitary', () => {
  const { marks, norm } = exp18.evolve(1280, equalTau);
  marks.forEach(k => {
    assert.ok(Math.abs(k.P - exp18.densityPrediction(exp18.A_RATE, k.t, k.tau)) < 0.01);
    assert.ok(Math.abs(k.Q - exp18.amplitudePrediction(exp18.MASS, k.tau)) < 0.02);
    assert.strictEqual(k.Qre, 0);
  });
  assert.ok(marks[0].P / marks[2].P > 5);
  assert.ok(Math.abs(norm - 1) < 1e-12);
});

test('light cone: memoryless agent escapes any finite speed, tends to diffusion', () => {
  const m = exp18.memorylessOutsideCone(1280);
  assert.ok(m.maxSpeed > 15);
  assert.ok(Math.abs(m.outside - exp18.erfc(Math.sqrt(exp18.A_RATE / 2))) < 0.003);
});
