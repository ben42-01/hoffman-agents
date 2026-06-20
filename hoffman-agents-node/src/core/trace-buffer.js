const crypto = require('crypto');

class TraceEvent {
  constructor(fromState, toState, timestamp, prediction, predictionCorrect, predictionError, token) {
    this.fromState = fromState;
    this.toState = toState;
    this.timestamp = timestamp;
    this.prediction = prediction;
    this.predictionCorrect = predictionCorrect;
    this.predictionError = predictionError;
    this.token = token ?? null;
    if (predictionError < 0 || predictionError > 1) {
      throw new Error(`prediction_error must be in [0, 1], got ${predictionError}`);
    }
  }
}

class TraceBuffer {
  constructor(maxlen = 50) {
    if (maxlen < 1) throw new Error(`maxlen must be >= 1, got ${maxlen}`);
    this._maxlen = maxlen;
    this._events = [];
    this._cursor = 0;
  }

  append(event) {
    if (this._events.length < this._maxlen) {
      this._events.push(event);
    } else {
      this._events[this._cursor] = event;
      this._cursor = (this._cursor + 1) % this._maxlen;
    }
  }

  getRecent(n) {
    if (n <= 0) return [];
    n = Math.min(n, this._events.length);
    if (this._events.length < this._maxlen) {
      return this._events.slice(-n);
    }
    const ordered = this._events.slice(this._cursor).concat(this._events.slice(0, this._cursor));
    return ordered.slice(-n);
  }

  asStateSequence() {
    return this._events.map(e => e.toState);
  }

  predictionErrorMean(window = 20) {
    const recent = this.getRecent(window);
    if (recent.length === 0) return 0;
    return recent.reduce((s, e) => s + e.predictionError, 0) / recent.length;
  }

  isFull() { return this._events.length >= this._maxlen; }

  get length() { return this._events.length; }
  get maxlen() { return this._maxlen; }

  *[Symbol.iterator]() {
    if (this._events.length < this._maxlen) {
      yield* this._events;
    } else {
      yield* this._events.slice(this._cursor);
      yield* this._events.slice(0, this._cursor);
    }
  }

  clear() {
    this._events = [];
    this._cursor = 0;
  }

  static fromEvents(events, maxlen) {
    const buf = new TraceBuffer(maxlen ?? events.length);
    for (const e of events) buf.append(e);
    return buf;
  }
}

module.exports = { TraceEvent, TraceBuffer };
