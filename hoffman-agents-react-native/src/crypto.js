/**
 * Pure-JS crypto shim for React Native compatibility.
 * Replaces Node's built-in `crypto` module.
 *
 * NOT cryptographically secure — these are for deterministic
 * agent state IDs, not security.
 */

function randomBytes(size) {
  const bytes = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    bytes[i] = Math.floor(Math.random() * 256);
  }
  // Add Array-like behavior for .readUInt32BE compatibility
  bytes.readUInt32BE = function (offset) {
    return ((this[offset] << 24) | (this[offset + 1] << 16) |
            (this[offset + 2] << 8) | this[offset + 3]) >>> 0;
  };
  bytes.toString = function (enc) {
    if (enc === 'hex') {
      let hex = '';
      for (let i = 0; i < this.length; i++) {
        hex += ('0' + this[i].toString(16)).slice(-2);
      }
      return hex;
    }
    return Array.prototype.join.call(this, ',');
  };
  return bytes;
}

function createHash(algorithm) {
  if (algorithm !== 'sha256') {
    throw new Error(`Unsupported hash algorithm: ${algorithm}`);
  }
  let data = '';
  return {
    update(d) {
      data += typeof d === 'string' ? d : String.fromCharCode.apply(null, d);
      return this;
    },
    digest(encoding) {
      const hash = _fnv1a(data);
      const hex = ('00000000' + hash.toString(16)).slice(-8);
      const fullHex = hex.repeat(8);
      if (encoding === 'hex') return fullHex;
      const buf = _hexToBytes(fullHex);
      buf.readUInt32BE = function (offset) {
        return ((this[offset] << 24) | (this[offset + 1] << 16) |
                (this[offset + 2] << 8) | this[offset + 3]) >>> 0;
      };
      return buf;
    },
  };
}

function randomUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

function _fnv1a(str) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

function _hexToBytes(hex) {
  const len = Math.floor(hex.length / 2);
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = parseInt(hex.substr(i * 2, 2), 16);
  }
  bytes.readUInt32BE = function (offset) {
    return ((this[offset] << 24) | (this[offset + 1] << 16) |
            (this[offset + 2] << 8) | this[offset + 3]) >>> 0;
  };
  return bytes;
}

module.exports = { randomBytes, createHash, randomUUID };
