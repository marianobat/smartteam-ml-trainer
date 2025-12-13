const HAND_SIZE = 21;
const FEATURES_PER_HAND = HAND_SIZE * 3; // 63
const LEFT_FLAG_IDX = 126;
const RIGHT_FLAG_IDX = 127;

function normalizeHand(
  src: Float32Array | number[],
  dst: Float32Array,
  handOffset: number,
  presentFlag: number
) {
  const present = presentFlag !== 0;
  if (!present) {
    dst.fill(0, handOffset, handOffset + FEATURES_PER_HAND);
    return;
  }

  const wristX = src[handOffset];
  const wristY = src[handOffset + 1];
  const wristZ = src[handOffset + 2];

  const middleIdx = handOffset + 9 * 3; // middle_mcp
  const dx = src[middleIdx] - wristX;
  const dy = src[middleIdx + 1] - wristY;
  const dz = src[middleIdx + 2] - wristZ;
  const scale = Math.max(Math.hypot(dx, dy, dz), 1e-6);

  for (let j = 0; j < HAND_SIZE; j++) {
    const idx = handOffset + j * 3;
    dst[idx] = (src[idx] - wristX) / scale;
    dst[idx + 1] = (src[idx + 1] - wristY) / scale;
    dst[idx + 2] = (src[idx + 2] - wristZ) / scale;
  }
}

export function normalize128(vec: Float32Array | number[]): Float32Array {
  const dst = new Float32Array(128);
  const leftFlag = vec[LEFT_FLAG_IDX];
  const rightFlag = vec[RIGHT_FLAG_IDX];

  normalizeHand(vec, dst, 0, leftFlag);
  normalizeHand(vec, dst, FEATURES_PER_HAND, rightFlag);

  dst[LEFT_FLAG_IDX] = leftFlag;
  dst[RIGHT_FLAG_IDX] = rightFlag;
  return dst;
}
