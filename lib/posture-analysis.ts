export type Landmark = { x: number; y: number; z?: number; visibility?: number };

export type PostureFeatures = {
  headForward: number;
  torsoDrop: number;
  torsoShift: number;
  shoulderTilt: number;
  movement: number;
};

export type Baseline = Omit<PostureFeatures, "movement">;

export type PosturePattern =
  | "forward-head"
  | "slouching"
  | "uneven-shoulders"
  | "stillness";

export type Thresholds = {
  head: number;
  slouch: number;
  shoulders: number;
  stillness: number;
};

export type ScoreSmoothingState = {
  samples: number[];
  filtered: number;
  displayed: number;
  lastFrameMs: number;
  lastPublishMs: number;
};

const mean = (a: number, b: number) => (a + b) / 2;
const distance = (a: Landmark, b: Landmark) =>
  Math.hypot(a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0));

/**
 * Turns camera coordinates into body-relative features. Shoulder width is the
 * scale reference, so distance from the camera and body size do not create
 * universal cutoffs. MediaPipe indices: ears 7/8, shoulders 11/12, hips 23/24.
 */
export function extractFeatures(
  points: Landmark[],
  previous?: PostureFeatures,
): PostureFeatures | null {
  if (points.length < 25) return null;
  const [le, re, ls, rs, lh, rh] = [
    points[7], points[8], points[11], points[12], points[23], points[24],
  ];
  if ([le, re, ls, rs].some((p) => !p || (p.visibility ?? 1) < 0.35)) return null;

  const shoulderWidth = Math.max(distance(ls, rs), 0.05);
  const earX = mean(le.x, re.x);
  const shoulderX = mean(ls.x, rs.x);
  const earZ = mean(le.z ?? 0, re.z ?? 0);
  const shoulderZ = mean(ls.z ?? 0, rs.z ?? 0);
  const shoulderY = mean(ls.y, rs.y);
  const hipY = lh && rh ? mean(lh.y, rh.y) : shoulderY + shoulderWidth * 1.6;

  // Z is the best frontal-camera proxy for "ahead"; a small X term also
  // supports side-on setups. Both are compared only with this user's baseline.
  const headForward = (shoulderZ - earZ) / shoulderWidth +
    Math.abs(earX - shoulderX) / shoulderWidth * 0.22;
  const torsoDrop = shoulderY;
  const torsoShift = (shoulderX - mean(lh?.x ?? shoulderX, rh?.x ?? shoulderX)) / shoulderWidth;
  const shoulderTilt = Math.abs(ls.y - rs.y) / shoulderWidth;
  const currentCore = [headForward, torsoDrop, torsoShift, shoulderTilt, hipY];
  const priorCore = previous
    ? [previous.headForward, previous.torsoDrop, previous.torsoShift, previous.shoulderTilt, hipY]
    : currentCore;
  const movement = currentCore.reduce((sum, value, i) => sum + Math.abs(value - priorCore[i]), 0);

  return { headForward, torsoDrop, torsoShift, shoulderTilt, movement };
}

/** Exponential smoothing prevents one-frame landmark jitter from becoming feedback. */
export function smoothFeatures(
  previous: PostureFeatures | null,
  next: PostureFeatures,
  alpha = 0.18,
): PostureFeatures {
  if (!previous) return next;
  return Object.fromEntries(
    Object.keys(next).map((key) => {
      const k = key as keyof PostureFeatures;
      return [k, previous[k] * (1 - alpha) + next[k] * alpha];
    }),
  ) as PostureFeatures;
}

/** Calibration is a median of samples, making it resistant to brief tracking jumps. */
export function calibrate(samples: PostureFeatures[]): Baseline | null {
  if (!samples.length) return null;
  const median = (key: keyof Baseline) => {
    const values = samples.map((s) => s[key]).sort((a, b) => a - b);
    return values[Math.floor(values.length / 2)];
  };
  return {
    headForward: median("headForward"),
    torsoDrop: median("torsoDrop"),
    torsoShift: median("torsoShift"),
    shoulderTilt: median("shoulderTilt"),
  };
}

/** All thresholds represent change from the person's own comfortable calibration. */
export function classifyPosture(
  value: PostureFeatures,
  baseline: Baseline,
  thresholds: Thresholds,
  stillForMs: number,
  stillnessLimitMs: number,
): PosturePattern[] {
  const patterns: PosturePattern[] = [];
  if (value.headForward - baseline.headForward > thresholds.head) patterns.push("forward-head");
  if (
    value.torsoDrop - baseline.torsoDrop > thresholds.slouch ||
    Math.abs(value.torsoShift - baseline.torsoShift) > thresholds.slouch
  ) patterns.push("slouching");
  if (value.shoulderTilt - baseline.shoulderTilt > thresholds.shoulders) patterns.push("uneven-shoulders");
  if (value.movement < thresholds.stillness && stillForMs >= stillnessLimitMs) patterns.push("stillness");
  return patterns;
}

/** A wellness indicator, not a clinical measurement or ergonomic certification. */
export function postureScore(value: PostureFeatures, baseline: Baseline, sensitivity = 1): number {
  const deviation =
    Math.max(0, value.headForward - baseline.headForward) * 65 +
    Math.max(0, value.torsoDrop - baseline.torsoDrop) * 150 +
    Math.abs(value.torsoShift - baseline.torsoShift) * 38 +
    Math.max(0, value.shoulderTilt - baseline.shoulderTilt) * 80;
  return Math.round(Math.max(0, Math.min(100, 100 - deviation * sensitivity)));
}

export function createScoreSmoother(initial = 100, now = 0): ScoreSmoothingState {
  return {
    samples: [],
    filtered: initial,
    displayed: initial,
    lastFrameMs: now,
    lastPublishMs: now,
  };
}

/**
 * Stabilizes the user-facing score without slowing posture classification.
 * A rolling median rejects single-frame tracking spikes, the time-based EMA
 * eases genuine changes over roughly 1.2 seconds, and the publishing gate
 * limits the visible number to small updates four times per second.
 */
export function smoothPostureScore(
  rawScore: number,
  previous: ScoreSmoothingState,
  now: number,
): { state: ScoreSmoothingState; display: number | null } {
  const samples = [...previous.samples, rawScore].slice(-15);
  const ordered = [...samples].sort((a, b) => a - b);
  const median = ordered[Math.floor(ordered.length / 2)];
  const elapsed = Math.min(Math.max(now - previous.lastFrameMs, 0), 100);
  const alpha = 1 - Math.exp(-elapsed / 1200);
  const filtered = previous.filtered + (median - previous.filtered) * alpha;

  if (now - previous.lastPublishMs < 250) {
    return {
      state: { ...previous, samples, filtered, lastFrameMs: now },
      display: null,
    };
  }

  const target = Math.round(filtered);
  const difference = target - previous.displayed;
  // A one-point dead zone avoids flicker around rounding boundaries.
  const displayed = Math.abs(difference) <= 1
    ? previous.displayed
    : previous.displayed + Math.sign(difference) * Math.min(2, Math.abs(difference));

  return {
    state: { samples, filtered, displayed, lastFrameMs: now, lastPublishMs: now },
    display: displayed === previous.displayed ? null : displayed,
  };
}

export function canAlert(
  pattern: PosturePattern,
  now: number,
  lastAlerts: Partial<Record<PosturePattern, number>>,
  cooldownMs: number,
) {
  return now - (lastAlerts[pattern] ?? 0) >= cooldownMs;
}
