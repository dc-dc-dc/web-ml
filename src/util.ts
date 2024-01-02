import type { Shape } from "./web-ml";

export function isSameShape(x: Shape, y: Shape): boolean {
  if (x.length !== y.length) {
    return false;
  }
  for (let i = 0; i < x.length; i++) {
    if (x[i] !== y[i]) {
      return false;
    }
  }
  return true;
}

export function broadcastShape(x: Shape, y: Shape): Shape {
  let ndim1 = x.length;
  let ndim2 = y.length;
  let ndim = Math.max(ndim1, ndim2);
  let diff = Math.abs(ndim1 - ndim2);
  let big = ndim1 > ndim2 ? x : y;
  let small = ndim1 > ndim2 ? y : x;
  let out: Shape = Array.from({ length: ndim }, () => 0);
  for (let i = ndim - 1; i >= diff; --i) {
    let a = big[i];
    let b = small[i - diff];
    if (b == a) {
      out[i] = a;
    } else if (a == 1 || b == 1) {
      out[i] = a * b;
    } else {
      throw new Error(`${x} and ${y} cannot be broadcasted`);
    }
  }
  for (let i = diff - 1; i >= 0; --i) {
    out[i] = big[i];
  }
  return out;
}