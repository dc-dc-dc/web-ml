import { Tensor, float32 } from "../src/tensor";
import {test, expect} from "vitest";

async function equalTensor(a: Tensor, b: Tensor) {
  expect(a.shape).toEqual(b.shape);
  expect(a.dtype).toEqual(b.dtype);
  expect(await a.list()).toEqual(await b.list());
}

test("add", async () => {
  const a = new Tensor({ shape: [2, 2], dtype: float32, data: [1, 2, 3, 4] });
  const b = new Tensor({ shape: [2, 2], dtype: float32, data: [5, 6, 7, 8] });
  const c = a.add(b);
  await equalTensor(c, new Tensor({ shape: [2, 2], dtype: float32, data: [6, 8, 10, 12] }));
});

test("sub", async() => {
  const a = new Tensor({ shape: [2, 2], dtype: float32, data: [1, 2, 3, 4] });
  const b = new Tensor({ shape: [2, 2], dtype: float32, data: [5, 6, 7, 8] });
  const c = a.sub(b);
  await equalTensor(c, new Tensor({ shape: [2, 2], dtype: float32, data: [-4, -4, -4, -4] }));
});

test("mul", async () => {
  const a = new Tensor({ shape: [2, 2], dtype: float32, data: [1, 2, 3, 4] });
  const b = new Tensor({ shape: [2, 2], dtype: float32, data: [5, 6, 7, 8] });
  const c = a.mul(b);
  await equalTensor(c, new Tensor({ shape: [2, 2], dtype: float32, data: [5, 12, 21, 32] }));
});

test("div", async () => {
  const a = new Tensor({ shape: [2, 2], dtype: float32, data: [5, 10, 15, 4] });
  const b = new Tensor({ shape: [2, 2], dtype: float32, data: [5, 2, 3, 1] });
  const c = a.div(b);
  await equalTensor(c, new Tensor({ shape: [2, 2], dtype: float32, data: [1, 5, 5, 4] }));
});

test("max", async () => {
  const a = new Tensor({ shape: [2, 2], dtype: float32, data: [5, 10, 15, 4] });
  const b = new Tensor({ shape: [2, 2], dtype: float32, data: [5, 2, 3, 8] });
  const c = a.max(b);
  await equalTensor(c, new Tensor({ shape: [2, 2], dtype: float32, data: [5, 10, 15, 8] }));
});

test("min", async () => {
  const a = new Tensor({ shape: [2, 2], dtype: float32, data: [5, 10, 15, 4] });
  const b = new Tensor({ shape: [2, 2], dtype: float32, data: [5, 2, 3, 8] });
  const c = a.min(b);
  await equalTensor(c, new Tensor({ shape: [2, 2], dtype: float32, data: [5, 2, 3, 4] }));
});

// test("pow", async () => {
//   const a = new Tensor({ shape: [2, 2], dtype: float32, data: [5, 10, 15, 4] });
//   const b = new Tensor({ shape: [2, 2], dtype: float32, data: [5, 2, 3, 8] });
//   const c = a.pow(b);
//   await equalTensor(c, new Tensor({ shape: [2, 2], dtype: float32, data: [3125, 100, 3375, 65536] }));
// });