import {test, expect} from "vitest";
import { Tensor, broadcast_shape } from "../src/tensor";

test("broadcast_shape", async () => {
    expect(broadcast_shape([2, 2], [2, 2])).toEqual([2, 2]);
    expect(broadcast_shape([2, 2], [2])).toEqual([2, 2]);
    expect(broadcast_shape([2, 2], [1])).toEqual([2, 2]);
    expect(broadcast_shape([1], [2, 2])).toEqual([2, 2]);
    expect(broadcast_shape([2, 2], [])).toEqual([2, 2]);
    expect(broadcast_shape([2, 2], [2, 1])).toEqual([2, 2]);
    expect(broadcast_shape([2, 2], [1, 2])).toEqual([2, 2]);
    expect(broadcast_shape([2, 2], [1, 1])).toEqual([2, 2]);
    expect(broadcast_shape([2, 2], [2, 2, 2])).toEqual([2, 2, 2]);
    expect(() => broadcast_shape([4, 3], [4])).toThrow();
});

test("strides", async () => {
    let a = new Tensor({ shape: [2, 2], data: [1, 2, 3, 4] });
    expect(a.strides).toEqual([2, 1]);
    a = new Tensor({ shape: [4, 2, 2], data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] });
    expect(a.strides).toEqual([4, 2, 1]);
});

test("list", async () => {
    expect(() => new Tensor({ shape: [2, 2], data: [1, 2, 3, 4, 5, 6] })).toThrow();

    let a = new Tensor({ shape: [2, 2], data: [1, 2, 3, 4] });
    expect(await a.list()).toEqual([[1, 2], [3, 4]]);
    a = new Tensor({ shape: [10], data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] });
    expect(await a.list()).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    a = new Tensor({ shape: [2, 2, 2], data: [1, 2, 3, 4, 5, 6, 7, 8] });
    expect(await a.list()).toEqual([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    a = new Tensor({ shape: [2, 2, 2, 2], data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] });
    expect(await a.list()).toEqual([
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
    ]);
});