import {test, expect} from "vitest";
import { Tensor } from "../src/tensor";

test("list", async () => {
    expect(() => new Tensor({ shape: [2, 2], data: [1, 2, 3, 4, 5, 6] })).toThrow();

    let a = new Tensor({ shape: [2, 2], data: [1, 2, 3, 4] });
    expect(await a.list()).toEqual([[1, 2], [3, 4]]);
    a = new Tensor({ shape: [10], data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] });
    expect(await a.list()).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
});