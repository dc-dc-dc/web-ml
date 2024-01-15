import type { Shape, Strides, TypedArray } from "./web-ml";

export function copySingle(x: number, res: TypedArray) {
    for (let i = 0; i < res.length; ++i) {
        res[i] = x;
    }
}

export function copy1D(src: TypedArray, shape: Shape, strides: Strides, res: TypedArray) {
    let sidx = 0;
    let didx = 0;
    for (let i = 0; i < shape[0]; ++i) {
        res[didx++] = src[sidx];
        sidx += strides[0];
    }
}

export function copy2D(src: TypedArray, shape: Shape, strides: Strides, res: TypedArray) {
    let sidx = 0;
    let didx = 0;
    for (let i = 0; i < shape[0]; ++i) {
        for (let j = 0; j < shape[1]; ++j) {
            res[didx++] = src[sidx];
            sidx += strides[1];
        }
        sidx += strides[0] - shape[1] * strides[1];
    }
}

export function copy3D(src: TypedArray, shape: Shape, strides: Strides, res: TypedArray) {
    let sidx = 0;
    let didx = 0;   
    for (let i = 0; i < shape[0]; ++i) {
        for (let j = 0; j < shape[1]; ++j) {
            for (let k = 0; k < shape[2]; ++k) {
                res[didx++] = src[sidx];
                sidx += strides[2];
            }
            sidx += strides[1] - shape[2] * strides[2];
        }
        sidx += strides[0] - shape[1] * strides[1];
    }
}

export function copy4D(src: TypedArray, shape: Shape, strides: Strides, res: TypedArray) {
    let sidx = 0;
    let didx = 0;   
    for (let i = 0; i < shape[0]; ++i) {
        for (let j = 0; j < shape[1]; ++j) {
            for (let k = 0; k < shape[2]; ++k) {
                for (let l = 0; l < shape[3]; ++l) {
                    res[didx++] = src[sidx];
                    sidx += strides[3];
                }
                sidx += strides[2] - shape[3] * strides[3];
            }
            sidx += strides[1] - shape[2] * strides[2];
        }
        sidx += strides[0] - shape[1] * strides[1];
    }
}