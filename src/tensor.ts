import { float32 } from "./dtype";
import type { Shape, Dtype, Op, TypedArray, OpId } from "./web-ml";
import { isSameShape, broadcastShape } from "./util";

type TensorArgs = {
  shape: Shape;
  dtype?: Dtype;
  strides?: number[];
  op?: OpId;
  inputs?: Tensor[];
  data?: number | number[];
}

const op_cache: Map<OpId, Op> = new Map();

async function getOp(op: OpId): Promise<Op> {
  if (op_cache.size != 0) {
    if (!op_cache.has(op)) {
      throw new Error(`Op ${op} not found`);
    }
    return op_cache.get(op)!;
  }
  const wgpu_ops = await import("./ops").then(m => m.default);
  for (let [key, value] of Object.entries(wgpu_ops)) {
    op_cache.set(key as OpId, value);
  }
  if (!op_cache.has(op)) {
    throw new Error(`Op ${op} not found`);
  }
  return op_cache.get(op)!;
}

export class Tensor {
  private _data?: TypedArray;
  private _dtype: Dtype;
  private _shape: Shape;
  private _strides: number[];
  private _numElements: number;
  private _op?: OpId;
  private _inputs?: Tensor[];
  private _evaluated: boolean = false;

  constructor({ shape, dtype = float32, op, inputs, data, strides }: TensorArgs) {
    this._dtype = dtype ?? float32;
    this._shape = shape;
    this._numElements = shape.reduce((a, b) => a * b, 1);
    if (strides == null) {
      this._strides = Array.from({ length: shape.length }, () => 0);
      for (let i = shape.length - 1; i >= 0; --i) {
        this._strides[i] = (i == shape.length - 1) ? 1 : this._strides[i + 1] * shape[i + 1];
      }
    } else {
      this._strides = strides;
    }
    if (op) {
      this._op = op;
      this._inputs = inputs;
    } else {
      if (data == null) {
        throw new Error("Cannot create tensor without data or op");
      }
      if (typeof data === "number") {
        switch (dtype) {
          case "float32":
            this._data = new Float32Array(this._numElements).fill(data);
            break;
          case "int32":
            this._data = new Int32Array(this._numElements).fill(data);
            break
          case "uint32":
            this._data = new Uint32Array(this._numElements).fill(data);
            break;
          default:
            throw new Error(`Unsupported dtype: ${dtype}`);
        }
        return
      }
      // TODO: this breaks expanding
      if (data.length !== this._numElements) {
        throw new Error(`Data length ${data.length} does not match shape ${shape}`);
      }
      switch (dtype) {
        case "float32":
          this._data = new Float32Array(data);
          break;
        case "int32":
          this._data = new Int32Array(data);
          break
        case "uint32":
          this._data = new Uint32Array(data);
          break;
        default:
          throw new Error(`Unsupported dtype: ${dtype}`);
      }
    }
  }

  get ndim() {
    return this._shape.length;
  }

  get shape() {
    return this._shape;
  }

  get dtype() {
    return this._dtype;
  }

  get numElements() {
    return this._numElements;
  }

  get strides() {
    return this._strides;
  }

  get data() {
    return this._data;
  }

  abs() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: "abs", inputs: [this] });
  }

  round() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: "round", inputs: [this] });
  }

  floor() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: "floor", inputs: [this] });
  }

  ceil() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: "ceil", inputs: [this] });
  }

  sin() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: "sin", inputs: [this] });
  }

  cos() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: "cos", inputs: [this] });
  }

  tan() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: "tan", inputs: [this] })
  }

  sinh() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: "sinh", inputs: [this] });
  }

  cosh() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: "cosh", inputs: [this] });
  }

  tanh() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: "tanh", inputs: [this] });
  }

  asin() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: "asin", inputs: [this] });
  }

  acos() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: "acos", inputs: [this] });
  }

  atan() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: "atan", inputs: [this] });
  }

  sqrt() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: "sqrt", inputs: [this] });
  }

  log() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: "log", inputs: [this] });
  }

  log2() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: "log2", inputs: [this] });
  }

  exp() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: "exp", inputs: [this] });
  }

  exp2() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: "exp2", inputs: [this] });
  }

  neg() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: "neg", inputs: [this] });
  }

  // Binary ops
  pow(t: Tensor | number) {
    return this.binaryOp(t, "pow");
  }

  max(t: Tensor | number) {
    return this.binaryOp(t, "max");
  }

  min(t: Tensor | number) {
    return this.binaryOp(t, "min");
  }

  add(t: Tensor | number): Tensor {
    return this.binaryOp(t, "add");
  }

  sub(t: Tensor | number): Tensor {
    return this.binaryOp(t, "sub");
  }

  mul(t: Tensor | number): Tensor {
    return this.binaryOp(t, "mul");
  }

  div(t: Tensor | number): Tensor {
    return this.binaryOp(t, "div");
  }

  equal(t: Tensor | number): Tensor {
    return this.binaryOp(t, "equal");
  }

  private binaryOp(t: Tensor | number, op: OpId) {
    t = typeof t === "number" ? new Tensor({ shape: this.shape, dtype: this._dtype, data: t }) : t;
    let new_shape = broadcastShape(this.shape, t.shape);
    let [a, b] = [this.reshape(new_shape), t.reshape(new_shape)];
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: op, inputs: [a, b] });
  }

  not_equal(t: Tensor | number): Tensor {
    return this.equal(t).mul(-1).add(1);
  }

  greater(t: Tensor | number): Tensor {
    return this.binaryOp(t, "greater");
  }

  greater_equal(t: Tensor | number): Tensor {
    return this.greater(t).add(this.equal(t));
  }

  less(t: Tensor | number): Tensor {
    t = typeof t === "number" ? new Tensor({ shape: this.shape, dtype: this._dtype, data: t }) : t;
    return t.greater(this);
  }

  less_equal(t: Tensor | number): Tensor {
    return this.less(t).add(this.equal(t));
  }

  reshape(new_shape: Shape) {
    if (isSameShape(this.shape, new_shape)) {
      return this;
    }
    if (!isSameShape(broadcastShape(this.shape, new_shape), new_shape)) {
      throw new Error("Cannot broadcast to new shape");
    }
    let strides = Array.from({ length: new_shape.length }, () => 0);
    let diff = new_shape.length - this.shape.length;
    for (let i = this.shape.length - 1; i >= 0; --i) {
      strides[i + diff] = (this.shape[i] == 1) ? 0 : this._strides[i];
    }
    return new Tensor({ shape: new_shape, dtype: this._dtype, strides: strides, op: "reshape", inputs: [this] });
  }

  async eval() {
    if (this._evaluated) {
      return;
    }
    if (!this._op || !this._inputs) {
      throw new Error("Cannot eval without op and inputs");
    }
    for (let input of this._inputs) {
      if (input._op) {
        await input.eval();
      }
    }
    const _op = await getOp(this._op);
    this._data = await _op.eval([...this._inputs, this]);
    this._evaluated = true;
  }

  async list(): Promise<number[] | number[][] | number[][][] | number[][][][]> {
    if (!this._data) {
      if (!this._evaluated) {
        await this.eval();
      }
      if (!this._data) {
        throw new Error("Cannot list without data");
      }
    }
    // TODO: this is a bit ugly and could be optimized
    switch (this.ndim) {
      case 1:
        return Array.from(this._data) as number[];
      case 2: {
        let res = Array.from({ length: this.shape[0] });
        let currentIndex = 0;
        for (let i = 0; i < this.shape[0]; i++) {
          res[i] = Array.from(this._data.slice(currentIndex, currentIndex + this.shape[1]));
          currentIndex += this.shape[1];
        }
        return res as number[][];
      }
      case 3: {
        const res = Array.from({ length: this.shape[0] }, () => Array.from({ length: this.shape[1] }));
        let currentIndex = 0;
        for (let i = 0; i < this.shape[0]; i++) {
          for (let j = 0; j < this.shape[1]; j++) {
            res[i][j] = Array.from(this._data.slice(currentIndex, currentIndex + this.shape[2]));
            currentIndex += this.shape[2];
          }
        }
        return res as number[][][];
      }
      case 4: {
        const res = Array.from({ length: this.shape[0] }, () => Array.from({ length: this.shape[1] }, () => Array.from({ length: this.shape[2] })));
        let currentIndex = 0;
        for (let i = 0; i < this.shape[0]; i++) {
          for (let j = 0; j < this.shape[1]; j++) {
            for (let k = 0; k < this.shape[2]; k++) {
              res[i][j][k] = Array.from(this._data.slice(currentIndex, currentIndex + this.shape[3]));
              currentIndex += this.shape[3];
            }
          }
        }
        return res as number[][][][];
      }
      default:
        throw new Error("Unsupported shape");
    }
  }
}