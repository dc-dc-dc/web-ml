import { uint32, float32, int32 } from "./dtype";
import { AbsOp, AddOp, CeilOp, CosOp, DivOp, FloorOp, MulOp, SinOp, SqrtOp, SubOp, TanOp, SinhOp, CoshOp, TanhOp, AcosOp, AsinOp, AtanOp, MaxOp, MinOp, PowOp, RoundOp, ExpOp, Exp2Op, Log2Op, LogOp, EqualOp, NotEqualOp, GreaterOp, NegOp } from "./ops";
import type { Shape, Dtype, Op, TypedArray } from "./web-ml";

function isSameShape(x: Shape, y: Shape): boolean {
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

export function broadcast_shape(x: Shape, y: Shape): Shape {
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

type TensorArgs = {
  shape: Shape;
  dtype?: Dtype;
  strides?: number[];
  op?: Op;
  inputs?: Tensor[];
  data?: number | number[];
}

export class Tensor {
  private _data?: TypedArray;
  private _dtype: Dtype;
  private _shape: Shape;
  private _strides: number[];
  private _numElements: number;
  private _op?: Op;
  private _inputs?: Tensor[];
  private _evaluated: boolean = false;

  constructor({ shape, dtype = float32, op, inputs, data, strides }: TensorArgs) {
    this._dtype = dtype;
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

  abs() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: AbsOp, inputs: [this] });
  }

  round() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: RoundOp, inputs: [this] });
  }

  floor() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: FloorOp, inputs: [this] });
  }

  ceil() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: CeilOp, inputs: [this] });
  }

  sin() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: SinOp, inputs: [this] });
  }

  cos() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: CosOp, inputs: [this] });
  }

  tan() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: TanOp, inputs: [this] })
  }

  sinh() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: SinhOp, inputs: [this] });
  }

  cosh() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: CoshOp, inputs: [this] });
  }

  tanh() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: TanhOp, inputs: [this] });
  }

  asin() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: AsinOp, inputs: [this] });
  }

  acos() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: AcosOp, inputs: [this] });
  }

  atan() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: AtanOp, inputs: [this] });
  }

  sqrt() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: SqrtOp, inputs: [this] });
  }

  log() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: LogOp, inputs: [this] });
  }

  log2() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: Log2Op, inputs: [this] });
  }

  exp() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: ExpOp, inputs: [this] });
  }

  exp2() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: Exp2Op, inputs: [this] });
  }

  neg() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: NegOp, inputs: [this] });
  }

  // Binary ops
  pow(t: Tensor | number) {
    t = typeof t === "number" ? new Tensor({ shape: this.shape, dtype: this._dtype, data: t }) : t;
    if (!isSameShape(this.shape, t.shape)) {
      throw new Error(`Shape mismatch: ${t.shape} !== ${this.shape}`);
    }
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: PowOp, inputs: [this, t] });
  }

  max(t: Tensor | number) {
    t = typeof t === "number" ? new Tensor({ shape: this.shape, dtype: this._dtype, data: t }) : t;
    if (!isSameShape(this.shape, t.shape)) {
      throw new Error(`Shape mismatch: ${t.shape} !== ${this.shape}`);
    }
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: MaxOp, inputs: [this, t] });
  }

  min(t: Tensor | number) {
    t = typeof t === "number" ? new Tensor({ shape: this.shape, dtype: this._dtype, data: t }) : t;
    if (!isSameShape(this.shape, t.shape)) {
      throw new Error(`Shape mismatch: ${t.shape} !== ${this.shape}`);
    }
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: MinOp, inputs: [this, t] });
  }

  add(t: Tensor | number): Tensor {
    t = typeof t === "number" ? new Tensor({ shape: this.shape, dtype: this._dtype, data: t }) : t;
    if (!isSameShape(this.shape, t.shape)) {
      throw new Error(`Shape mismatch: ${t.shape} !== ${this.shape}`);
    }
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: AddOp, inputs: [this, t] });
  }

  sub(t: Tensor | number): Tensor {
    t = typeof t === "number" ? new Tensor({ shape: this.shape, dtype: this._dtype, data: t }) : t;
    if (!isSameShape(this.shape, t.shape)) {
      throw new Error(`Shape mismatch: ${t.shape} !== ${this.shape}`);
    }
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: SubOp, inputs: [this, t] });
  }

  mul(t: Tensor | number): Tensor {
    t = typeof t === "number" ? new Tensor({ shape: this.shape, dtype: this._dtype, data: t }) : t;
    if (!isSameShape(this.shape, t.shape)) {
      throw new Error(`Shape mismatch: ${t.shape} !== ${this.shape}`);
    }
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: MulOp, inputs: [this, t] });
  }

  div(t: Tensor | number): Tensor {
    t = typeof t === "number" ? new Tensor({ shape: this.shape, dtype: this._dtype, data: t }) : t;
    if (!isSameShape(this.shape, t.shape)) {
      throw new Error(`Shape mismatch: ${t.shape} !== ${this.shape}`);
    }
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: DivOp, inputs: [this, t] });
  }

  equal(t: Tensor | number): Tensor {
    t = typeof t === "number" ? new Tensor({ shape: this.shape, dtype: this._dtype, data: t }) : t;
    if (!isSameShape(this.shape, t.shape)) {
      throw new Error(`Shape mismatch: ${t.shape} !== ${this.shape}`);
    }
    return new Tensor({
      shape: this.shape,
      dtype: float32,
      op: EqualOp,
      inputs: [this, t],
    });
  }

  not_equal(t: Tensor | number): Tensor {
    return this.equal(t).mul(-1).add(1);
  }

  greater(t: Tensor | number): Tensor {
    t = typeof t === "number" ? new Tensor({ shape: this.shape, dtype: this._dtype, data: t }) : t;
    if (!isSameShape(this.shape, t.shape)) {
      throw new Error(`Shape mismatch: ${t.shape} !== ${this.shape}`);
    }
    return new Tensor({
      shape: this.shape,
      dtype: float32,
      op: GreaterOp,
      inputs: [this, t],
    });
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
    if (broadcast_shape(this.shape, new_shape) !== new_shape) {
      throw new Error("Cannot broadcast to new shape");
    }

    let strides = Array.from({ length: new_shape.length }, () => 0);
    let diff = new_shape.length - this.shape.length;
    for (let i = this.shape.length - 1; i >= 0; --i) {
      strides[i + diff] = (this.shape[i] == 1) ? 0 : this._strides[i];
    }
    return new Tensor({ shape: new_shape, dtype: this._dtype, data: this._data, strides: strides });
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
    this._data = await this._op.eval(this._inputs.map(t => t._data!));
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

    switch (this.shape.length) {
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
        const res = Array.from({length: this.shape[0]}, () => Array.from({length: this.shape[1]}, () => Array.from({length: this.shape[2]})));
        let currentIndex = 0;
        for(let i = 0; i < this.shape[0]; i++) {
          for(let j = 0; j < this.shape[1]; j++) {
            for(let k = 0; k < this.shape[2]; k++) {
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