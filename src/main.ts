import { AddOp, DivOp, MulOp, Op, SubOp } from "./ops";

type Dtype = "float32";
const float32: Dtype = "float32";
type Shape = number[];

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

type TensorArgs = {
  shape: Shape;
  dtype: Dtype;
  op?: Op;
  inputs?: Tensor[];
  data?: number[];
}

class Tensor {
  private _data?: Float32Array;
  private _dtype: Dtype;
  private _shape: Shape;
  private _op?: Op;
  private _inputs?: Tensor[];
  private _evaluated: boolean = false;

  public constructor({ shape, dtype, op, inputs, data }: TensorArgs) {
    this._dtype = dtype;
    this._shape = shape;
    if (op) {
      this._op = op;
      this._inputs = inputs;
    } else {
      switch (dtype) {
        case "float32":
          this._data = new Float32Array(data);
          break;
        default:
          throw new Error(`Unsupported dtype: ${dtype}`);
      }
    }
  }

  get shape() {
    return this._shape;
  }

  add(t: Tensor): Tensor {
    if (!isSameShape(this.shape, t.shape)) {
      throw new Error(`Shape mismatch: ${t.shape} !== ${this.shape}`);
    }
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: AddOp, inputs: [this, t] });
  }

  sub(t: Tensor): Tensor {
    if (!isSameShape(this.shape, t.shape)) {
      throw new Error(`Shape mismatch: ${t.shape} !== ${this.shape}`);
    }
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: SubOp, inputs: [this, t] });
  }

  mul(t: Tensor): Tensor {
    if (!isSameShape(this.shape, t.shape)) {
      throw new Error(`Shape mismatch: ${t.shape} !== ${this.shape}`);
    }
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: MulOp, inputs: [this, t] });
  }

  div(t: Tensor): Tensor {
    if (!isSameShape(this.shape, t.shape)) {
      throw new Error(`Shape mismatch: ${t.shape} !== ${this.shape}`);
    }
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: DivOp, inputs: [this, t] });
  }

  async eval() {
    // goes through the graph and computes
    this._data = await this._op!.eval(this._inputs!.map(t => t._data!));
    this._evaluated = true;
  }

  async list(): Promise<Array<number>> {
    if (!this._evaluated) {
      await this.eval();
    }
    return Array.from(this._data);
  }
}

console.time("creating");
const a = new Tensor({ shape: [2, 2], dtype: float32, data: [1, 2, 3, 4] });
const b = new Tensor({ shape: [2, 2], dtype: float32, data: [1, 2, 3, 4] });
const c = a.add(b);
console.timeEnd("creating");
console.time("evaluating");
console.log(await c.list());
console.timeEnd("evaluating");
console.time("evaluating 2");
console.log(await c.list());
console.timeEnd("evaluating 2");