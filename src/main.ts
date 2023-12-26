import { AbsOp, AddOp, CeilOp, CosOp, DivOp, FloorOp, MulOp, Op, SinOp, SqrtOp, SubOp, TanOp, SinhOp, CoshOp, TanhOp, AcosOp, AsinOp, AtanOp } from "./ops";

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
    if(shape.length < 0 || shape.length > 2) {
      throw new Error("Only 1D and 2D tensors are supported");
    }
    this._dtype = dtype;
    this._shape = shape;
    if (op) {
      this._op = op;
      this._inputs = inputs;
    } else {
      if (!data) {
        throw new Error("Cannot create tensor without data or op");
      }
      if(data.length !== shape.reduce((a, b) => a * b, 1)) {
        throw new Error(`Data length ${data.length} does not match shape ${shape}`);
      }
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

  abs() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: AbsOp, inputs: [this] });
  }

  round() {
    return new Tensor({ shape: this.shape, dtype: this._dtype, op: AbsOp, inputs: [this] });
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
    if(this._evaluated) {
      return;
    }
    if(!this._op || !this._inputs) {
      throw new Error("Cannot eval without op and inputs");
    }
    for(let input of this._inputs) {
      if(input._op) {
        await input.eval();
      }
    }
    this._data = await this._op.eval(this._inputs.map(t => t._data!));
    this._evaluated = true;
  }

  async list(): Promise<number[] | number[][]> {
    if (!this._evaluated) {
      await this.eval();
    }
    if(!this._data) {
      throw new Error("Cannot list without data");
    }
    switch(this.shape.length) {
    case 1:
      return Array.from(this._data);
    case 2:
      const rows = this.shape.length > 1 ? this.shape[1] : this.shape[0];
      let res = Array.from({length: this.shape[0]}, () => Array(rows).fill(0));
      for(let i = 0; i < this._data.length; i++) {
        res[Math.floor(i / rows)][i % rows] = this._data[i];
      }
      return res;
    default:
      throw new Error("Unsupported shape");
    }
  }
}

console.time("creating");
const a = new Tensor({ shape: [2, 2], dtype: float32, data: [1, 2, 3, 4] });
const b = new Tensor({ shape: [2, 2], dtype: float32, data: [1, 2, 3, 4] });
const c = a.mul(b).add(b.add(a));
const d = c.mul(c);
console.log(await d.list());