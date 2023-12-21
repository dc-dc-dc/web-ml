let device;
if(__DEV__) {
  const webnodegpu = await import("webnode-gpu");
  const gpu = webnodegpu.create([]);
  const adapter = await gpu.requestAdapter();
  device = await adapter.requestDevice();
} else {
  const adapter = await navigator.gpu.requestAdapter();
  device = await adapter.requestDevice();
}

type Dtype = "float32";
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

class Tensor {
  private _data: Float32Array;
  constructor(_data: number[], private _shape: Shape, private _dtype: Dtype) {
    switch (_dtype) {
      case "float32":
        this._data = new Float32Array(_data);
        break;
      default:
        throw new Error(`Unsupported dtype: ${_dtype}`);
    }
  }

  get shape() {
    return this._shape;
  }

  add(t: Tensor): Tensor {
    if (!isSameShape(this.shape, t.shape)) {
      throw new Error(`Shape mismatch: ${t.shape} !== ${this.shape}`);
    }
    return this;
  }

  list(): Array<number> {
    return Array.from(this._data);
  }
}

const a = new Tensor([1, 2, 3, 4], [2, 2], "float32");
const b = new Tensor([1, 2, 3, 4], [2, 2], "float32");

async function execute(name: string, code: string, args: Float32Array[]): Float32Array {
  if(args.length < 1) {
    throw new Error("Atleast one argument is required");
  }
  const module = device.createShaderModule({ label: name, code });
  const pipeline = device.createComputePipeline({
    label: `${name} pipeline`,
    layout: "auto",
    compute: {
      module,
      entryPoint: name
    }
  });
  const buffers = [];
  for (let i = 0; i < args.length; i++) {
    let input = args[i];
    buffers.push(device.createBuffer({
      label: `${name} buffer ${i}`,
      size: 4 * input.length,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }));
    device.queue.writeBuffer(buffers[i], 0, input);
  }

  const bindGroup = device.createBindGroup({
    label: `${name} bind group`,
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buff, index) => ({ binding: index, resource: { buffer: buff } })),
  });
  const resultBuffer = device.createBuffer({
    label: `${name} result buffer`,
    size: 4 * args[0].length,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(args[0].length);
  pass.end();

  encoder.copyBufferToBuffer(buffers[0], 0, resultBuffer, 0, args[0].length * 4);
  const commandBuffer = encoder.finish();
  device.queue.submit([commandBuffer]);

  // Weird that this is even necessary
  await resultBuffer.mapAsync(GPUMapMode.READ);
  const d = new Float32Array(args[0].length);
  d.set(new Float32Array(resultBuffer.getMappedRange()));
  resultBuffer.unmap();
  return d;
}

async function unary_op(name: string, op: string, a: Float32Array): Float32Array {
  return execute(name, `
  @group(0) @binding(0) var<storage, read_write> a: array<f32>;
  @compute @workgroup_size(1)
  fn ${name}(
    @builtin(global_invocation_id) gid: vec3<u32>
  ) {
    a[gid.x] = ${op}(a[gid.x]);
  }
  `, [a]);
}

async function binary_op(name: string, char: string, a: Float32Array, b: Float32Array): Float32Array {
  return execute(name, `
      @group(0) @binding(0) var<storage, read_write> a: array<f32>;
      @group(0) @binding(1) var<storage, read_write> b: array<f32>;
  
      @compute @workgroup_size(1) 
      fn ${name}(
        @builtin(global_invocation_id) gid: vec3<u32>
      ) {
        a[gid.x] = a[gid.x] ${char} b[gid.x];
      }
    `, [a, b]);
}

async function round(a: Float32Array): Float32Array {
  return unary_op("_round", "round", a);
}

async function abs(a: Float32Array): Float32Array {
  return unary_op("_abs", "abs", a);
}

async function add(a: Float32Array, b: Float32Array): Float32Array {
  return binary_op("add", "+", a, b);
}

async function sub(a: Float32Array, b: Float32Array): Float32Array {
  return binary_op("sub", "-", a, b);
}

async function mul(a: Float32Array, b: Float32Array): Float32Array {
  return binary_op("mul", "*", a, b);
}

async function div(a: Float32Array, b: Float32Array): Float32Array {
  return binary_op("div", "/", a, b);
}

console.log(await round(new Float32Array([1.1, 2.2, 3.3, 4.4])));
console.log(await abs(new Float32Array([-1, -2, -3, -4])));
console.log(await add(new Float32Array([1, 2, 3, 4]), new Float32Array([1, 2, 3, 4])));
console.log(await sub(new Float32Array([1, 2, 3, 4]), new Float32Array([1, 2, 3, 4])));
console.log(await mul(new Float32Array([1, 2, 3, 4]), new Float32Array([1, 2, 3, 4])));
console.log(await div(new Float32Array([1, 2, 3, 4]), new Float32Array([1, 2, 3, 4])));