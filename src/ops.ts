class WebGpuBackend {
    private _device;
    private static _instance;
    constructor() { }

    public async init() {
        if (__DEV__) {
            const webnodegpu = await import("webnode-gpu");
            const gpu = webnodegpu.create([]);
            const adapter = await gpu.requestAdapter();
            this._device = await adapter.requestDevice();
        } else {
            const adapter = await navigator.gpu.requestAdapter();
            this._device = await adapter.requestDevice();
        }
    }

    public static async instance(): Promise<WebGpuBackend> {
        if (!WebGpuBackend._instance) {
            WebGpuBackend._instance = new WebGpuBackend();
            await WebGpuBackend._instance.init();
        }
        return WebGpuBackend._instance;        
    }

    public async execute(name: string, code: string, args: Float32Array[]): Promise<Float32Array> {
        if (!this._device) {
            await this.init();
        }
        if (args.length < 1) {
            throw new Error("Atleast one argument is required");
        }
        const module = this._device.createShaderModule({ label: name, code });
        const pipeline = this._device.createComputePipeline({
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
            buffers.push(this._device.createBuffer({
                label: `${name} buffer ${i}`,
                size: 4 * input.length,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            }));
            this._device.queue.writeBuffer(buffers[i], 0, input);
        }

        const bindGroup = this._device.createBindGroup({
            label: `${name} bind group`,
            layout: pipeline.getBindGroupLayout(0),
            entries: buffers.map((buff, index) => ({ binding: index, resource: { buffer: buff } })),
        });
        const resultBuffer = this._device.createBuffer({
            label: `${name} result buffer`,
            size: 4 * args[0].length,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        const encoder = this._device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(args[0].length);
        pass.end();

        encoder.copyBufferToBuffer(buffers[0], 0, resultBuffer, 0, args[0].length * 4);
        const commandBuffer = encoder.finish();
        this._device.queue.submit([commandBuffer]);

        // Weird that this is even necessary
        await resultBuffer.mapAsync(GPUMapMode.READ);
        const d = new Float32Array(args[0].length);
        d.set(new Float32Array(resultBuffer.getMappedRange()));
        resultBuffer.unmap();
        return d;
    }
}

export interface Op {
    eval(inputs: Float32Array[]): Promise<Float32Array>;
}

async function unary_op(name: string, op: string, a: Float32Array) {
    return (await WebGpuBackend.instance()).execute(name, `
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@compute @workgroup_size(1)
fn ${name}(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    a[gid.x] = ${op}(a[gid.x]);
}
`, [a]);
}

async function binary_op(name: string, char: string, a: Float32Array, b: Float32Array) {
    return (await WebGpuBackend.instance()).execute(name, `
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

class UnaryOp implements Op {
    constructor(private _name: string, private _op: string) { }

    async eval(inputs: Float32Array[]): Promise<Float32Array> {
        return unary_op(this._name, this._op, inputs[0]);
    }
}

class BinaryOp implements Op {
    constructor(private _name: string, private _op: string) { }

    async eval(inputs: Float32Array[]): Promise<Float32Array> {
        return binary_op(this._name, this._op, inputs[0], inputs[1]);
    }

}

export const RoundOp = new UnaryOp("_round", "round");
export const AbsOp = new UnaryOp("_abs", "abs");
export const CeilOp = new UnaryOp("_ceil", "ceil");
export const FloorOp = new UnaryOp("_floor", "floor");

export const SqrtOp = new UnaryOp("_sqrt", "sqrt");
export const SinOp = new UnaryOp("_sin", "sin");
export const CosOp = new UnaryOp("_cos", "cos");
export const TanOp = new UnaryOp("_tan", "tan");
export const AsinOp = new UnaryOp("_asin", "asin");
export const AcosOp = new UnaryOp("_acos", "acos");
export const AtanOp = new UnaryOp("_atan", "atan");
export const SinhOp = new UnaryOp("_sinh", "sinh");
export const CoshOp = new UnaryOp("_cosh", "cosh");
export const TanhOp = new UnaryOp("_tanh", "tanh");

export const AddOp = new BinaryOp("add", "+");
export const SubOp = new BinaryOp("sub", "-");
export const MulOp = new BinaryOp("mul", "*");
export const DivOp = new BinaryOp("div", "/");