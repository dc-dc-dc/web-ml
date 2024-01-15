import type { Dtype, Op, OpId, TypedArray } from "./web-ml";
import { getDtypeSize, float32, int32, uint32 } from "./dtype";
import { Tensor } from "./tensor";
import { copy1D, copy2D, copy3D, copy4D, copySingle } from "./copy";

class WebGpuBackend {
    // this is set in init since constructors cant be async
    private _device!: GPUDevice;
    private _adapter!: GPUAdapter;
    private static _instance?: WebGpuBackend;
    constructor() { }

    public async init() {
        if (__DEV__) {
            const webnodegpu = await import("webnode-gpu");
            const gpu = webnodegpu.create([]);
            const adapter = await gpu.requestAdapter();
            if (!adapter) {
                throw new Error("WebGPU not supported");
            }
            this._adapter = adapter;
            this._device = await this._adapter.requestDevice();
        } else {
            if (!navigator.gpu) {
                throw new Error("WebGPU not supported");
            }
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                throw new Error("WebGPU not supported");
            }
            this._adapter = adapter;
            this._device = await this._adapter.requestDevice();
        }
    }

    public static async instance(): Promise<WebGpuBackend> {
        if (!WebGpuBackend._instance) {
            WebGpuBackend._instance = new WebGpuBackend();
            await WebGpuBackend._instance.init();
        }
        return WebGpuBackend._instance;
    }


    public async execute(name: string, code: string, args: TypedArray[], out: Dtype): Promise<TypedArray> {
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
                size: getDtypeSize(getDtypeFromTypedArray(input)) * input.length,
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
            size: getDtypeSize(out) * args[0].length,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        const encoder = this._device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(args[0].length);
        pass.end();

        encoder.copyBufferToBuffer(buffers[0], 0, resultBuffer, 0, args[0].length * getDtypeSize(out));
        const commandBuffer = encoder.finish();
        this._device.queue.submit([commandBuffer]);

        // Weird that this is even necessary
        await resultBuffer.mapAsync(GPUMapMode.READ);
        switch (out) {
            case float32:
                const d = new Float32Array(args[0].length);
                d.set(new Float32Array(resultBuffer.getMappedRange()));
                resultBuffer.unmap();
                return d;
            case int32:
                const i = new Int32Array(args[0].length);
                i.set(new Int32Array(resultBuffer.getMappedRange()));
                resultBuffer.unmap();
                return i;
            case uint32:
                const u = new Uint32Array(args[0].length);
                u.set(new Uint32Array(resultBuffer.getMappedRange()));
                resultBuffer.unmap();
                return u;
            default:
                throw new Error("Unsupported dtype");
        }
    }
}

function getDtypeFromTypedArray(t: TypedArray): Dtype {
    if (t instanceof Float32Array) {
        return float32;
    } else if (t instanceof Int32Array) {
        return int32;
    } else if (t instanceof Uint32Array) {
        return uint32;
    }
    throw new Error("Unsupported type");
}

function dtypeToWebgpuType(t: Dtype): string {
    if (t === float32) {
        return "f32";
    } else if (t === int32) {
        return "i32";
    } else if (t === uint32) {
        return "u32";
    }
    throw new Error(`Unsupported type=${t}`);
}

async function unary_op(name: string, op: string, a: Tensor) {
    return (await WebGpuBackend.instance()).execute(name, `
@group(0) @binding(0) var<storage, read_write> a: array<${dtypeToWebgpuType(a.dtype)}>;
@compute @workgroup_size(1)
fn ${name}(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    a[gid.x] = ${op}(a[gid.x]);
}
`, [a.data], float32);
}

async function binary_op(name: string, char: string, a: Tensor, b: Tensor, is_func = false) {
    const _func = is_func ? `${char}(a[gid.x], b[gid.x])` : `a[gid.x] ${char} b[gid.x]`;
    return (await WebGpuBackend.instance()).execute(name, `
      @group(0) @binding(0) var<storage, read_write> a: array<${dtypeToWebgpuType(a.dtype)}>;
      @group(0) @binding(1) var<storage, read_write> b: array<${dtypeToWebgpuType(b.dtype)}>;
  
      @compute @workgroup_size(1) 
      fn ${name}(
        @builtin(global_invocation_id) gid: vec3<u32>
      ) {
        a[gid.x] = ${_func};
      }
    `, [a.data, b.data], float32);
}

async function compare_op(name: string, char: string, a: Tensor, b: Tensor) {
    return (await WebGpuBackend.instance()).execute(name, `
        @group(0) @binding(0) var<storage, read_write> a: array<${dtypeToWebgpuType(a.dtype)}>;
        @group(0) @binding(1) var<storage, read_write> b: array<${dtypeToWebgpuType(b.dtype)}>;
    
        @compute @workgroup_size(1) 
        fn ${name}(
            @builtin(global_invocation_id) gid: vec3<u32>
        ) {
            a[gid.x] = select(0.0f, 1.0f, a[gid.x] ${char} b[gid.x]); 
        }
    `, [a.data, b.data], float32);
}

class UnaryOp implements Op {
    constructor(private _name: string, private _op: string) { }

    async eval(inputs: Tensor[]): Promise<TypedArray> {
        if (inputs.length != 1) throw new Error("UnaryOp requires one input");
        return unary_op(this._name, this._op, inputs[0]);
    }
}

class BinaryOp implements Op {
    constructor(private _name: string, private _op: string, private _isFunc = false) { }

    async eval(inputs: Tensor[]): Promise<Tensor> {
        if (inputs.length < 2) throw new Error("BinaryOp requires two inputs");
        return binary_op(this._name, this._op, inputs[0], inputs[1], this._isFunc);
    }
}

class CompareOp implements Op {
    constructor(private _name: string, private _op: string) { }
    async eval(inputs: Tensor[]): Promise<TypedArray> {
        if (inputs.length < 2) throw new Error("CompareOp requires two inputs");
        return compare_op(this._name, this._op, inputs[0], inputs[1]);
    }
}

class Reshape implements Op {
    async eval(inputs: Tensor[]): Promise<TypedArray> {
        if (inputs.length != 2) throw new Error("Reshape requires two inputs");
        const input = inputs[0];
        if (input.ndim > 4) throw new Error("Reshape only supports up to 4 dimensions");
        const output = inputs[1];
        const res = new Float32Array(output.numElements);
        if (input.numElements == 1) {
            copySingle(input.data[0], res);
        } else {
            switch (input.ndim) {
                case 1:
                    copy1D(input.data, input.shape, input.strides, res);
                    break;
                case 2:
                    copy2D(input.data, input.shape, input.strides, res);
                    break;
                case 3:
                    copy3D(input.data, input.shape, input.strides, res);
                    break;
                case 4:
                    copy4D(input.data, input.shape, input.strides, res);
                    break;
            }
        }
        return res;
    }
}

// Op Map
const res: Record<OpId, Op> = {
    // mem ops
    "reshape": new Reshape(),

    // unary ops
    "neg": new UnaryOp("_neg", "-"),
    "round": new UnaryOp("_round", "round"),
    "abs": new UnaryOp("_abs", "abs"),
    "ceil": new UnaryOp("_ceil", "ceil"),
    "floor": new UnaryOp("_floor", "floor"),
    "log": new UnaryOp("_log", "log"),
    "exp": new UnaryOp("_exp", "exp"),
    "log2": new UnaryOp("_log2", "log2"),
    "exp2": new UnaryOp("_exp2", "exp2"),
    "sqrt": new UnaryOp("_sqrt", "sqrt"),
    "sin": new UnaryOp("_sin", "sin"),
    "cos": new UnaryOp("_cos", "cos"),
    "tan": new UnaryOp("_tan", "tan"),
    "asin": new UnaryOp("_asin", "asin"),
    "acos": new UnaryOp("_acos", "acos"),
    "atan": new UnaryOp("_atan", "atan"),
    "sinh": new UnaryOp("_sinh", "sinh"),
    "cosh": new UnaryOp("_cosh", "cosh"),
    "tanh": new UnaryOp("_tanh", "tanh"),

    // binary ops
    "add": new BinaryOp("add", "+"),
    "sub": new BinaryOp("sub", "-"),
    "mul": new BinaryOp("mul", "*"),
    "div": new BinaryOp("div", "/"),
    "max": new BinaryOp("_max", "max", true),
    "min": new BinaryOp("_min", "min", true),
    "pow": new BinaryOp("_pow", "pow", true),

    // compare ops
    "equal": new CompareOp("_equal", "=="),
    "greater": new CompareOp("_greater", ">")
};

export default res;