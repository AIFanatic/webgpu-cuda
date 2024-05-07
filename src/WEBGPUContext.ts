import * as wgpu from "@webgpu/types";

export enum MemcpyType {
    HostToDevice,
    DeviceToHost,
    Readback
}

export class WEBGPUContext {
    private static canvas: HTMLCanvasElement;

    private static device: GPUDevice;
    private static context: GPUCanvasContext;

    private static isInitialized: boolean;

    public static async setup(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        return navigator.gpu.requestAdapter().then(async adapter => {
            if (!adapter) throw Error("Could not get adapter");

            const device = await adapter.requestDevice();

            const context = canvas.getContext("webgpu");
            if (!context) throw Error("Could not get context");

            context.configure({
                device,
                format: 'bgra8unorm'
            });

            this.device = device;
            this.context = context;

            const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
            this.context.configure({
                device: this.device,
                format: presentationFormat,
                alphaMode: "opaque"
            });

            this.isInitialized = true;
            return true;
        });
    }


    public static createBuffer(data: Float32Array | number, type: MemcpyType, isAtomic: boolean = false): GPUBuffer {
        if (!this.isInitialized) throw Error("WEBGPUContext not initialized!");

        let flags = 0;
        if (type === MemcpyType.HostToDevice) flags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        else if (type === MemcpyType.DeviceToHost) flags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
        else if (type === MemcpyType.Readback) flags = GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ

        let size = 0;
        if (data instanceof Float32Array) size = data.byteLength;
        else size = 1 * 4;

        const gpuBuffer = this.device.createBuffer({
            size: size,
            usage: flags
        });

        if (type === MemcpyType.HostToDevice) {
            if (data instanceof Float32Array) this.device.queue.writeBuffer(gpuBuffer, 0, data);
            else {
                // int check
                if (data % 1 === 0) this.device.queue.writeBuffer(gpuBuffer, 0, new Int32Array([data]));
                else this.device.queue.writeBuffer(gpuBuffer, 0, new Float32Array([data]));
            }
        }

        if (!(data instanceof Float32Array)) gpuBuffer.label = "number";
        if (isAtomic) gpuBuffer.label = "atomic";
        return gpuBuffer;
    }

    private static printfFormat(str: string) {
        str = str.replace("printf(", "").replace(")", "").replace(";", "");
        console.log(str);
        const args = str.split('",');
        if (args.length < 1) throw Error("Invalid printf");

        console.log("args", args);
        const text: string = args[0].replaceAll('"', "").replaceAll("'", "");
        const params = args.length >= 2 ? args[1].replaceAll(" ", "").split(",") : null;
        console.log("text", text);
        console.log("params", params);
        console.log("c", text.split("%").length);

        if (params !== null && text.split("%").length - 1 !== params.length) throw Error("Number of parameters doesn't match");

        let out = "";
        let pc = 0;
        let i = 0;
        const tA = text.split("");
        while (tA.length > 0) {
            const c = tA.shift();

            if (c === "%") {
                const type = tA.shift();
                console.log("type", type);
                const param = params[pc];
                console.log("param", param)
                // if (type === "i") out += `printfResult[${i}] = i32(${param});\n`;
                // else if (type === "u") out += `printfResult[${i}] = u32(${param});\n`;
                // else if (type === "f") out += `printfResult[${i}] = f32(${param});\n`;
                out += `printfResult[${i}] = -f32(${param});\n`;
                pc++;
            }
            else {
                out += `printfResult[${i}] = f32(${c.charCodeAt(0)});\n`;
            }
            i++;
            console.log(c);
        }
        return out;
    }

    public static run(gridDim: number[], blockDim: number[], code: string, entrypoint: string, ...params: GPUBuffer[]) {
        gridDim = Object.assign([1,1,1], gridDim);
        blockDim = Object.assign([1,1,1], blockDim);
        console.log("gridDim", gridDim);
        console.log("blockDim", blockDim);
        // console.log("code", code);
        // console.log("entrypoint", entrypoint);
        // console.log("params", params);

        // // Hack starts here

        // Get entrypoint
        const entrypointCharIndex = code.indexOf(`fn ${entrypoint}`);
        if (entrypointCharIndex === -1) throw Error("Entrypoint not found");

        // Make entrypoint main
        const a = code.slice(entrypointCharIndex);
        const entry = a.slice(0, a.indexOf(")") + 1);
        const realEntry = `fn ${entrypoint}(@builtin(global_invocation_id) globalIdx: vec3<u32>, @builtin(local_invocation_id) threadIdx: vec3<u32>, @builtin(workgroup_id) blockIdx: vec3<u32>)`;
        code = code.replace(entry, realEntry);

        // Get argument names and types
        const argsStrArray = entry.replace(`fn ${entrypoint}(`, "").replace(")", "").replaceAll(" ", "").split(",");
        let args: {name: string, type: "u32" | "i32" | "f32"}[] = [];

        for (let arg of argsStrArray) {
            const argInfo = arg.split(":");
            if (argInfo.length !== 2) throw Error(`Error parsing argument ${arg}`);
            if (argInfo[1] !== "u32" && argInfo[1] !== "i32" && argInfo[1] !== "f32") throw Error(`Argument ${argInfo} doesn't have a valid type`);
            args.push({name: argInfo[0], type: argInfo[1]});
        }

        // Add workgroup and compute decorator
        const workgroupCode = `@compute @workgroup_size(${blockDim})`;
        code = code.slice(0, entrypointCharIndex) + `${workgroupCode}\n` + code.slice(entrypointCharIndex);

        // Add workgroup and compute decorator
        const blockDimCode = `const blockDim: vec3<u32> = vec3(${blockDim});`;
        code = code.slice(0, entrypointCharIndex) + `${blockDimCode}\n` + code.slice(entrypointCharIndex);


        if (args.length !== params.length) throw Error(`Got ${args.length} arguments but got ${params.length} parameters`);

        // Create bindings
        for (let i = 0; i < params.length; i++) {
            const param = params[i];
            const arg = args[i];

            if (param.label === "number") {
                code = `@group(0) @binding(${i}) var<storage, read_write> ${arg.name} : ${arg.type};\n` + code;
            }
            else if (param.label === "atomic") {
                code = `@group(0) @binding(${i}) var<storage, read_write> ${arg.name} : atomic<${arg.type}>;\n` + code;
            }
            else {
                code = `@group(0) @binding(${i}) var<storage, read_write> ${arg.name} : array<${arg.type}, ${param.size / 4}>;\n` + code;
            }
        }

        // Handle printf, cuz why not, just one allowed for now
        let hasPrintf = code.indexOf("printf(") !== -1;
        if (hasPrintf === true) {
            let paramCount = params.length;
            code = `@group(0) @binding(${paramCount}) var<storage, read_write> printfResult : array<f32, 1024>;\n` + code;
            
            const printfIndex = code.indexOf("printf(");

            const printfFunctionLength = code.slice(printfIndex, code.length).indexOf(")") + 1;
            const printfFunction = code.slice(printfIndex, printfIndex + printfFunctionLength);
            const printfWGSL = this.printfFormat(printfFunction);
            code = code.replace(printfFunction, printfWGSL);
        }


        console.log("code", code);
        // return;

        // Actually run kernel
        const computeModule = this.device.createShaderModule({code: code});
        const computePipeline = this.device.createComputePipeline({
            label: 'compute shader',
            layout: 'auto',
            compute: {
                module: computeModule,
                entryPoint: entrypoint,
            },
        });

        let entries: GPUBindGroupEntry[] = [];
        for (let i = 0; i < params.length; i++) {
            entries.push({ binding: i, resource: { buffer: params[i] } });
        }
        // printf
        let printfBuffer: GPUBuffer | null = null;
        if (hasPrintf) {
            printfBuffer = this.createBuffer(new Float32Array(1024), MemcpyType.DeviceToHost);
            entries.push({ binding: entries.length, resource: { buffer: printfBuffer } });
        }

        const computeUniforms = this.device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: entries
        });

        const commandEncoder = this.device.createCommandEncoder();

        console.time("compute");
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(computePipeline);
        computePass.setBindGroup(0, computeUniforms);
        computePass.dispatchWorkgroups(gridDim[0], gridDim[1], gridDim[2]);
        computePass.end();
        console.timeEnd("compute");

        this.device.queue.submit([commandEncoder.finish()]);


        // printf
        if (hasPrintf && printfBuffer !== null) {
            const printfReadBuffer = this.createBuffer(new Float32Array(1024), MemcpyType.Readback);
            this.readBuffer(printfReadBuffer, printfBuffer).then(v => {
                let str = "";
                for (let c of v) {
                    if (c < 0) str += -c;
                    else str += String.fromCharCode(c);
                }
                console.warn("[GPU]:", str);
            })
        }
    }

    public static async readBuffer(to: GPUBuffer, from: GPUBuffer) {
        const commandEncoder = this.device.createCommandEncoder();

        commandEncoder.copyBufferToBuffer(
            from /* source buffer */, 0 /* source offset */,
            to /* destination buffer */, 0 /* destination offset */,
            to.size /* size */
        );

        this.device.queue.submit([commandEncoder.finish()]);

        await to.mapAsync(GPUMapMode.READ);
        const copyArrayBuffer = to.getMappedRange();

        return new Float32Array(copyArrayBuffer);
    }
}