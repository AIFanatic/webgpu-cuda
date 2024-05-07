import * as types from "@webgpu/types";
import { MemcpyType, WEBGPUContext } from "./WEBGPUContext";

type TEST = Float32Array;

export class App {
    constructor() {
        const canvas = document.createElement("canvas");

        WEBGPUContext.setup(canvas).then(() => {
            this.postInit();
        })
    }

    private async postInit() {
        const code = `

            fn matrixMultiplicationKernel(A: f32, B: f32, C: f32, N: u32) {
                let x_index: u32 = blockDim.x * blockIdx.x + threadIdx.x;
                let y_index: u32 = blockDim.y * blockIdx.y + threadIdx.y;
              
                var sum: f32 = 0;
              
                if (x_index < N && y_index < N) {
                    for (var i: u32 = 0u; i < N; i++) {
                        sum += A[y_index * N + i] * B[i * N + x_index];
                    }
                    C[y_index * N + x_index] = sum;
                }
            }
        `

        const N = 8;
        const SIZE = N * N;
        
        const h_A = new Float32Array(SIZE);
        const h_B = new Float32Array(SIZE);
        const h_C = new Float32Array(SIZE);

        for (let i = 0; i < SIZE; i++) {
            h_A[i] = i;
            h_B[i] = i * 2;
        }

        console.log(h_A)
        console.log(h_B)
    

        const d_A = WEBGPUContext.createBuffer(h_A, MemcpyType.HostToDevice);
        const d_B = WEBGPUContext.createBuffer(h_B, MemcpyType.HostToDevice);
        const d_C = WEBGPUContext.createBuffer(h_C, MemcpyType.DeviceToHost);
        
        const d_N = WEBGPUContext.createBuffer(N, MemcpyType.HostToDevice);

        const N_THREADS = 16;
        const N_BLOCKS = Math.floor((SIZE + N_THREADS - 1) / N_THREADS);
      
        const threads = [N_THREADS, N_THREADS];
        const blocks = [N_BLOCKS, N_BLOCKS];

        console.log(blocks, threads)


        console.time("run");
        WEBGPUContext.run(blocks, threads, code, "matrixMultiplicationKernel", d_A, d_B, d_C, d_N);
        console.timeEnd("run");

        const out_read = WEBGPUContext.createBuffer(new Float32Array(SIZE), MemcpyType.Readback);
        const r = await WEBGPUContext.readBuffer(out_read, d_C);
        console.log("r", r);


        // const out_read_print = WEBGPUContext.createBuffer(new Float32Array(1024), MemcpyType.Readback);
        // const r_print = await WEBGPUContext.readBuffer(out_read_print, d_PrintResult);
        // console.log("r_print", new Uint32Array(r_print.buffer));

    }
}