# WEBGPU-Cuda
A quick and dirt attempt into making WEBGPU more CUDA like (not working yet).

### Folder: src
The directory src has a hacky way to make running webgpu compute more friendly:
```typescript
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

const d_A = WEBGPUContext.createBuffer(h_A, MemcpyType.HostToDevice);
const d_B = WEBGPUContext.createBuffer(h_B, MemcpyType.HostToDevice);
const d_C = WEBGPUContext.createBuffer(h_C, MemcpyType.DeviceToHost);

const d_N = WEBGPUContext.createBuffer(N, MemcpyType.HostToDevice);

const N_THREADS = 16;
const N_BLOCKS = Math.floor((SIZE + N_THREADS - 1) / N_THREADS);

const threads = [N_THREADS, N_THREADS];
const blocks = [N_BLOCKS, N_BLOCKS];

console.time("run");
WEBGPUContext.run(blocks, threads, code, "matrixMultiplicationKernel", d_A, d_B, d_C, d_N);
console.timeEnd("run");

const out_read = WEBGPUContext.createBuffer(new Float32Array(SIZE), MemcpyType.Readback);
const r = await WEBGPUContext.readBuffer(out_read, d_C);
console.log("r", r);
// [2240, 2296, 2352, 2408, 2464, 2520, 2576, 2632, 5824, 6008, 6192, 6376, 6560, 6744, 6928, 7112, 9408, 9720, 10032, 10344, 10656, 10968, 11280, 11592, 12992, 13432, 13872, 14312, 14752, 15192, 15632, 16072, 16576, 17144, 17712, 18280, 18848, 19416, 19984, 20552, 20160, 20856, 21552, 22248, 22944, 23640, 24336, 25032, 23744, 24568, 25392, 26216, 27040, 27864, 28688, 29512, 27328, 28280, 29232, 30184, 31136, 32088, 33040, 33992]
```

The WGSL code above gets translated into:
```glsl
@group(0) @binding(3) var<storage, read_write> N : u32;
@group(0) @binding(2) var<storage, read_write> C : array<f32, 64>;
@group(0) @binding(1) var<storage, read_write> B : array<f32, 64>;
@group(0) @binding(0) var<storage, read_write> A : array<f32, 64>;

const blockDim: vec3<u32> = vec3(16,16,1);
@compute @workgroup_size(16,16,1)
fn matrixMultiplicationKernel(
    @builtin(global_invocation_id) globalIdx: vec3<u32>,
    @builtin(local_invocation_id) threadIdx: vec3<u32>,
    @builtin(workgroup_id) blockIdx: vec3<u32>) 
{
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
        
```

This works by manipulating the source code and injecting the proper bindings etc.


### Folder: src-ast
src-ast attempts to convert CUDA code into WGSL.

For now it only has a lexer->code_generator.<br/>
Ideally it should be lexer->parser->ast->transformer->ast->code_generator
```C++
void kernel_1t1c(float *A, float *B, float *C, uint WIDTH) {
    // To DO: Each thread = 1 output row
    int colID = threadIdx.x + blockIdx.x * blockDim.x;
    if(colID < WIDTH) {
        for(int i = 0; i<WIDTH; i++){
            C[colID + i*WIDTH] = A[colID + i*WIDTH] + B[colID + i*WIDTH];
        }
    }
}
```
Converts to (formatted manually):
```rust
fn kernel_1t1c(A: f32, B: f32, C: f32, WIDTH: u32) {
    // To DO: Each thread = 1 output row
    var colID: i32 = threadIdx.x + blockIdx.x * blockDim.x;
    if(colID < WIDTH) {
        for(var i = 0; i < WIDTH; i++) {
            C[colID + i*WIDTH] = A[colID + i*WIDTH] + B[colID + i*WIDTH];
        }
    }
}
```
See ./dist/index.html for more examples

The idea with this repo is possibly to come up with some combination of the two folders in order to provide some interoperability and/or make WGSL more frictionless to use.