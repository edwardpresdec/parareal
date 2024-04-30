#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TOL pow(10,-15)
#define length 5000
#define BLOCK_SIZE 1024

// Define the differential equation dy/dt = f(t, y)
__device__ __host__ double f(double t, double y) {
    return t * y; // Example: dy/dt = t * y
}

__device__ __host__ double sol(double t, double t0, double y0) {
    return exp(pow(t, 2) / 2) * exp(log(y0) - (pow(t0, 2) / 2));
}

__device__ __host__ double euler(double t0, double y0, double h) {
    return y0 + h * f(t0, y0);
}

__device__ double rungeKutta(double h, double t0, double y0) {
    double k1, k2, k3, k4;
    k1 = h * f(t0, y0);
    k2 = h * f(t0 + h / 2, y0 + k1 / 2);
    k3 = h * f(t0 + h / 2, y0 + k2 / 2);
    k4 = h * f(t0 + h, y0 + k3);

    return y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
}

__global__ void solveODE(double *result, double *U, double y0, double t0, double h) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("Block index: %d, Thread index: %d, tid: %d, U[tid]: %f\n", blockIdx.x, threadIdx.x, tid,U[tid]);

    if (tid < length) {
        //initialize time values specific to the thread
        double y, actual, h1, t2;

        int n = 1;
        double t1 = t0 + tid * h;
        double tfinal = t1 + h;
        actual = sol(tfinal, t1, U[tid]);

        double y1 = rungeKutta(h, t1, U[tid]);

        while (fabs(actual - y1) > TOL) {
            n = n*2;
            h1 = h/n;
            t2 = t1;
            y1 = U[tid];

            for (int k = 0; k<n; k++) {
                y = rungeKutta(h1,t2,y1);
                t2 = t2 + h1;
            }
        }

        result[tid] = y1;
    }
}

int main(int argc, char **argv) {


    int TRIALS = 10;
    double AVG = 0;

    for(int k = 0;k<TRIALS;k++){

    double y0 = 1.3;
    double t0 = 0.0;
    double tfinal = 20;
    double h = (tfinal-t0)/length;
    double t = t0;
    // timing
    cudaEvent_t start, stop;
    float runtime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double *U, *data, *result;
    cudaMemset(U, 0, length * sizeof(double));

    cudaMallocManaged(&result, length * sizeof(double));

    //U is the array for storing the initial course values for each processor using eulers method
    cudaMallocManaged(&U, length* sizeof(double));
    int num_blocks = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(num_blocks, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);

    cudaEventRecord(start, 0);
    U[0] = y0;
    t += h;
    for (int i = 1; i < length; i++) {
        U[i] = euler(t, y0, h);
        t += h;
        y0 = U[i];
        //printf("U[%d]: %f\n",i,U[i]);
    }


    solveODE<<<grid,block>>>(result,U, y0, t0, h);
    cudaEventRecord(stop, 0);

    cudaDeviceSynchronize();

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&runtime, start, stop);

    for (int i = 0; i < length; i++) {
        //printf("Process %d: y = %f U[%d]: %f\n", i, result[i],i,U[i]);
    }

    cudaFree(data);
    cudaFree(result);
    cudaFree(U);

    printf("Parallel GPU runtime = %f ms\n", runtime);
    if( k!=0){
        AVG += runtime;
    }

    }
    printf("Average parallel runtime = %f",AVG/(TRIALS-1));
    return 0;
}