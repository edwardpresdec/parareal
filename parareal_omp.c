#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define TOL 0.000005

// Define the differential equation dy/dt = f(t, y)
double f(double t, double y) {
    return t * y; // Example: dy/dt = t * y
}

double sol(double t, double t0, double y0) {
    return exp(pow(t, 2) / 2) * exp(log(y0) - (pow(t0, 2) / 2));
}

double euler(double t0, double y0, double h) {
    return y0 + h * f(t0, y0);
}

double rungeKutta(double h, double t0, double y0) {
    double k1, k2, k3, k4;
    k1 = h * f(t0, y0);
    k2 = h * f(t0 + h / 2, y0 + k1 / 2);
    k3 = h * f(t0 + h / 2, y0 + k2 / 2);
    k4 = h * f(t0 + h, y0 + k3);

    return y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
}

int main(int argc, char **argv) {

    int num_threads = atoi(argv[1]); // Number of threads specified as a command-line argument

    double y0 = 1.0;
    double t0 = 0.0;
    double tfinal = 1.0;
    double h = (tfinal - t0) / num_threads;

    #pragma omp parallel num_threads(num_threads) 
    
    {
        int thread_id = omp_get_thread_num();
        double t = t0 + thread_id * h;
        double data = y0;

        // Perform the computation
        double actual = sol(t + h, t, data);

        printf("Thread ID: %d, Solution: %.8f, Data: %.8f\n", thread_id, actual, data);

        double y1 = rungeKutta(h, t, data);
        int n = 1;

        while (fabs(actual - y1) > TOL) {
            n *= 2;
            h = h / n;
            double t1 = t0 + thread_id * h;
            y1 = data;

            #pragma omp parallel for reduction(+:y1) // Distribute the loop iterations among threads
                for (int k = 0; k < n; k++) {
                    y1 += rungeKutta(h, t1 + k * h, y1); // Each thread computes part of the solution
                }
        }
    }

    return 0;
}