#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TOL pow(10,-18)
#define NUM 100

double f(double t, double y) {
    return t * y;
}

double sol(double t, double t0, double y0) {
    return exp(pow(t,2)/2) * exp(log(y0)-(pow(t0,2)/2));
}

double euler(double t0, double y0, double h) {
    return y0 + h * f(t0, y0);
}

double rungeKutta(double t0, double y0, double h) {
    double k1, k2, k3, k4;
    k1 = h * f(t0, y0);
    k2 = h * f(t0 + h / 2, y0 + k1 / 2);
    k3 = h * f(t0 + h / 2, y0 + k2 / 2);
    k4 = h * f(t0 + h, y0 + k3);

    return y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
}


int main(int argc, char** argv) {
    double AVG = 0;
    int TRIALS = 10;
    double total_time[TRIALS];
    for (int l=0; l<TRIALS; l++) {
        clock_t start_time = clock();
        double y0 = 1.3;
        double t0 = 0.0;
        double t = t0;
        double y = y0;
        double tfinal = 20.0;
        double h = (tfinal - t0)/(NUM);

        double U[NUM];
        U[0] = y0;
        t = t + h;

        for (int i = 1; i<NUM; i++) {
            U[i] = euler(t,y,h);
            y = U[i];
            t = t + h;
        }


        double actual, h1;
        int n = 1;
        y = y0;

        for (int i = 0; i<NUM; i++) {
            actual = sol(t0 + h, t0, U[i]);
            //printf("actual: %.4f\n", actual);
            y = rungeKutta(t0, U[i], h);
            //printf("runge: %.4f\n",y);
            while(abs(actual-y)>TOL) {
                n = n*2;
                h1 = h/n;
                t = t0;
                y = U[i];

                for (int k = 0; k<n; k++) {
                    y = rungeKutta(t,y,h1);
                    t = t + h1;
                }
            }
            U[i] = y;
            t0 = t0 + h;
        }

        clock_t end_time = clock();

        total_time[l] = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        printf("Execution time of trial [%d]: %f seconds \n", l, total_time[l]);
        AVG += total_time[l];
    }
    
    printf("The average execution time of 10 trials is: %f ms", (AVG/TRIALS)*1000);
}