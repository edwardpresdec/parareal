#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

#define TOL pow(10,-15)

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
    int TRIALS = 50;
    double total_time[TRIALS];
    int numproc, procid;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Status status;

    for (int l=0; l<TRIALS; l++) {
        double y0 = 1.3;
        double t0 = 0.0;
        double tfinal = 20.0;
        double h = (tfinal - t0)/(numproc);

        if (procid == 0) {
            clock_t start_time = clock();
            double U[numproc]; 
            double t = t0;
            U[0] = y0;
            t = t + h;
            for (int i = 1; i<numproc; i++) {
                U[i] = euler(t,y0,h);
                y0 = U[i];
                t = t + h;
                MPI_Send(&U[i], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }

            double recv_data;
            for(int i=0; i<numproc-1; i++){
                MPI_Recv(&recv_data, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                int source_rank = status.MPI_SOURCE;
                U[source_rank] = recv_data;
            }

            clock_t end_time = clock();
            total_time[l] = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
            printf("Execution time of trial [%d]: %f seconds \n", l, total_time[l]);
            if(l!=0) {
                AVG += total_time[l];
            }
            
        } else {
            double data, y1, t1, h1;
            int n = 1;
            t0 = h * procid;
            tfinal = t0 + h;
            MPI_Recv(&data, 1, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            double actual = sol(tfinal, t0, data);

            y1 = rungeKutta(t0,data,h);

            while (abs(actual-y1)>TOL) {
                n = n*2;
                h1 = h/n;
                t1 = t0;
                y1 = data;

                for (int k = 0; k<n; k++) {
                y1 = rungeKutta(t1,y1,h1);
                t1 = t1 + h1;
                }
            }
            MPI_Send(&y1, 1, MPI_DOUBLE, 0, procid, MPI_COMM_WORLD);
        }
    }

    if(procid==0) {
        printf("The average execution time of 10 trials is: %f ms", AVG/TRIALS*1000);
    }

    MPI_Finalize();
    
    return 0;
}