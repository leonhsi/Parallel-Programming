#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	int local_num_in_circle = 0;
	int global_num_in_circle = 0;

    // TODO: use MPI_Reduce
	double x, y, distance;
	int my_chunk = tosses / world_size;
	int my_start = world_rank * my_chunk;
	int my_end = (world_rank == world_size-1) ? tosses : my_start + my_chunk;

	unsigned int seed = time(NULL) * (world_rank + 1);
	for(int i=my_start; i<my_end; i++){
		x = (double)rand_r(&seed) / RAND_MAX * 2 - 1;
		y = (double)rand_r(&seed) / RAND_MAX * 2 - 1;
		distance = x*x + y*y;
		if(distance <= 1){
			local_num_in_circle++;
		}
	}

	MPI_Reduce(&local_num_in_circle, &global_num_in_circle, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // TODO: PI result
		pi_result = 4.0 * (double)global_num_in_circle / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
