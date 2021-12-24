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
	//int *local_num_in_circle = (int *)malloc(world_size * sizeof(int));
	int local_num_in_circle = 0;
	int *global_num_in_circle = (int *)malloc(world_size * sizeof(int));
	for(int i=0; i<world_size; i++){
		//local_num_in_circle[i] = 0;
		global_num_in_circle[i] = 0;
	}

    // TODO: use MPI_Gather
	double x, y, distance;
	int my_chunk = tosses / world_size;
	int my_start = my_chunk * world_rank;
	int my_end = (world_rank == world_size-1) ? tosses : my_start + my_chunk;

	unsigned int seed = time(NULL) * (world_rank+1);
	for(int i=my_start; i<my_end; i++){
		x = (double)rand_r(&seed) / RAND_MAX * 2 - 1;
		y = (double)rand_r(&seed) / RAND_MAX * 2 - 1;
		distance = x*x + y*y;
		if(distance <= 1){
			//local_num_in_circle[world_rank]++;
			local_num_in_circle++;
		}
	}
	//printf("local number of rank %d : %d\n", world_rank, local_num_in_circle);

	MPI_Gather(&local_num_in_circle, 1, MPI_INT, global_num_in_circle, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // TODO: PI result
		int result = 0;
		for(int i=0; i<world_size; i++){
			result += global_num_in_circle[i];
			//printf("global number of rank %d : %d\n", i, global_num_in_circle[i]);
		}

		pi_result = 4.0 * (double)result / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
	//free(local_num_in_circle);
	free(global_num_in_circle);

    MPI_Finalize();
    return 0;
}
