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

    MPI_Win win;

    // TODO: MPI init
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	int *local_num_in_circle;

    if (world_rank == 0)
    {
        // Master
		MPI_Alloc_mem(world_size * sizeof(int), MPI_INFO_NULL, &local_num_in_circle);

		for(int i=0; i<world_size; i++){
			local_num_in_circle[i] = 0;
		}

		MPI_Win_create(local_num_in_circle, world_size * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    }
    else
    {
        // Workers
		MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

		double x, y, distance;
		int local_tmp = 0;
		int my_chunk = tosses / (world_size-1);
		int my_start = my_chunk * (world_rank-1);
		int my_end = (world_rank == world_size-1) ? tosses : my_start + my_chunk;
		
		unsigned int seed = time(NULL) * world_rank;
		for(int i=my_start; i<my_end; i++){
			x = (double)rand_r(&seed) / RAND_MAX * 2 - 1;
			y = (double)rand_r(&seed) / RAND_MAX * 2 - 1;
			distance = x*x + y*y;
			if(distance <= 1){
				local_tmp++;
			}
		}

		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
		MPI_Put(&local_tmp, 1, MPI_INT, 0, world_rank, 1, MPI_INT, win);
		MPI_Win_unlock(0, win);
    }

    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
		int global_num_in_circle = 0;
		for(int i=1; i<world_size; i++){
			global_num_in_circle += local_num_in_circle[i];
		}
		pi_result = 4.0 * (double)global_num_in_circle / (double)tosses;
		
		MPI_Free_mem(local_num_in_circle);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}
