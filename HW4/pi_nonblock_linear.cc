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
	int global_num_in_circle = 0;
	int local_num_in_circle[world_size+5];
	for(int i=0; i<world_size; i++){
		local_num_in_circle[i] = 0;
	}

    if (world_rank > 0)
    {
        // TODO: MPI workers
		int my_chunk = tosses / (world_size -1);
		int my_start = (world_rank-1) * my_chunk;
		int my_end = (world_rank == world_size-1) ? tosses : my_start + my_chunk;

		double x, y, distance;
		unsigned int seed = time(NULL) * (world_rank + 1);
		for(int i=my_start; i<my_end; i++){
			x = (double)rand_r(&seed) / RAND_MAX * 2 - 1;
			y = (double)rand_r(&seed) / RAND_MAX * 2 - 1;
			distance = x*x + y*y;
			if(distance <= 1){
				local_num_in_circle[world_rank]++;
			}
		}
		
		MPI_Send(&local_num_in_circle[world_rank], 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Request requests[world_size-1];
		for(int i=0; i<world_size-1; i++){
			requests[i] = MPI_REQUEST_NULL;
		}

		for(int rank=1; rank<world_size; rank++){
			MPI_Irecv(&local_num_in_circle[rank], 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &requests[rank-1]);
		}

        //MPI_Waitall(world_size-1, requests, MPI_STATUSES_IGNORE);
		for(int rank=1; rank<world_size; rank++){
			MPI_Wait(&requests[rank-1], MPI_STATUSES_IGNORE);
		}
    }

    if (world_rank == 0)
    {
        // TODO: PI result
		for(int i=1; i<world_size; i++){
			global_num_in_circle += local_num_in_circle[i];
		}

		pi_result = 4.0 * (double)(global_num_in_circle) / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();

    return 0;
}
