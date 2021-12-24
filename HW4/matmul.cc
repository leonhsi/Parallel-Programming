# include <stdio.h>
# include <stdlib.h>
# include <mpi.h>

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr){
    int world_size, world_rank;

	//init MPI
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0){

		scanf("%d %d %d", n_ptr, m_ptr, l_ptr);

		*a_mat_ptr = (int*)malloc(sizeof(int) * (*n_ptr) * (*m_ptr));
		*b_mat_ptr = (int*)malloc(sizeof(int) * (*m_ptr) * (*l_ptr));

    	int *tmp;

		for (int i = 0; i < (*n_ptr); i++){
		    for (int j = 0; j < (*m_ptr); j++){
				tmp = *a_mat_ptr + i * (*m_ptr) + j;
				scanf("%d", tmp);
		    }
		}

		for (int i = 0; i < (*m_ptr); i++){
		    for (int j = 0; j < (*l_ptr); j++){
				tmp = *b_mat_ptr + i * (*l_ptr) + j;
				scanf("%d", tmp);
		    }
		}

		/* debug purpose */
		// for (int i = 0; i < n; i++){
		//     for (int j = 0; j < m; j++){
		// 	ptr = *a_mat_ptr + i * m + j;
		// 	printf("%d ", *ptr);
		//     }
		//     printf("\n");
		// }

		// for (int i = 0; i < m; i++){
		//     for (int j = 0; j < l; j++){
		// 	ptr = *b_mat_ptr + i * l + j;
		// 	printf("%d ", *ptr);
		//     }
		//     printf("\n");
		// }
    }
}

void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat){
    int world_size, world_rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int source, mtype, rows, averow, extra, offset;
    int N, M, L;
    int i, j, k;
    if (world_rank == 0){
		int *c;
    	c = (int*)malloc(sizeof(int) * n * l);
        /* Send matrix data to the worker tasks */
        averow = n / (world_size-1);
        extra = n % (world_size-1);
        offset = 0;
        mtype = 1;
        for (int rank = 1; rank <world_size; rank++){
            MPI_Send(&n, 1, MPI_INT, rank, mtype, MPI_COMM_WORLD);
            MPI_Send(&m, 1, MPI_INT, rank, mtype, MPI_COMM_WORLD);
            MPI_Send(&l, 1, MPI_INT, rank, mtype, MPI_COMM_WORLD);
            rows = (rank <= extra)? averow + 1: averow;
            MPI_Send(&offset, 1, MPI_INT, rank, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, rank, mtype, MPI_COMM_WORLD);
	    	// printf("master send info to rank %d\n", rank);
            MPI_Send(&a_mat[offset * m], rows * m, MPI_INT, rank, mtype, MPI_COMM_WORLD);
            MPI_Send(&b_mat[0], m * l, MPI_INT, rank, mtype, MPI_COMM_WORLD);
            offset += rows;
        }
        /* Receive results from worker tasks */
        mtype = 2;
        for (source = 1; source < world_size; i++){
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset * l], rows * l, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
	    	// printf("master receive from %d\n", source);
        }
        /* Print results */
        for (i = 0; i < n; i++){
            for (j = 0; j < l; j++){
        		printf("%d", c[i * l + j]);
				if (j != l-1) printf(" ");
            }
            printf("\n");
        }
		free(c);
    }
    if (world_rank > 0){
        MPI_Recv(&N, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&M, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&L, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
		int *a;
    	int *b;
    	int *c;
    	a = (int*)malloc(sizeof(int) * N * M);
    	b = (int*)malloc(sizeof(int) * M * L);
    	c = (int*)malloc(sizeof(int) * N * L);
		// printf("n: %d, m: %d, l: %d\n", N, M, L);
        MPI_Recv(&offset, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
		// printf("rank %d receive from master\n", world_rank);
        MPI_Recv(&a[0], rows * M, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&b[0], M * L, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
		// printf("\n");
		// for (int i = 0; i < rows * M; i++) printf("a[%d]: %d\n", i, a[i]);
		// for (int i = 0; i < L * M; i++) printf("b[%d]: %d\n", i, b[i]);
		// printf("\n");

        for (k = 0; k < L; k++){
            for (i = 0; i < rows; i++){
    	    	c[i * L + k] = 0;
        		for (j = 0; j < M; j++){
        		    c[i * L + k] += a[i * M + j] * b[j * L + k];
		    		// printf("a[%d][%d] = %d\n", i, j, a[i*M + j]);
		    		// printf("b[%d][%d] = %d\n", j, k, b[j*L + k]);
        		}
            }
        }

        mtype = 2;
        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&c[0], rows * L, MPI_INT, 0, 2, MPI_COMM_WORLD);
		// printf("rank %d send result\n", world_rank);
		free(a);
    	free(b);
		free(c);
    }
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat){
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank == 0){
		free(a_mat);
		free(b_mat);
    }
}
