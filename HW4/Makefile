all : matmul
matmul : main.cc matmul.cc
	mpicxx main.cc matmul.cc -o matmul

run : 
	mpirun -np 4 --hostfile hosts matmul
