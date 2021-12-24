#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<pthread.h>

typedef struct thread_data{
    long long int start;
    long long int tosses_per_thread;
}thread_data;

unsigned long long total_sum= 0;
pthread_mutex_t mutex;

void *count(void *args){
    thread_data tdata = *(thread_data *)args;
    
    long long sum = 0;

    double x,y;

    unsigned int seed = time(NULL) * (tdata.start+1);

    for(long long int i=tdata.start; i<(tdata.tosses_per_thread + tdata.start); i++){
        x = (double)rand_r(&seed)/RAND_MAX * 2 -1;
        y = (double)rand_r(&seed)/RAND_MAX * 2 -1;
        if(x*x + y*y <= 1.0){
			sum++;
		}
        //printf("x[%d] : %lf\ny[%d] : %lf\n", i,x,i,y);
    }
	//printf("thread %d with sum : %lld\n",tdata.rank,sum);
    
    pthread_mutex_lock(&mutex); 
    total_sum += sum;
    pthread_mutex_unlock(&mutex); 

    pthread_exit(NULL);
}


int main(int argc, char **argv){
    
    if(argc != 3 ){
        printf("input error\n");
        exit(0);
    }

    int num_thread = atoi(argv[1]);
    long long int num_tosses = atoi(argv[2]);
    long long int tosses_per_thread = num_tosses / num_thread;

    pthread_t *thread_handles;
    thread_handles = (pthread_t *)malloc(num_thread * sizeof(pthread_t));
    pthread_mutex_init(&mutex, NULL);
    thread_data tdata[num_thread];

    for(int i=0; i<num_thread; i++){
        tdata[i].tosses_per_thread = tosses_per_thread;
        tdata[i].start = i * tosses_per_thread ;

        pthread_create(&thread_handles[i], NULL, count, (void *)&tdata[i]);
    }

    for(int i=0; i<num_thread; i++){
        pthread_join(thread_handles[i], NULL);
		//printf("thread %d finished\n",i);
    }
    
    pthread_mutex_destroy(&mutex);
    free(thread_handles);
    //printf("total_sum : %lld\n",total_sum);
    printf("%lf\n",4.0 * (double)total_sum / (double)num_tosses);
    return 0;
}
