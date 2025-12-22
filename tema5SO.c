#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define data_length 300

sem_t H, O;
int h_count = 0;

void *hydrogen(void* arg)
{
    for(int i = 0; i < data_length; i++)
    {
        sem_wait(&H);
        sem_wait(&H);
        printf("HH");
        usleep(rand() % 2); //*random delaty
        h_count++;
        h_count++;
        if(h_count % 2 == 1) {
            sem_post(&H);
        } else {
            sem_post(&O); 
        }
    }
    return NULL;
}

void* oxygen(void* arg)
{
    for(int i = 0; i < data_length / 2; i++)
    {
        sem_wait(&O);
        printf("O ");
        usleep(rand() % 1); //*random delat
        sem_post(&H);
        sem_post(&H);
    }
    return NULL;
}

int main(void)
{
    setbuf(stdout, NULL); //?disable buffer
    pthread_t h_thread, o_thread;

    //* semaphore init = 0
    sem_init(&H, 0, 2);
    sem_init(&O, 0, 0);


    //*create threads
    pthread_create(&h_thread, NULL, hydrogen, NULL);
    pthread_create(&o_thread, NULL, oxygen, NULL);

    //* wait for finish
    pthread_join(h_thread, NULL);
    pthread_join(o_thread, NULL);

    //*clear
    sem_destroy(&H);
    sem_destroy(&O);

    return 0;
}

//HHO in loc de OOH
