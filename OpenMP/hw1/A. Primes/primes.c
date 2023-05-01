#include <stdio.h>
#include <omp.h>

#define UPTO 10000000
//since we are using a chunk size we use the following empirical rule 
#define CHUNK_SIZE (int)(UPTO * 0.01) 
long int count,      /* number of primes */
         lastprime;  /* the last prime found */


void serial_primes(long int n) {
	long int i, num, divisor, quotient, remainder;

	if (n < 2) return;
	count = 1;                         /* 2 is the first prime */
	lastprime = 2;

	for (i = 0; i < (n-1)/2; ++i) {    /* For every odd number */
		num = 2*i + 3;

		divisor = 1;
		do 
		{
			divisor += 2;                  /* Divide by the next odd */
			quotient  = num / divisor;  
			remainder = num % divisor;  
		} while (remainder && divisor <= quotient);  /* Don't go past sqrt */

		if (remainder || divisor == num) /* num is prime */
		{
			count++;
			lastprime = num;
		}
	}
}


void openmp_primes_with_chunk_size(long int n) {
	long int i, num, divisor,quotient, remainder;

	if (n < 2) return;
	count = 1;                         /* 2 is the first prime */
	lastprime = 2;

	/* 
	 * Parallelize the serial algorithm but you are NOT allowed to change it!
	 * Don't add/remove/change global variables
	 */
	//static scheduling is best here since it is essential to know the thread execution order
	#pragma omp parallel for schedule(static,CHUNK_SIZE)  private(num, divisor,remainder,quotient) reduction(+:count) lastprivate(lastprime)
	for (i = 0; i < (n-1)/2; ++i) {    /* For every odd number */
		num = 2*i + 3;

		divisor = 1;
		do 
		{
			divisor += 2;                  /* Divide by the next odd */
			quotient  = num / divisor;  
			remainder = num % divisor;  
		//reason for not going past sqrt: if the number was not a prime 
		//it could have been expressed as a factor of two numbers before the sqrt
		} while (remainder && divisor <= quotient);  /* Don't go past sqrt */

		if (remainder || divisor == num) /* num is prime */
		{
			count++;
			lastprime = num;
		}
	}
}

void openmp_primes(long int n) {
	long int i, num, divisor,quotient, remainder;

	if (n < 2) return;
	count = 1;                         /* 2 is the first prime */
	lastprime = 2;

	/* 
	 * Parallelize the serial algorithm but you are NOT allowed to change it!
	 * Don't add/remove/change global variables
	 */
	//static scheduling is best here since it is essential to know the thread execution order
	#pragma omp parallel for schedule(static)  private(num, divisor,remainder,quotient) reduction(+:count) lastprivate(lastprime)
	for (i = 0; i < (n-1)/2; ++i) {    /* For every odd number */
		num = 2*i + 3;

		divisor = 1;
		do 
		{
			divisor += 2;                  /* Divide by the next odd */
			quotient  = num / divisor;  
			remainder = num % divisor;  
		//reason for not going past sqrt: if the number was not a prime 
		//it could have been expressed as a factor of two numbers before the sqrt
		} while (remainder && divisor <= quotient);  /* Don't go past sqrt */

		if (remainder || divisor == num) /* num is prime */
		{
			count++;
			lastprime = num;
		}
	}
}

int main()
{
	printf("Serial and parallel prime number calculations:\n\n");
	
	//prevent dynamic thread number 
	omp_set_dynamic(0); 

	/* Time the following to compare performance 
	 */


	double start = omp_get_wtime(); 
	serial_primes(UPTO);        /* time it */
	double finish = omp_get_wtime(); 
	double time_it_took = finish - start; 
	printf("[serial] count = %ld, last = %ld (time = %lf)\n", count, lastprime, time_it_took);


	start = omp_get_wtime(); 
	openmp_primes(UPTO);        /* time it */
	finish = omp_get_wtime(); 
	time_it_took = finish - start; 
	
	printf("[openMP without chunk size] count = %ld, last = %ld (time = %lf)\n", count, lastprime, time_it_took);
	start = omp_get_wtime(); 
	openmp_primes_with_chunk_size(UPTO);        /* time it */
	finish = omp_get_wtime(); 
	time_it_took = finish - start; 
	
	printf("[openMP with chunk size] count = %ld, last = %ld (time = %lf)\n", count, lastprime, time_it_took);
	return 0;
}
