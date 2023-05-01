#include <stdio.h>
#include <omp.h>

#define UPTO 10000000

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


void openmp_primes(long int n) {
	long int i, num, divisor,quotient, remainder;

	if (n < 2) return;
	count = 1;                         /* 2 is the first prime */
	lastprime = 2;

	/* 
	 * Parallelize the serial algorithm but you are NOT allowed to change it!
	 * Don't add/remove/change global variables
	 */
	#pragma omp parallel for private(num, divisor,remainder,quotient) reduction(+:count) lastprivate(lastprime)
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


int main()
{
	printf("Serial and parallel prime number calculations:\n\n");
	
	/* Time the following to compare performance 
	 */
	double start = omp_get_wtime(); 
	serial_primes(UPTO);        /* time it */
	double finish = omp_get_wtime(); 
	double time_it_took = finish - start; 
	printf("[serial] count = %ld, last = %ld (time = %lf)\n", count, lastprime, time_it_took);
	//set the runtime threads to two for each next parallel region 
	omp_set_num_threads(4);
	start = omp_get_wtime(); 
	openmp_primes(UPTO);        /* time it */
	finish = omp_get_wtime(); 
	time_it_took = finish - start; 
	
	printf("[openMP] count = %ld, last = %ld (time = %lf)\n", count, lastprime, time_it_took);
	return 0;
}
