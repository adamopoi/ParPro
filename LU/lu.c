/*************************************************************************/
/*                                                                       */
/*  Copyright (c) 1994 Stanford University                               */
/*                                                                       */
/*  All rights reserved.                                                 */
/*                                                                       */
/*  Permission is given to use, copy, and modify this software for any   */
/*  non-commercial purpose as long as this copyright notice is not       */
/*  removed.  All other uses, including redistribution in whole or in    */
/*  part, are forbidden without prior written permission.                */
/*                                                                       */
/*  This software is provided with absolutely no warranty and no         */
/*  support.                                                             */
/*                                                                       */
/*************************************************************************/

/*************************************************************************/
/*                                                                       */
/*  Parallel dense blocked LU factorization (no pivoting)                */
/*                                                                       */
/*  This version contains one dimensional arrays in which the matrix     */
/*  to be factored is stored.                                            */
/*                                                                       */
/*  Command line options:                                                */
/*                                                                       */
/*  -nN : Decompose NxN matrix.                                          */
/*  -bB : Use a block size of B. BxB elements should fit in cache for    */
/*        good performance. Small block sizes (B=8, B=16) work well.     */
/*  -s  : Print individual processor timing statistics.                  */
/*  -t  : Test output.                                                   */
/*  -o  : Print out matrix values.                                       */
/*  -h  : Print out command line options.                                */
/*                                                                       */
/*  Note: This version works under both the FORK and SPROC models        */
/*                                                                       */
/*************************************************************************/

/*
 * Modified by Ioannis E. Venetis for use as an assignment in the course:
 *
 * Parallel Computing
 * Computer Engineering and Informatics Department
 * University of Patras, Greece
 */

#include "lu.h"

/******************************************************************************/

int main ( int argc, char *argv[] )
{
	long ch;

	while ( ( ch = getopt ( argc, argv, "n:b:ctoh" ) ) != -1 ) {
		switch ( ch ) {
		case 'n':
			n = atoi ( optarg );
			break;
		case 'b':
			block_size = atoi ( optarg );
			break;
		case 't':
			test_result = !test_result;
			break;
		case 'o':
			doprint = !doprint;
			break;
		case 'h':
			printf ( "Usage: LU <options>\n\n" );
			printf ( "options:\n" );
			printf ( "  -nN : Decompose NxN matrix.\n" );
			printf ( "  -bB : Use a block size of B. BxB elements should fit in cache for \n" );
			printf ( "        good performance. Small block sizes (B=8, B=16) work well.\n" );
			printf ( "  -c  : Copy non-locally allocated blocks to local memory before use.\n" );
			printf ( "  -t  : Test output.\n" );
			printf ( "  -o  : Print out matrix values.\n" );
			printf ( "  -h  : Print out command line options.\n\n" );
			printf ( "Default: LU -n%1d -b%1d\n", DEFAULT_N, DEFAULT_B );
			exit ( 0 );
			break;
		}
	}

	printf ( "\n" );
	printf ( "Blocked Dense LU Factorization\n" );
	printf ( "     %ld by %ld Matrix\n",n,n );
	printf ( "     %ld by %ld Element Blocks\n",block_size,block_size );
	printf ( "\n" );
	printf ( "\n" );

	a = ( double * ) malloc ( n*n*sizeof ( double ) );;
	if ( a == NULL ) {
		printerr ( "Could not malloc memory for a.\n" );
		exit ( -1 );
	}
	rhs = ( double * ) malloc ( n*sizeof ( double ) );;
	if ( rhs == NULL ) {
		printerr ( "Could not malloc memory for rhs.\n" );
		exit ( -1 );
	}

	InitA ( rhs );
	if ( doprint ) {
		printf ( "Matrix before decomposition:\n" );
		PrintA();
	}

	OneSolve ( n, block_size );

	if ( doprint ) {
		printf ( "\nMatrix after decomposition:\n" );
		PrintA();
	}

	if ( test_result ) {
		printf ( "                             TESTING RESULTS\n" );
		CheckResult ( n, a, rhs );
	}

	exit ( 0 );
}

/******************************************************************************/

void OneSolve ( long n, long block_size )
{
	unsigned long	start, end;
	struct timeval  FullTime;

	/* to remove cold-start misses, all processors begin by touching a[] */
	TouchA ( block_size );

	gettimeofday(&FullTime, NULL);
	start = (unsigned long)(FullTime.tv_usec + FullTime.tv_sec * 1000000);

	lu ( n, block_size );

	gettimeofday(&FullTime, NULL);
	end = (unsigned long)(FullTime.tv_usec + FullTime.tv_sec * 1000000);

	printf("Total execution time: %f seconds\n\n", (double)(end - start) / 1000000.0);
}

/******************************************************************************/

void lu0 ( double *a, long n, long stride )
{
	long j, k;
	double alpha;

	for ( k=0; k<n; k++ ) {
		/* modify subsequent columns */
		for ( j=k+1; j<n; j++ ) {
			a[k+j*stride] /= a[k+k*stride];
			alpha = -a[k+j*stride];
			daxpy ( &a[k+1+j*stride], &a[k+1+k*stride], n-k-1, alpha );
		}
	}
}

/******************************************************************************/

void bdiv ( double *a, double *diag, long stride_a, long stride_diag, long dimi, long dimk )
{
	long j, k;
	double alpha;

	for ( k=0; k<dimk; k++ ) {
		for ( j=k+1; j<dimk; j++ ) {
			alpha = -diag[k+j*stride_diag];
			daxpy ( &a[j*stride_a], &a[k*stride_a], dimi, alpha );
		}
	}
}

/******************************************************************************/

void bmodd ( double *a, double *c, long dimi, long dimj, long stride_a, long stride_c )
{
	long j, k;
	double alpha;

	for ( k=0; k<dimi; k++ )
		for ( j=0; j<dimj; j++ ) {
			c[k+j*stride_c] /= a[k+k*stride_a];
			alpha = -c[k+j*stride_c];
			daxpy ( &c[k+1+j*stride_c], &a[k+1+k*stride_a], dimi-k-1, alpha );
		}
}

/******************************************************************************/

void bmod ( double *a, double *b, double *c, long dimi, long dimj, long dimk, long stride )
{
	long j, k;
	double alpha;

	for ( k=0; k<dimk; k++ ) {
		for ( j=0; j<dimj; j++ ) {
			alpha = -b[k+j*stride];
			daxpy ( &c[j*stride], &a[k*stride], dimi, alpha );
		}
	}
}

/******************************************************************************/

void daxpy ( double *a, double *b, long n, double alpha )
{
	long i;

	for ( i=0; i<n; i++ ) {
		a[i] += alpha*b[i];
	}
}

/******************************************************************************/

void lu ( long n, long bs )
{
	long i, il, j, jl, k, kl, I, J, K;
	double *A, *B, *C, *D;
	long strI;

	strI = n;
	for ( k=0, K=0; k<n; k+=bs, K++ ) {
		kl = k+bs;
		if ( kl>n ) {
			kl = n;
		}

		/* factor diagonal block */
		A = & ( a[k+k*n] );
		lu0 ( A, kl-k, strI );

		/* divide column k by diagonal block */
		D = & ( a[k+k*n] );
		for ( i=kl, I=K+1; i<n; i+=bs, I++ ) {
			il = i + bs;
			if ( il > n ) {
				il = n;
			}
			A = & ( a[i+k*n] );
			bdiv ( A, D, strI, n, il-i, kl-k );
		}
		/* modify row k by diagonal block */
		for ( j=kl, J=K+1; j<n; j+=bs, J++ ) {
			jl = j+bs;
			if ( jl > n ) {
				jl = n;
			}
			A = & ( a[k+j*n] );
			bmodd ( D, A, kl-k, jl-j, n, strI );
		}

		/* modify subsequent block columns */
		for ( i=kl, I=K+1; i<n; i+=bs, I++ ) {
			il = i+bs;
			if ( il > n ) {
				il = n;
			}
			A = & ( a[i+k*n] );
			for ( j=kl, J=K+1; j<n; j+=bs, J++ ) {
				jl = j + bs;
				if ( jl > n ) {
					jl = n;
				}
				B = & ( a[k+j*n] );
				C = & ( a[i+j*n] );
				bmod ( A, B, C, il-i, jl-j, kl-k, n );
			}
		}
	}
}

/******************************************************************************/

void InitA ( double *rhs )
{
	long i, j;

	srand48 ( ( long ) 1 );
	for ( j=0; j<n; j++ ) {
		for ( i=0; i<n; i++ ) {
			a[i+j*n] = ( double ) lrand48() /MAXRAND;
			if ( i == j ) {
				a[i+j*n] *= 10;
			}
		}
	}

	for ( j=0; j<n; j++ ) {
		rhs[j] = 0.0;
	}
	for ( j=0; j<n; j++ ) {
		for ( i=0; i<n; i++ ) {
			rhs[i] += a[i+j*n];
		}
	}
}

/******************************************************************************/

double TouchA ( long bs )
{
	long i, j, I, J;
	double tot = 0.0;

	for ( J=0; J*bs<n; J++ ) {
		for ( I=0; I*bs<n; I++ ) {
			for ( j=J*bs; j< ( J+1 ) *bs && j<n; j++ ) {
				for ( i=I*bs; i< ( I+1 ) *bs && i<n; i++ ) {
					tot += a[i+j*n];
				}
			}
		}
	}
	return ( tot );
}

/******************************************************************************/

void PrintA()
{
	long i, j;

	for ( i=0; i<n; i++ ) {
		for ( j=0; j<n; j++ ) {
			printf ( "%8.1f ", a[i+j*n] );
		}
		printf ( "\n" );
	}
}

/******************************************************************************/

void CheckResult ( long n, double *a, double *rhs )
{
	long i, j, bogus = 0;
	double *y, diff, max_diff;

	y = ( double * ) malloc ( n*sizeof ( double ) );
	if ( y == NULL ) {
		printerr ( "Could not malloc memory for y\n" );
		exit ( -1 );
	}
	for ( j=0; j<n; j++ ) {
		y[j] = rhs[j];
	}
	for ( j=0; j<n; j++ ) {
		y[j] = y[j]/a[j+j*n];
		for ( i=j+1; i<n; i++ ) {
			y[i] -= a[i+j*n]*y[j];
		}
	}

	for ( j=n-1; j>=0; j-- ) {
		for ( i=0; i<j; i++ ) {
			y[i] -= a[i+j*n]*y[j];
		}
	}

	max_diff = 0.0;
	for ( j=0; j<n; j++ ) {
		diff = y[j] - 1.0;
		if ( fabs ( diff ) > 0.00001 ) {
			bogus = 1;
			max_diff = diff;
		}
	}
	if ( bogus ) {
		printf ( "TEST FAILED: (%.5f diff)\n", max_diff );
	} else {
		printf ( "TEST PASSED\n" );
	}
	free ( y );
}

/******************************************************************************/

void printerr ( char *s )
{
	fprintf ( stderr,"ERROR: %s\n",s );
}

/******************************************************************************/
