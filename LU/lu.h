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

#ifndef _LU_H
#define _LU_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <malloc.h>


#define MAXRAND					32767.0
#define DEFAULT_N				512
#define DEFAULT_B				16
#define PAGE_SIZE				4096
#define min(a, b)				((a) < (b) ? (a) : (b))

long n = DEFAULT_N;          /* The size of the matrix */
long block_size = DEFAULT_B; /* Block dimension */
double *a;                   /* a = lu; l and u both placed back in a */
double *rhs;
long test_result = 0;        /* Test result of factorization? */
long doprint = 0;            /* Print out matrix values? */


void OneSolve ( long n, long block_size );
void lu0 ( double *a, long n, long stride );
void bdiv ( double *a, double *diag, long stride_a, long stride_diag, long dimi, long dimk );
void bmodd ( double *a, double *c, long dimi, long dimj, long stride_a, long stride_c );
void bmod ( double *a, double *b, double *c, long dimi, long dimj, long dimk, long stride );
void daxpy ( double *a, double *b, long n, double alpha );
void lu ( long n, long bs );
void InitA ( double *rhs );
double TouchA ( long bs );
void PrintA ( void );
void CheckResult ( long n, double *a, double *rhs );
void printerr ( char *s );

#endif
