/*
 * poisson.cuh
 *
 *  Created on: 28.09.2018
 *      Author: Karl Royen
 */

#ifndef POISSON_CUH_
#define POISSON_CUH_

#include <stdlib.h>
#include <cufft.h>
#include <complex.h>

#ifndef SINGLE_GPU
#include <cufftXt.h>
#endif

/** Public interface */
#define MAX_GPUS 8

typedef struct _poisson_plan {
	cufftComplex* diag[MAX_GPUS];
	int nx, ny, nz;
	cufftHandle transf_plan;
	cufftHandle itransf_plan;
	int dirichlet;
	void* workbuffer[MAX_GPUS];
	int r2c;
	int num_gpus;
} poisson_plan;

/** Free resources allocated by the plan. */
void poisson_destroy_plan(poisson_plan p);

/**
 * Execute the passed poisson solver plan for the passed src and write
 * result to the passed dst.
 *
 * Pointers to src and dst can be the same for inplace calculation.
 *
 * Src and dst should point to memory at the gpu. The data in src and dst
 * are real values with a padding of two in z direction. So a y-z-slice
 * of size 2 x 3 looks like
 *
 * 1 2 3 x x 4 5 6 x x
 *
 * in memory.
 *
 * In case of Dirichlet boundary conditions src and dst must point to a location
 * in memory twice as big as the normal dataset since that space is needed as
 * temporary memory during execution.
 */
void poisson_execute_plan(poisson_plan p, void* dst, void* src);

/**
 * Create a plan for solving the poisson equation at the gpu for data with the
 * passed parameters.
 *
 * nx, ny, nz - number of samples in x-, y- and z-direction
 * dx, dy, dz - sampling distances
 * dirichlet - if 0, use periodic boundary conditions. If 1, use Dirichlet BC.
 */
poisson_plan poisson_create_plan(int nx, int ny, int nz,
	double dx, double dy, double dz,
	int dirichlet, int num_gpus);

/**
 * Copy the passed input data set f at src from the host to the passed
 * dst location at the gpu.
 *
 * This functions adds the necessary two-padding in z direction when
 * uploading the continous data from the cpu.
 *
 * If the poisson equation is
 * A*f = g
 * then src points to the data of g.
 */
void poisson_plan_copy_to_host(poisson_plan p, float* dst, float* src);

/**
 * Copy the passed result data set f at dst from the gpu to the passed
 * src location at the host.
 *
 * This functions removes the two-padding in z direction from the result when
 * downloading the data from the gpu.
 *
 * If the poisson equation is
 * A*f = g
 * then dst points to the data of f.
 */
void poisson_plan_copy_to_gpu(poisson_plan p, float* dst, float* src);

/**
 * Allocate the space needed by the poisson solver of the passed plan for the
 * input/output data at the gpu.
 *
 * The caller is responsible for freeing the memory when it's not needed
 * anymore.
 */
float* poisson_plan_allocate(poisson_plan p);

#ifndef SINGLE_GPU
cudaLibXtDesc* poisson_mgpu_plan_allocate(poisson_plan p);
void poisson_mgpu_plan_copy_to_host(poisson_plan p, float* dst,
		cudaLibXtDesc* src);
void poisson_mgpu_plan_copy_to_gpu(poisson_plan p,
		cudaLibXtDesc* dst, float* src);
void poisson_mgpu_execute_plan(poisson_plan p,
		cudaLibXtDesc* dst, cudaLibXtDesc* src);
#endif



/** Internal interface */

/**
 * Calculate the diagonal entries of the diagonalized differentiation 
 * matrix D of the discretized Poisson equation
 * 	Df = g
 * for periodic or Dirichlet boundaries in three dimensions.
 *
 * k - output array of complex numbers of the diagonal entries. Should be
 *      large enough for nx*ny*nz entries (or nx*ny*(nz/2+1) if for_r2c 
 *      is set).
 * nx,ny,nz - the resolution of the dataset
 * dx,dy,dz - the sampling distances
 * dirichlet - define whether periodic (0) or Dirichlet (1) boundary
 * 		conditions should be used
 * for_r2c - if not zero, returns only nx*ny*(nz/2+1) entries of the
 * 		diagonal, since the others are symmetric anyway. This returns an 
 * 		diagonal entry set compatible for the result of fftws or cuffts 
 * 		real to complex fourier transforms.
 */
double complex* poisson_calcdiag(poisson_plan& p, 
        double dx, double dy, double dz);
void _create_transfer_plans(poisson_plan& p);
poisson_plan _init_plan(int nx, int ny, int nz, int dirichlet, int r2c,
		int num_gpus);
void _upload_diagonal(double complex* diag, poisson_plan& p);
size_t diag_size(poisson_plan p);
void scale_diagonal(double complex* diag, poisson_plan p);

#endif /* POISSON_CUH_ */
