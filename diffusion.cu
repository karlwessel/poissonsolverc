/*
 * diffusion.c
 *
 *  Created on: 28.09.2018
 *      Author: Karl Royen
 */

#include "diffusion.h"
#include <complex.h>
#include <cufft.h>
#include <stdio.h>

#ifndef SINGLE_GPU
#include <cufftXt.h>
#endif

void diffusion_destroy_plan(diffusion_plan p) {
	if(p.plan.num_gpus == 1)
		cudaFree(p.dest);
	else
		cufftXtFree((cudaLibXtDesc*)p.dest);

	poisson_destroy_plan(p.plan);
}

void diffusion_execute_plan(diffusion_plan p)
{
	if(p.plan.num_gpus == 1)
		poisson_execute_plan(p.plan, p.dest, p.dest);
	else
		poisson_mgpu_execute_plan(p.plan,
				(cudaLibXtDesc*)p.dest, (cudaLibXtDesc*)p.dest);
}

void create_crank_diag(double complex* diag, double dt, size_t n)
{
	for (int i = 0; i < n; i += 1)
	{
		diag[i] = (2.0-dt*diag[i])/(2.0+dt*diag[i]);
	}
	return;
}


diffusion_plan diffusion_create_plan(int nx, int ny, int nz,
	double dx, double dy, double dz, double dt,
	int dirichlet, int num_gpus)
{
	diffusion_plan p;
	int r2c = num_gpus > 1 ? 0 : 1;
	p.plan = _init_plan(nx, ny, nz, dirichlet, r2c, num_gpus);
	p.dest = 0;

	_create_transfer_plans(p.plan);

	double complex* diag = poisson_calcdiag(p.plan, dx, dy, dz);
	create_crank_diag(diag, dt, diag_size(p.plan));
	scale_diagonal(diag, p.plan);
	_upload_diagonal(diag, p.plan);
	free(diag);

	if(num_gpus == 1)
		p.dest = poisson_plan_allocate(p.plan);
	else
		p.dest = poisson_mgpu_plan_allocate(p.plan);
	return p;
}

void diffusion_direct(float* u, int nx, int ny, int nz,
	double dx, double dy, double dz, double dt,
	int num_steps, int dirichlet)
{
	// create the normal diffusion plan...
	diffusion_plan p;
	int r2c = 1;
	p.plan = _init_plan(nx, ny, nz, dirichlet, r2c, 1);
	p.dest = 0;

	_create_transfer_plans(p.plan);

	double complex* diag = poisson_calcdiag(p.plan, dx, dy, dz);
	create_crank_diag(diag, dt, diag_size(p.plan));

	//... but instead of executing the diffusion step num_steps times
	// just take the diagonal entries to the power of num_steps ...
	for(int i = 0; i < diag_size(p.plan); ++i) {
		diag[i] = cpow(diag[i], num_steps);
	}

	//... and proceed normal finishing the plan and moving data to gpu...
	scale_diagonal(diag, p.plan);
	_upload_diagonal(diag, p.plan);
	free(diag);

	p.dest = poisson_plan_allocate(p.plan);

	// copy ic to plan memory
	diffusion_plan_copy_to_gpu(p, u);

	// ... but do only one diffusion step
	diffusion_execute_plan(p);

	// copy back from plan memory
	diffusion_plan_copy_to_host(p, u);

	diffusion_destroy_plan(p);
}

void diffusion_plan_copy_to_host(diffusion_plan p, float* dst)
{
	if(p.plan.num_gpus == 1)
		poisson_plan_copy_to_host(p.plan, dst, (float*)p.dest);
	else
		poisson_mgpu_plan_copy_to_host(p.plan, dst, (cudaLibXtDesc*)p.dest);
}

void diffusion_plan_copy_to_gpu(diffusion_plan p, float* src)
{
	if(p.plan.num_gpus == 1)
		poisson_plan_copy_to_gpu(p.plan, (float*)p.dest, src);
	else
		poisson_mgpu_plan_copy_to_gpu(p.plan, (cudaLibXtDesc*)p.dest, src);
}

void diffusion(float* u, int nx, int ny, int nz,
	double dx, double dy, double dz, double dt,
	int numsteps, int dirichlet, int num_gpus)
{
	// plan solver
	diffusion_plan p = diffusion_create_plan(nx, ny, nz, dx, dy, dz, dt,
			dirichlet, num_gpus);

	// copy ic to plan memory
	diffusion_plan_copy_to_gpu(p, u);

	printf("Executing for %d steps...\n", numsteps);
	// execute iteration steps
	for(int i = 0; i < numsteps; ++i) diffusion_execute_plan(p);

	// copy back from plan memory
	diffusion_plan_copy_to_host(p, u);

	diffusion_destroy_plan(p);
}



