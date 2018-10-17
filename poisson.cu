/*
 * poisson.cu
 *
 *  Created on: 28.09.2018
 *      Author: Karl Royen
 */


#include <fftw3.h>
#include <complex.h>
#include <stdlib.h>
#include <stdio.h>
#include "poisson.cuh"
#include <cufft.h>
#include <assert.h>

#ifndef SINGLE_GPU
#include <cufftXt.h>
#endif

#define checkResult(x) _checkResult(x, __LINE__)
#define check(x) _check(x, __LINE__)


bool _checkResult(cufftResult r, int line)
{
	if(r!=CUFFT_SUCCESS) {
		printf("Call at line %d failed with error code %d\n", line, r);
        exit(1);
	}
	return r==CUFFT_SUCCESS;
}

bool _check(cudaError_t r, int line)
{
	if(r!=cudaSuccess) {
		printf("Call at line %d failed with error code %d\n", line, r);
        exit(1);
	}
	return r==cudaSuccess;
}



// Utility routine to perform complex pointwise multiplication
__global__ void elemwise_mul(const cufftComplex *a, cufftComplex *b, int size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = cuCmulf(b[i], a[i]);
    }
    return;
}


/** Inplace complex to complex fourier transform of array k of size n*/
void fft(fftw_complex* k, int n)
{
	fftw_plan plan = fftw_plan_dft_1d(n, k, k, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

/** Inplace real to complex fourier transform of array k of size n.
 *
 * k should be large enough to hold n/2+1 complex entries.*/
void fftr2c(fftw_complex* k, int n)
{
	fftw_plan plan = fftw_plan_dft_r2c_1d(n, (double*)k, k, FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

/** Inplace discrete cosine transform of type I of array x of size n*/
void DCTI(double* x, int n)
{
	fftw_plan p = fftw_plan_r2r_1d(n, x, x,
			FFTW_REDFT00, FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);
}

/**
 * Calculate the diagonal entries of the diagonalized differentiation matrix D
 * of the discretized poisson equation
 * 	Df = g
 * for periodic boundaries in one dimension.
 *
 * k - output array of complex numbers of the diagonal entries. Should be large
 * enough for n complex entries.
 * n - the resolution of the dataset
 * dx - the sampling distance
 */
int diagonalperiodic(double complex* k, int n, double dx)
{
	size_t size = n;

	double fak = 1.0 / (dx*dx) / 12.0;
	k[0] = -30.0 * fak;
	k[1] = 16.0 * fak;
	k[2] = -fak;
	for(size_t i = 3; i < size-2; ++i) k[i] = 0.0;
	k[n-1] = 16.0 * fak;
	k[n-2] = -fak;

	fft((fftw_complex*)k, n);

	return n;
}

/**
 * Calculate n/2+1 of the diagonal entries of the diagonalized differentiation
 * matrix D of the discretized poisson equation
 * 	Df = g
 * for periodic boundaries in one dimension.
 *
 * The result is compatible for results of fftws or cuffts real to complex
 * transformations.
 *
 * k - output array of complex numbers of the diagonal entries. Should be large
 * enough for n/2+1 complex entries.
 * n - the resolution of the dataset
 * dx - the sampling distance
 */
int diagonalperiodicr2c(double complex* k, int n, double dx)
{
	double* kin = (double*)k;

	double fak = 1.0 / (dx*dx) / 12.0;
	kin[0] = -30.0 * fak;
	kin[1] = 16.0 * fak;
	kin[2] = -fak;
	for(size_t i = 3; i < n-2; ++i) kin[i] = 0.0;
	kin[n-1] = 16.0 * fak;
	kin[n-2] = -fak;

	fftr2c((fftw_complex*)k, n);

	return n;
}


/**
 * Calculate the diagonal entries of the diagonalized differentiation matrix D
 * of the discretized poisson equation
 * 	Df = g
 * for Dirichlet-boundaries in one dimension.
 *
 * Assumes f and g such that the boundaries (which are zero) are not included
 * in the data set to calculate.
 *
 * k - output array of complex numbers of the diagonal entries. Should be large
 * enough for n complex entries.
 * n - the resolution of the dataset
 * dx - the sampling distance
 */
int diagonaldirichlet(double complex* k, int n, double dx)
{
	size_t size = n+2;
	double* temp = (double*)malloc(sizeof(double)*size);

	double fak = 1.0 / (dx*dx) / 12.0;
	temp[0] = -30.0 * fak;
	temp[1] = 16.0 * fak;
	temp[2] = -fak;
	for(size_t i = 3; i < size; ++i) temp[i] = 0.0;

	DCTI(temp, size);
	for(size_t i = 0; i < n; ++i) k[i] = temp[i+1];
	free(temp);

	return n;
}

/** Same as diagonal dirichlet but includes one more value in front (which is
 * ignored by the poisson solver in the end result but needed by the fft)*/
int diagonaldirichlet_ext(double complex* k, int n, double dx)
{
	size_t size = n+1;
	double* temp = (double*)malloc(sizeof(double)*size);

	double fak = 1.0 / (dx*dx) / 12.0;
	temp[0] = -30.0 * fak;
	temp[1] = 16.0 * fak;
	temp[2] = -fak;
	for(size_t i = 3; i < size; ++i) temp[i] = 0.0;

	DCTI(temp, size);
	for(size_t i = 0; i < n; ++i) k[i] = temp[i];
	free(temp);

	return n;
}


cufftHandle _create_plan(poisson_plan p, cufftType type, size_t *worksize)
{
	int nx = p.nx;
	if(p.dirichlet) nx *= 2;

	cufftHandle plan;
	// plan creation...
	cufftCreate(&plan);

	cufftSetAutoAllocation(plan, 0);

	if(p.num_gpus > 1) {
		// set gpus to use
		int whichGPUs[MAX_GPUS];
		for(int i = 0; i < MAX_GPUS; ++i) whichGPUs[i] = i;
		checkResult(cufftXtSetGPUs(plan, p.num_gpus, whichGPUs));
	}

	// init plans and get their needed workbuffer size
	checkResult(cufftMakePlan3d(plan, nx,p.ny,p.nz, type, worksize));

	return plan;
}

void _create_transfer_plans(poisson_plan& p)
{
	// multi gpu needs c2c-transform at the moment
	assert(p.num_gpus == 1 || !p.r2c);

	// get the plans and their needed workbuffer size
	size_t worksize[MAX_GPUS];
	if(p.r2c) {
		p.transf_plan = _create_plan(p, CUFFT_R2C, worksize);
		size_t worksizeBW[MAX_GPUS];
		p.itransf_plan = _create_plan(p, CUFFT_C2R, worksizeBW);
		// determine whether FW or BW transform needs more workarea and use the
		// larger
		for(int i = 0; i < p.num_gpus; ++i) {
			if (worksizeBW[i] > worksize[i]) worksize[i] = worksizeBW[i];
		}
	} else {
		p.transf_plan = _create_plan(p, CUFFT_C2C, worksize);
		p.itransf_plan = p.transf_plan;
	}

	// allocate workbuffer
	for(int i = 0; i < p.num_gpus; ++i) {
		cudaSetDevice(i);
		check(cudaMalloc(&p.workbuffer[i], worksize[i]));
	}

	// and tell it to the plans
	if(p.num_gpus == 1) {
		cufftSetWorkArea(p.transf_plan, p.workbuffer[0]);
		if(p.r2c) cufftSetWorkArea(p.itransf_plan, p.workbuffer[0]);
	} else {
		cufftXtSetWorkArea(p.transf_plan, p.workbuffer);
		if(p.r2c) cufftXtSetWorkArea(p.itransf_plan, p.workbuffer);
	}
}

size_t diag_size(poisson_plan p)
{
	int nx = p.dirichlet ? p.nx : p.nx;
	if (p.r2c)
		return nx*p.ny*(p.nz/2+1);
	else
		return nx*p.ny*p.nz;
}

float* poisson_plan_allocate(poisson_plan p)
{
	assert(p.num_gpus == 1);

	int nx = p.dirichlet ? p.nx*2 : p.nx;
	int nz = p.r2c ? p.nz/2+1 : p.nz;
	size_t size = nx*p.ny*nz;
	float* out;
	// in and out memory allocation
	check(cudaMalloc(&out, sizeof(cufftComplex)*size));

	return out;
}

void scale_diagonal(double complex* diag, poisson_plan p)
{
	size_t size = diag_size(p);

	// apply normalization for backwards transform to diagonal entries
	double complex Z = 1.0 /(p.nx*p.ny*p.nz);
	if(p.dirichlet) Z /= 2.0;
	for (int i = 0; i < size; i += 1) diag[i] = Z / diag[i];
}

void _upload_diagonal(double complex* diag, poisson_plan& p)
{
	size_t size = diag_size(p);

	// convert double to float
	float complex* tmp = (float complex*)malloc(sizeof(float complex)*size);
	for(size_t i = 0; i < size; ++i) tmp[i] = diag[i];

	// copy float diagonal entries to gpu(s)
	// for multi-gpu fft the data is divided along the y-axis *after*
	// the forward transform, therefore the diagonal also has to be divided
	// that way and transfered to each gpu

	// since the blocks divided along the y-axis are not continuous at the host
	// we use memcpy2d to transfer the not continuous blocks at host to
	// continuous blocks at the gpus
	int nz = p.r2c ? p.nz/2+1 : p.nz;
	size_t slice_size = nz*p.ny/p.num_gpus;
	size_t src_width = sizeof(cufftComplex)*nz*p.ny;
	size_t dst_width = sizeof(cufftComplex)*slice_size;
	size_t num_cols = p.nz;
	for(int i = 0; i < p.num_gpus; ++i) {
		cudaSetDevice(i);
		check(cudaMalloc(&p.diag[i], dst_width*num_cols));
		cudaMemcpy2D(p.diag[i], dst_width,
					tmp+i*slice_size, src_width,
					dst_width, num_cols,
					cudaMemcpyHostToDevice);
	}

	free(tmp);
}


double complex* poisson_calcdiag(poisson_plan& p,
		double dx, double dy, double dz)
{
	int nx = p.dirichlet ? p.nx : p.nx;
	size_t size = diag_size(p);

	double complex* diag = (double complex*)malloc(sizeof(double complex)*size);

	double complex* kx = (double complex*) malloc(sizeof(double complex)*nx);
	if (p.dirichlet){
		diagonaldirichlet_ext(kx, nx, dx);
	}else
		diagonalperiodic(kx, nx, dx);

	double complex* ky = (double complex*) malloc(sizeof(double complex)*p.ny);
	diagonalperiodic(ky, p.ny, dy);

	int zlen = p.r2c ? p.nz/2+1 : p.nz;
	double complex* kz = (double complex*) malloc(sizeof(double complex)*zlen);
	p.r2c ? diagonalperiodicr2c(kz, p.nz, dz) : diagonalperiodic(kz, p.nz, dz);

	double complex* k = diag;
	for(int ix = 0; ix < nx;++ix) {
		for(int iy = 0; iy < p.ny;++iy) {
			for(int iz = 0; iz < zlen;++iz) {
				(*k++) = kx[ix]+ky[iy]+kz[iz];
			}
		}
	}
	free(kx); free(ky); free(kz);
	return diag;
}

poisson_plan _init_plan(int nx, int ny, int nz, int dirichlet, int r2c,
		int num_gpus)
{
	poisson_plan p;
	p.r2c=r2c;
	p.nx=nx, p.ny=ny, p.nz=nz;
	p.dirichlet = dirichlet;
	p.num_gpus = num_gpus;
	return p;
}

poisson_plan poisson_create_plan(int nx, int ny, int nz,
	double dx, double dy, double dz, int dirichlet,
	int num_gpus)
{
	poisson_plan p = _init_plan(nx,ny,nz,dirichlet,0, num_gpus);

	_create_transfer_plans(p);
	double complex* diag = poisson_calcdiag(p, dx, dy, dz);
	scale_diagonal(diag, p);
	_upload_diagonal(diag, p);
	free(diag);

	return p;
}

void poisson_destroy_plan(poisson_plan p) {
	if(p.itransf_plan != p.transf_plan)
		cufftDestroy(p.itransf_plan);
	cufftDestroy(p.transf_plan);


	for(int i = 0; i < p.num_gpus; ++i){
		cudaFree(p.workbuffer[i]);
		cudaFree(p.diag[i]);
	}
}

void poisson_plan_copy_to_gpu(poisson_plan p, float* dst, float* src)
{
	assert(p.num_gpus == 1);

	size_t src_row_len, num_cols, tmp_row_len;
	if(p.r2c) {
		// for r2c fft we need to pad the real input data along the x-axis
		// by two real entries
		src_row_len = p.nz*sizeof(float);
		num_cols = p.nx*p.ny;
		tmp_row_len = (p.nz+2)*sizeof(float);
	} else {
		// for c2c fft we need to pad the real input data at each entry by
		// one real entry (for the zero imaginary part)
		src_row_len = sizeof(float);
		num_cols = p.nx*p.ny*p.nz;
		tmp_row_len = 2*sizeof(float);
		cudaMemset(dst, 0, sizeof(cufftComplex)*num_cols);
	}

	cudaMemcpy2D(dst, tmp_row_len,
			src, src_row_len,
			src_row_len, num_cols,
			cudaMemcpyHostToDevice);
}

void poisson_plan_copy_to_host(poisson_plan p, float* dst, float* src)
{
	assert(p.num_gpus == 1);

	size_t dst_row_len, num_cols, tmp_row_len;
	if(p.r2c) {
		// for r2c fft we need to remove the two real entry padding of the
		// output data
		dst_row_len = p.nz*sizeof(float);
		num_cols = p.nx*p.ny;
		tmp_row_len = (p.nz+2)*sizeof(float);
	} else {
		// for c2c fft we need to remove one entry padding (the imaginary part)
		// of the ouput data
		dst_row_len = sizeof(float);
		num_cols = p.nx*p.ny*p.nz;
		tmp_row_len = 2*sizeof(float);
	}

	cudaMemcpy2D(dst, dst_row_len,
				src, tmp_row_len,
				dst_row_len, num_cols,
				cudaMemcpyDeviceToHost);
}

__device__ cufftComplex postprozess_dst(cufftComplex a, cufftComplex a_anti) {
	return cuCsubf(a, a_anti);
}

__device__ void prepare_dst(cufftComplex new_a,
		cufftComplex* a, cufftComplex* a_anti) {
	*a_anti = make_cuFloatComplex(-new_a.x, -new_a.y);
	*a = new_a;
}



/** Beginning at second slice in x direction:
 * - postprocess the transform result
 * - divide by the diagonal entries and
 * - preprocess the data for the back-transform
 * slice by slice.
 */
__global__ void post_process_and_prepare_ext(const cufftComplex *b,
	cufftComplex *a, int n, int blocksize)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	const int numThreadsy = blockDim.y * gridDim.y;
	const int threadIDy = blockIdx.y * blockDim.y + threadIdx.y;
	for (int i = threadIDy+1; i < n; i+=numThreadsy)
	{
		cufftComplex* srcblock = &a[blocksize*i];
		cufftComplex* dstblock = &a[(n*2-i)*blocksize];
		const cufftComplex* bsrcblock = &b[blocksize*(i)];
		for(int j = threadID; j < blocksize; j+= numThreads)
		{
			cufftComplex a = postprozess_dst(srcblock[j], dstblock[j]);

			a = cuCmulf(a, bsrcblock[j]);

			prepare_dst(a, &srcblock[j], &dstblock[j]);
		}
	}
	return;
}

void _execute_kernel(poisson_plan p, cufftComplex* dst_c, cufftComplex* diag,
		size_t slice_size_c)
{
	if(p.dirichlet) {
		dim3 gridsize = dim3(4, 16, 1);
		dim3 blocksize = dim3(256, 1, 1);
		post_process_and_prepare_ext<<<gridsize, blocksize>>>(
				diag, dst_c, p.nx,
				slice_size_c);

		//since the kernel does not handle them make sure first and middle
		//slice in x direction are zero
		cudaMemset(dst_c, 0, sizeof(cufftComplex)*slice_size_c);
		cudaMemset(dst_c+slice_size_c*p.nx, 0,
				sizeof(cufftComplex)*slice_size_c);
	} else {
		elemwise_mul<<<64, 256>>>(diag, dst_c, p.nx*slice_size_c);
	}
}

void poisson_execute_plan(poisson_plan p, void* dst, void* src)
{
	assert(p.num_gpus == 1);

	size_t slice_size_c = p.r2c ? p.ny*(p.nz/2+1) : p.ny*p.nz;
	size_t slice_size_f = slice_size_c*2;
	float* src_f = (float*)src;
	cufftComplex* dst_c = (cufftComplex*)dst;

	// for dirichlet-BC init the duplicate second half of the input array with
	// zeros
	if(p.dirichlet) {
		cudaMemset(src_f+slice_size_f*p.nx, 0, sizeof(float)*slice_size_f*p.nx);
	}
	// forward transform
	if(p.r2c)
		cufftExecR2C(p.transf_plan, src_f, dst_c);
	else
		cufftExecC2C(p.transf_plan, (cufftComplex*)src_f, dst_c, CUFFT_FORWARD);

	// kernel
	_execute_kernel(p, (cufftComplex*)dst_c, (cufftComplex*)p.diag[0],
			slice_size_c);

	// backwards transform
	if(p.r2c)
		cufftExecC2R(p.itransf_plan, dst_c, (float*)dst);
	else
		cufftExecC2C(p.itransf_plan, (cufftComplex*)dst_c, dst_c,
				CUFFT_INVERSE);

}


#ifndef SINGLE_GPU
cudaLibXtDesc* poisson_mgpu_plan_allocate(poisson_plan p)
{
	assert(p.num_gpus > 1);

	cudaLibXtDesc* out;
	// in and out memory allocation
	checkResult(cufftXtMalloc(p.transf_plan, &out, CUFFT_XT_FORMAT_INPLACE));

	return out;
}

void poisson_mgpu_plan_copy_to_host(poisson_plan p, float* dst,
		cudaLibXtDesc* src)
{
	assert(p.num_gpus > 1);
	assert(!p.r2c);

	size_t hostsize = p.nx*p.ny*p.nz;
	size_t size = p.dirichlet ? 2*hostsize : hostsize;

	cufftComplex* tmp;
	check(cudaHostAlloc(&tmp, sizeof(cufftComplex)*size,0));

	checkResult(cufftXtMemcpy(p.transf_plan, tmp, src,
			CUFFT_COPY_DEVICE_TO_HOST));

	for(size_t i = 0; i < hostsize; ++i) dst[i] = tmp[i].x;
	cudaFreeHost(tmp);
}

void poisson_mgpu_plan_copy_to_gpu(poisson_plan p,
		cudaLibXtDesc* dst, float* src)
{
	assert(p.num_gpus > 1);
	assert(!p.r2c);

	size_t hostsize = p.nx*p.ny*p.nz;
	size_t size = p.dirichlet ? 2*hostsize : hostsize;

	cufftComplex* tmp;
	cudaHostAlloc(&tmp, sizeof(cufftComplex)*size,0);

	for(size_t i = 0; i < hostsize; ++i){
		tmp[i].x = src[i];
		tmp[i].y = 0.0f;
	}

	// copy data to device
	checkResult(cufftXtMemcpy(p.transf_plan, dst, tmp,
			CUFFT_COPY_HOST_TO_DEVICE));
	cudaFreeHost(tmp);
}

void poisson_mgpu_execute_plan(poisson_plan p,
		cudaLibXtDesc* dst, cudaLibXtDesc* src)
{
	// the number of devices needs to be divisible by two
	assert(p.num_gpus % 2 == 0);
	assert(!p.r2c);

	size_t slice_size_c = p.ny*p.nz;
	size_t slice_size_f = slice_size_c*2;

	// for dirichlet-BC init the duplicate second half of the input array with
	// zeros
	if(p.dirichlet) {
		// for # of gpus divisible by two the second half of the data in x
		// direction should be at the second half of gpus
		// therefore their input should be set completely to zero

		size_t floats_per_gpu = slice_size_f*p.nx*2 / p.num_gpus;

		for(int i = p.num_gpus/2; i < p.num_gpus; ++i){
			cudaSetDevice(src->descriptor->GPUs[i]);
			cudaMemset(src->descriptor->data[i], 0,
					sizeof(float)*floats_per_gpu);

		}
	}

	// forward transform
	cufftXtExecDescriptorC2C(p.transf_plan, src, dst, CUFFT_FORWARD);

	for(int i = 0; i < p.num_gpus; ++i){
		cudaSetDevice(dst->descriptor->GPUs[i]);
		_execute_kernel(p, (cufftComplex*)dst->descriptor->data[i],
				p.diag[i], slice_size_c/p.num_gpus);
	}

	// backwards transform
	cufftXtExecDescriptorC2C(p.itransf_plan, dst, dst, CUFFT_INVERSE);
}
#endif
