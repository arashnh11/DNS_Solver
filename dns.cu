#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <string>
#include <map>
#include <stdint.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <cuda_profiler_api.h>

#include "freeglut/include/GL/glut.h"

#include "HSV_RGB.h"
#include "ThrustCachedAllocator.h"

using namespace std;

const int WINDOW_WIDTH = 500;
const int WINDOW_HEIGHT = 500;

struct RenderData2D
{
    uint32_t m_width;
    std::vector<rgb> m_data;

    void Init(std::vector<float>& i_data, uint32_t i_width, float maxvalue, float minValue)
    {
        m_width = i_width;
		//float maxvalue = *std::max_element(i_data.begin(), i_data.end());
        //float minValue = *std::min_element(i_data.begin(), i_data.end());
        m_data.resize(i_data.size());
        for (int i = 0; i < i_data.size(); ++i)
        {
            hsv color;
            color.h = (i_data[i] - minValue) / (maxvalue - minValue) * 360;
            color.s = 0.7;
            color.v = 0.7;
            
            m_data[i] = (hsv2rgb(color));
        }
    }
	
	void Init(thrust::host_vector<float>& i_data, uint32_t i_width, float maxvalue, float minValue)
    {
        m_width = i_width;
		//float maxvalue = *std::max_element(i_data.begin(), i_data.end());
        //float minValue = *std::min_element(i_data.begin(), i_data.end());
        m_data.resize(i_data.size());
        for (int i = 0; i < i_data.size(); ++i)
        {
            hsv color;
			float h = -1*((i_data[i] - minValue) / (maxvalue - minValue)) * 360  + 240;
			if (h > 360) h -= 360;
			if (h < 0) h += 360;
			
			color.h = h;
            color.s = 0.7;
            color.v = 0.7;
            
            m_data[i] = (hsv2rgb(color));
        }
    }
};


RenderData2D s_drawData;

thrust::host_vector<float> u_global, v_global;
float wu_global, wv_global;

void idleCPU()
{
	//DNSCPU::RunSimulation();
	//std::vector<float>& data = DNSCPU::getU();
	//float maxvalue = *std::max_element(data.begin(), data.end());
	//float minvalue = *std::min_element(data.begin(), data.end());

 //   maxvalue = 3.0f;
 //   minvalue = -3.0f;

	//s_drawData.Init(data, DNSCPU::getUWidth(), maxvalue, minvalue);
	//glutPostRedisplay();
}

void idleGPU()
{
	//DNSGPU::RunSimulation();
	//
	//std::vector<float>& data = DNSGPU::getU();
	//if(data.size())
	//{
	//	float maxvalue = *std::max_element(data.begin(), data.end());
	//	float minvalue = *std::min_element(data.begin(), data.end());

	//	maxvalue = 3.0f;
	//	minvalue = -3.0f;

	//	s_drawData.Init(data, DNSGPU::getUWidth(), maxvalue, minvalue);
	//}
	//glutPostRedisplay();
}

void display();

// The original SOR solver lives in DoStuff();
// At the end of DoStuff we pass in the data to be drawn on screen
void DoStuff();
void DoStuffGPU();

int main(int argc, char**argv)
{ 
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("points and lines");
    glutDisplayFunc(display);
    
    
    // Uncomment this to run the original solver. 
    //DoStuff();
	DoStuffGPU();
	
    // The idle func (defined in thsi while) will run the simulation (currently its jacobi)
    //glutIdleFunc(idleGPU);
	
    //glutMainLoop();
    return 0;
} // end main

#define BLOCK_SIZE 32 // Number of threads in x and y direction - Maximum Number of threads per block = 32 * 32 = 1024

__global__ void Temperature_solver(int nx, int ny, int wu, int wv, int wT, float dx, float dy, float dt, float Re, float Pr, float *u, float *v, float *Told, float *T)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i > 0 && i < nx && j > 0 && j < ny){

		Told[i * wT + j] = T[i * wT + j];
		T[i * wT + j] = T[i * wT + j] + dt*(-0.5*(u[i * wu + j] + u[(i - 1) * wu + j])*(1.0 / (2.0*dx)*(T[(i + 1) * wT + j] - T[(i - 1) * wT + j])) - 0.5*(v[i * wv + j] + v[i * wv + j - 1])*(1.0 / (2.0*dy)*(T[i * wT + j + 1] - T[i * wT + j - 1])) + 1 / (Re*Pr)*(1 / pow(dx, 2.0f)*(T[(i + 1) * wT + j] - 2.0*T[i * wT + j] + T[(i - 1) * wT + j]) + 1 / pow(dy, 2.0f)*(T[i * wT + j + 1] - 2 * T[i * wT + j] + T[i * wT + j - 1])));
	}
	__syncthreads();
}



__global__ void PressureSolve(float * p_d, const float * p_old, float * abs_d, const float * us_d, const float * vs_d, int p_xlength, int p_ylength, int wp, int wu, int wv, float dx, float dy, float dx2, float dy2, float dt, float beta)
{

	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int j = threadIdx.y + blockDim.y*blockIdx.y;


	if (i > 0 && i < p_xlength && j > 0 && j < p_ylength)
	{
		float p_0 = p_d[i * wp + j];
		p_d[i * wp + j] = beta*(dx2*dy2 / (-2.0*(dx2 + dy2))*(-1.0 / dx2*(p_d[(i + 1) * wp + j] + p_d[(i - 1) * wp + j] + p_d[i * wp + j + 1] + p_d[i * wp + j - 1]) + 1.0 / dt*(1.0 / dx*(us_d[i * wu + j] - us_d[(i - 1) * wu + j]) + 1.0 / dy*(vs_d[i * wv + j] - vs_d[i * wv + j - 1])))) + (1.0 - beta)*p_d[i * wp + j];
		
		float error = p_d[i * wp + j] - p_0;
		abs_d[i * wp + j] = abs_d[i * wp + j]+ error*error;
		
	} // end if


} // end global


__global__ void PressureBC(float * p_d, float * p_ref, int nx, int ny, float dy, int wp)
{

	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int j = threadIdx.y + blockDim.y*blockIdx.y;

	if (i >= 0 && i < nx + 1 && j == 0){
		p_d[i * wp + j] = p_ref[i * wp + j + 1]; // bottom wall - Final
	}

	if (i >= 0 && i < nx + 1 && j == ny){
		p_d[i * wp + j] = p_ref[i * wp + j - 1]; // Upper - no flux
	}

	if (j >= 0 && j < ny + 1 && i == 0){
		p_d[i * wp + j] = p_ref[(i + 1) * wp + j]; // left wall - not the inlet - Final
	}

	if (j >= 0 && j < ny + 1 && i == nx && j*dy < 2.0){
		p_d[i * wp + j] = p_ref[(i - 1) * wp + j]; // right wall - not the outlet - Final

		// printf("POSITIVE ");
	}

	if (j >= 0 && j < ny + 1 && i == nx && j*dy >= 2.0){
		p_d[i * wp + j] = -p_ref[(i - 1) * wp + j]; // pressure outlet - static pressure is zero - Final
		// printf("NEGATIVE ");    

	}
	//__syncthreads();

}
void DoStuffGPU()
{
try
	{
		// output format

		ofstream f("result_gpu.txt"); // Solution Results
		f.setf(ios::fixed | ios::showpoint);
		f << setprecision(5);

		ofstream g("convergence_gpu.txt"); // Convergence history
		g.setf(ios::fixed | ios::showpoint);
		g << setprecision(5);
		cout.setf(ios::fixed | ios::showpoint);
		cout << setprecision(5);

		//ofstream file_p_before("p_before_gpu_BC.txt");
		//file_p_before.setf(ios::fixed | ios::showpoint);
		//file_p_before << setprecision(3); 
		//ofstream file_p_after("p_after_gpu_BC.txt");
		//file_p_after.setf(ios::fixed | ios::showpoint);
		//file_p_after << setprecision(3);

		// Input parameters 
		float Re, Pr, Fr, T_L, T_0, T_amb, dx, dy, t, eps,  beta, tf, st, counter, column, u_wind, T_R, Lx, Ly;
		Lx = 6.0; Ly = 7.0; // Domain dimensions
		int ni = 12.0; // Number of nodes per unit length in x direction
		int nj = 12.0; // Number of nodes per unit length in y direction
		int nx = Lx * ni; int ny = Ly * nj; // Number of Nodes in each direction
		int maxiter;
		u_wind = 1; // Reference velocity
		st = 0.00005 * 2; // Total variance criteria
		eps = 0.001f; // Pressure convergence criteria (epsilon squared)
		tf = 100.01; // Final time step
		Pr = 0.5*(0.709 + 0.711); // Prandtl number
		Re = 250.0; Fr = 0.3; // Non-dimensional numbers for inflow conditions
		//dx = (float)1/ni; dy = (float)1/nj; // dx and dy
		dx = Lx / (nx - 1); dy = Ly / (ny - 1);
		beta = 0.8f; // Successive over relaxation factor (SOR)
		t = 0; // Initial time step
		T_L = 100.0; // Left wall temperature (C)
		T_R = 50.0; // Right wall temperature (C)
		T_amb = 25.0; // Ambient air temperature (C)
		T_0 = 50.0; // Initial air temperature
		T_L = T_L + 273.15; T_0 = T_0 + 273.15; T_amb = T_amb + 273.15; T_R = T_R + 273.15;// Unit conversion to (K)
		maxiter = 1000; // Maximum iteration at each time step
		counter = 0; // initial row for output monitoring
		column = 1; // Column number for output display

		// Records number of clicks a step takes
		std::map<string, uint32_t> stepTimingAccumulator;

		// Host Vectors

		thrust::host_vector<float> u(nx * (ny + 1));
		thrust::host_vector<float> us(nx*(ny + 1));
		thrust::host_vector<float> uold(nx * (ny + 1));
		int wu = ny + 1;

		thrust::host_vector<float> v((nx + 1) * ny);
		thrust::host_vector<float> vs((nx + 1) * ny);
		thrust::host_vector<float> vold((nx + 1) * ny);
		int wv = ny;

		thrust::host_vector<float> p((nx + 1) * (ny + 1));
		//   thrust::host_vector<float> abs((nx + 1) * (ny + 1));
		int wp = ny + 1;


		thrust::host_vector<float> T((nx + 1) * (ny + 1));
		int wT = ny + 1;

		thrust::host_vector<float> Told((nx + 1) * (ny + 1));
		thrust::host_vector<float> om(nx * ny);
		thrust::host_vector<float> vc(nx * ny);
		thrust::host_vector<float> uc(nx * ny);
		thrust::host_vector<float> pc(nx * ny);
		thrust::host_vector<float> Tc(nx*ny);
		// thrust::host_vector<float> abs_h((nx+1) * (ny + 1));
		int wc = ny;

		cudaProfilerStart();
		cudaFree(0);
		thrust::device_vector<float> us_d(nx*(ny + 1));
		thrust::device_vector<float> vs_d((nx + 1) * ny);
		thrust::device_vector<float> p_d((nx + 1) * (ny + 1));
		thrust::device_vector<float> p_old((nx + 1) * (ny + 1));
		thrust::device_vector<float> p_ref((nx + 1) * (ny + 1));
		thrust::device_vector<float> abs_d((nx + 1) * (ny + 1));

		cached_allocator alloc;

		// Time step size stability criterion

		float mt1 = 0.25*pow(dx, 2.0) / (1.0 / Re); float Rer = 1.0 / Re; float mt2 = 0.25*pow(dy, 2.0) / (1.0 / Re);
		float dt;

		if (mt1 > Rer)
		{
			dt = Rer;
		}
		else
		{
			dt = mt1;
		}

		if (dt > mt2)
		{
			dt = mt2;
		}

		float start_clock = clock();

		//......................................................................................
		// Step 0 - It can be parallelized
		// Initializing the flow variable (Temperature)  
		// Boundary conditions for T (Initialization)
		int step0_start = clock();
		for (int i = 0; i < nx + 1; i++)
		{
			for (int j = 0; j < ny + 1; j++)
			{
				T[i * wT + j] = T_0 / T_amb;
			} // end for j
		} // end for i
		//......................................................................................
		int step0_end = clock();
		stepTimingAccumulator["Step 0, Initializing Temperature"] += step0_end - step0_start;
		//......................................................................................

		// Marching in Time - Outermost loop

		while (t <= tf)
		{

			int iter = 0;

			int stepi1_start = clock();
			//........................................................................................
			// Step i1 - it can be parallelized 
			// boundary conditions for u velocity

			for (int i = 0; i < nx; i++)
			{
				for (int j = 0; j < ny + 1; j++)
				{
					if (i == 0 && j > 0 && j < ny)
					{
						if (j*dy < 2.0)
						{
							u[i * wu + j] = 0; // left wall - Final
						}
						else
						{
							u[i * wu + j] = u_wind; // left inlet - Final
						}
					}
					else if (i == nx - 1 && j>0 && j < ny)
					{
						if (j*dy < 2.0)
						{
							u[i * wu + j] = 0; // Right wall has 0 horizontal velocity - Final
						}
						else
						{
							u[i * wu + j] = u[(i - 1) * wu + j]; // right outlet - no velocity change
						}
					}
					else if (j == 0)
					{
						u[i * wu + j] = -u[i * wu + j + 1]; // bottom ghost - Final
					}
					else if (j == ny)
					{
						u[i * wu + j] = u[i * wu + j - 1]; // upper ghost - Final
					}
				} // end for j
			} // end for i
			int stepi1_end = clock();
			stepTimingAccumulator["Step i1 - Set Horizontal Velocity Boundary Conditions"] += stepi1_end - stepi1_start;
			//...............................................................................................


			//.........................................................................................
			// Step i2 - it can be parallelized
			// boundary conditions for v velocity
			int stepi2_start = clock();

			for (int i = 0; i < nx + 1; i++)
			{
				for (int j = 0; j < ny; j++)
				{
					if (j == 0 && i > 0 && i < nx)
					{
						v[i * wv + j] = 0; // bottom wall - Final
					}
					else if (j == ny - 1 && i > 0 && i < nx)
					{
						v[i * wv + j] = v[i * wv + j - 1]; // upper wall - Final
					}
					else if (i == 0)
					{
						v[i * wv + j] = -v[(i + 1) * wv + j]; // left ghost (Left Wall and inlet has 0 vertical velocity) - Final
					}
					else if (i == nx)
					{
						if (j*dy < 2.0)
						{
							v[i * wv + j] = -v[(i - 1) * wv + j]; // right ghost (Right wall has 0 vertical velocity) - Final
						}
						else
						{
							v[i * wv + j] = v[(i - 1) * wv + j]; // right outlet - no velocity gradient
						}
					}
				} // end for j
			} // end for I
			int stepi2_end = clock();
			stepTimingAccumulator["Step i2 - Set Vertical Velocity Boundary Conditions"] += stepi2_end - stepi2_start;
			//...............................................................................................

			//...............................................................................................
			int step1_start = clock();
			//.........................................................................................
			// Step 1 - it can be parallelized - Solve for intermediate velocity values

			// u - us - vh - a 

			for (int i = 1; i < nx - 1; i++)
			{
				for (int j = 1; j < ny; j++)
				{
					float vh = 1.0 / 4.0*(v[i * wv + j] + v[(i + 1) * wv + j] + v[i * wv + j - 1] + v[(i + 1) * wv + j - 1]); // v hat
					float a = u[i * wu + j] * 1.0 / (2.0*dx)*(u[(i + 1) * wu + j] - u[(i - 1) * wu + j]) + vh*1.0 / (2.0*dy)*(u[i * wu + j + 1] - u[i * wu + j - 1]); // a
					us[i * wu + j] = dt / Re*(1.0 / pow(dx, 2.0)*(u[(i + 1) * wu + j] - 2.0*u[i * wu + j] + u[(i - 1) * wu + j]) + 1.0 / pow(dy, 2.0)*(u[i * wu + j + 1] - 2.0*u[i * wu + j] + u[i * wu + j - 1])) - a*dt + u[i * wu + j]; // u star
				} // end for j
			} // end for i

			//..........................................................................................
			// Step 1 - it can be parallelized
			// v - vs - uh - b
			for (int i = 1; i < nx; i++)
			{
				for (int j = 1; j < ny - 1; j++)
				{
					float uh = 1.0 / 4.0*(u[i * wu + j] + u[i * wu + j + 1] + u[(i - 1) * wu + j] + u[(i - 1) * wu + j + 1]);
					float b = uh*1.0 / (2.0*dx)*(v[(i + 1) * wv + j] - v[(i - 1) * wv + j]) + v[i * wv + j] * 1.0 / (2.0*dy)*(v[i * wv + j + 1] - v[i * wv + j - 1]); // b
					vs[i * wv + j] = dt / Re*(1.0 / pow(dx, 2.0)*(v[(i + 1) * wv + j] - 2.0*v[i * wv + j] + v[(i - 1) * wv + j]) + 1.0 / pow(dy, 2.0)*(v[i * wv + j + 1] - 2.0*v[i * wv + j] + v[i * wv + j - 1])) + dt / pow(Fr, 2.0)*(0.5*(T[i * wT + j] + T[i * wT + j - 1]) - 1) / (0.5*(T[i * wT + j] + T[i * wT + j - 1])) - b*dt + v[i * wv + j]; // v 
				} // end for j
			} // end for i

			//...........................................................................................
			// vs and us on Boundary conditions

			for (int i = 0; i < nx; i++)
			{
				us[i * wu + 0] = -us[i * wu + 1]; // bottom ghost - Final
			} // end for j

			//...........................................................................................
			for (int j = 0; j < ny + 1; j++)
			{
				if (j*dy < 2.0)
				{
					us[0 * wu + j] = 0; // left wall - FInal
					us[(nx - 1) * wu + j] = 0; // right wall - Final
				}
				else
				{
					us[0 * wu + j] = u_wind; // left inlet - Final
				}
			}
			//...........................................................................................

			for (int j = 0; j < ny; j++)
			{
				vs[0 * wv + j] = -vs[1 * wv + j]; // left ghost (Both wall and inlet have 0 vs) - Final
				if (j*dy < 2.0)
				{
					vs[nx * wv + j] = -vs[(nx - 1) * wv + j]; // right ghost (Only the right wall - Final
				}
				else
				{
					vs[nx * wv + j] = vs[(nx - 1) * wv + j]; // right outlet - no flux
				}
			}
			//............................................................................................

			for (int i = 0; i < nx + 1; i++)
			{
				vs[i * wv + 0] = 0; // Bottom wall - Final
			} // end for i
			//............................................................................................

			int step1_end = clock();
			stepTimingAccumulator["Step 1 - Solve for intermediate velocities"] += step1_end - step1_start;


			//...............................................................................................
			// Step 2 - Parallel GPU version
			// Poisson equation for pressure

			int step2_start = clock();

			// Cuda set up
			int p_xlength = nx;
			int p_ylength = ny;

			float *ptr_us = thrust::raw_pointer_cast(&us_d[0]);
			float *ptr_vs = thrust::raw_pointer_cast(&vs_d[0]);
			float *ptr_p = thrust::raw_pointer_cast(&p_d[0]);
			float *ptr_p_old = thrust::raw_pointer_cast(&p_old[0]);
			float *ptr_abs = thrust::raw_pointer_cast(&abs_d[0]);
			float *ptr_p_ref = thrust::raw_pointer_cast(&p_ref[0]);

			iter = 0;
			// float diffp = 0;
			us_d = us;
			vs_d = vs;
			cout << "Time:" <<t;
	
			while (iter < maxiter){
				// SOR pressure solver
				PressureBC << < dim3((ny + 1) / BLOCK_SIZE + 1, (nx + 1) / BLOCK_SIZE + 1, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1) >> >(ptr_p, ptr_p, nx, ny, dy, wp);

				PressureSolve<<< dim3( (ny+1)/BLOCK_SIZE + 1, (nx+1)/BLOCK_SIZE + 1, 1) , dim3(BLOCK_SIZE,BLOCK_SIZE,1)>>>(ptr_p, ptr_p_old, ptr_abs, ptr_us, ptr_vs, p_xlength, p_ylength, wp, wu, wv, dx, dy, dx*dx, dy*dy, dt, beta);
				
				iter = iter + 1;
	
			} // end while eps
			PressureBC << < dim3((ny + 1) / BLOCK_SIZE + 1, (nx + 1) / BLOCK_SIZE + 1, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1) >> >(ptr_p, ptr_p, nx, ny, dy, wp);
			p = p_d;

			// Compute the total error over all iterations
			float error = sqrt(thrust::reduce(thrust::cuda::par(alloc), abs_d.begin(), abs_d.end()) / maxiter);
			// Reset the error vector for next time
			thrust::fill(abs_d.begin(), abs_d.end(), 0.0f);

			// Adjust the maxiter based on the total error last time. 
			float errorRatio = error/eps;
			if(error < eps)
			{
				maxiter = maxiter * 0.95 > maxiter - 1 ? maxiter - 1 : maxiter * 0.95;
				if (maxiter < 10)
					maxiter = 10;
				//std::cout<< "\nDecreasing maxiter to: " << maxiter << endl;
			}
			else if(error > eps && maxiter < 1000)
			{
				//std::cout<< "\nIncreasing maxiter to: " << maxiter << endl;
				maxiter = maxiter * 1.05 < maxiter + 1 ? maxiter + 1 : maxiter * 1.05;
			}


			
			int step2_end = clock();

			std::cout << "\tIters:" << iter << "\t Error:" << error << "\tPressure loop time:" << step2_end - step2_start;

			stepTimingAccumulator["Step 2 - Solve for pressure until tolerance or max iterations"] += step2_end - step2_start;


			//.................................................................................................
			// Step 3 - It can be parallelized 
			// velocity update - projection method
			int step3_start = clock();

			// u

			for (int i = 1; i < nx - 1; i++)
			{
				for (int j = 1; j < ny; j++)
				{
					uold[i * wu + j] = u[i * wu + j];
					u[i * wu + j] = us[i * wu + j] - dt / dx*(p[(i + 1) * wp + j] - p[i * wp + j]);
				} // end for j
			} // end for i
			//................................................

			// v

			for (int i = 1; i < nx; i++)
			{
				for (int j = 1; j < ny - 1; j++)
				{
					vold[i * wv + j] = v[i * wv + j];
					v[i * wv + j] = vs[i * wv + j] - dt / dy*(p[i * wp + j + 1] - p[i * wp + j]);
				} // end for j
			} // end for i
			int step3_end = clock();
			stepTimingAccumulator["Step 3 - Velocity Update"] += step3_end - step3_start;
			//...............................................................................................

			// TODO: Is this CUDA implementation even faster than CPU? (CHECK)
			//...............................................................................................
			// Step 4 - It can be parallelized
			// Solving for temperature
			int step4_start = clock();

			thrust::device_vector<float> d_T = T;
			thrust::device_vector<float> d_Told = Told;
			thrust::device_vector<float> d_u = u;
			thrust::device_vector<float> d_v = v;

			int gridsize_x = nx/BLOCK_SIZE + 1;
			int gridsize_y = ny/BLOCK_SIZE + 1;

			dim3 dimgrid(gridsize_x, gridsize_y, 1); // The grid has #gridsize blocks in x and 1 block in y and 1 block in z direction
			dim3 dimblock(BLOCK_SIZE, BLOCK_SIZE, 1);

			float *ptr_u = thrust::raw_pointer_cast(&d_u[0]);
			float *ptr_v = thrust::raw_pointer_cast(&d_v[0]);
			float *ptr_T = thrust::raw_pointer_cast(&d_T[0]);
			float *ptr_Told = thrust::raw_pointer_cast(&d_Told[0]);

			Temperature_solver<<<dimgrid, dimblock>>>(nx, ny, wu, wv, wT, dx, dy, dt, Re, Pr, ptr_u, ptr_v, ptr_Told, ptr_T);

			thrust::copy(d_Told.begin(), d_Told.end(), Told.begin());
			thrust::copy(d_T.begin(), d_T.end(), T.begin());

			int step4_end = clock();
			stepTimingAccumulator["Step 4 - Solving for temperature"] += step4_end - step4_start;
			//................................................................................................

			//...............................................................................................
			// Step i3 - Initializing boundary conditions for temperature 
			// boundary conditions for Temperature
			int stepi3_start = clock();

			for (int i = 0; i < nx + 1; i++)
			{
				for (int j = 0; j < ny + 1; j++)
				{
					if (j == 0)
					{
						T[i * wT + j] = T[i * wT + j + 1]; // bottom wall - Insulated - no flux - Final
					}
					else if (j == ny)
					{
						T[i * wT + j] = 2.0*(T_0) / T_amb - T[i * wT + j - 1]; // upper boundary - lid with ambient temperature (as air) - Final
					}
					else if (i == 0)
					{
						if (j*dy < 2.0)
						{
							T[i * wT + j] = 2.0*T_L / T_amb - T[(i + 1) * wT + j]; // left wall at T_L - Constant Temperature - Final
						}
						else
						{
							T[i * wT + j] = 2.0*T_0 / T_amb - T[(i + 1) * wT + j]; // left inlet at T_0 (initial temperature) - Final
						}
					}
					else if (i == nx)
					{
						if (j*dy < 2.0)
						{
							T[i * wT + j] = 2.0*T_R / T_amb - T[(i - 1) * wT + j]; // right wall at T_R - Final
						}
					}
				} // end for j
			} // end for i
			int stepi3_end = clock();
			stepTimingAccumulator["Step i3 - Initializing boundary conditions for temperature"] += stepi3_end - stepi3_start;
			//...............................................................................................

			//...............................................................................................
			// Step 5 - Checking if solution reached steady state
			// Checking the steady state condition
			int step5_start = clock();

			float TV, diffv; TV = 0;
			for (int i = 1; i < nx - 1; i++)
			{
				for (int j = 1; j < ny - 2; j++)
				{
					diffv = v[i * wv + j] - vold[i * wv + j];
					TV = TV + pow(pow(diffv, 2), 0.5);
				} // end for i
			} // end for j

			TV = TV / ((nx - 1)*(ny - 2));

			if (TV < st && error < eps)
			{
				cout << "Steady state time = " << t << " (s) " << endl;
				break;
			}
			counter = counter + 1;
			if (fmod(counter, 10) == 0 || counter == 1)
			{
				//cout << "" << endl;
				//cout << "Column" << setw(30) << "time(s)" << setw(30) << "Iterations on Pressure" << setw(30) << "Pressure Residual" << setw(30) << "Total Variance" << endl;
			} // end if
			int step5_end = clock();
			stepTimingAccumulator["Step 5 - Check for steady state"] += step5_end - step5_start;
			//...............................................................................................


			//cout << column << setw(30) << t << setw(30) << iter << setw(30) << error << setw(30) << TV << endl;
			g << column << setw(30) << t << setw(30) << iter << setw(30) << error << setw(30) << TV << endl;
			t = t + dt;
			column = column + 1;
			int renderStartTime = clock();

			s_drawData.Init(T, wT, (100 + 273.15) / (25 + 273.15), (50 + 273.15) / (25 + 273.15));//*(std::max_element(T.begin(), T.end())), *(std::min_element(T.begin(), T.end())));

			u_global = u; wu_global = wu;
			v_global = v; wv_global = wv;
			display();

			int renderEndTime = clock();
			stepTimingAccumulator["Render time"] += renderEndTime - renderStartTime;
			cout << "\tRenderTime: " << (renderEndTime - renderStartTime );


			cout << endl;
		} // end while time
		cudaProfilerStop();
		//........................................................................................................

		// Step 6
		// Co-locate the staggered grid points 
		int step6_start = clock();
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				vc[i * wc + j] = 1.0 / 2.0*(v[(i + 1) * wv + j] + v[i * wv + j]);
				pc[i * wc + j] = 1.0 / 4.0*(p[i * wp + j] + p[(i + 1) * wp + j] + p[i * wp + j + 1] + p[(i + 1) * wp + j + 1]);
				uc[i * wc + j] = 1.0 / 2.0*(u[i*wu + j] + u[i * wu + j + 1]);
				om[i * wc + j] = 1.0 / dx*(v[(i + 1) * wv + j] - v[i * wv + j]) - 1.0 / dy*(u[i * wu + j + 1] - u[i * wu + j]);
				Tc[i * wc + j] = 1.0 / 4.0*(T[i * wT + j] + T[(i + 1) * wT + j] + T[i * wT + j + 1] + T[(i + 1) * wT + j + 1]);
			} // end for j
		} // end for i
		//........................................................................................................
		int step6_end = clock();
		stepTimingAccumulator["Step 6 - Co-locate staggered grid points"] += step6_end - step6_start;

		// Steady state results

		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				f << setw(15) << t - dt << setw(15) << i*dx << setw(15) << j*dy << setw(15) << uc[i * wc + j] << setw(15) << vc[i * wc + j] << setw(15) << pc[i * wc + j] << setw(15) << Tc[i * ny + j] * T_amb - 273.15 << setw(15) << om[i * wc + j] << endl;
			} // end for i
		} // end for j
		//.........................................................................................................

		float end_clock = clock();
		cout << "CPU time = " << (end_clock - start_clock) / CLOCKS_PER_SEC << " (s)" << endl;
		//cout << "Re = " << Re << endl;
		//cout << "Fr = " << Fr << endl;

		for (auto it = stepTimingAccumulator.begin(); it != stepTimingAccumulator.end(); it++)
		{
			float seconds = (float)it->second / CLOCKS_PER_SEC;
			std::cout << it->first << "\t" << seconds << endl;
		}
	}
	catch(thrust::system_error e)
	{
		std::cerr <<  e.what() << std::endl;
	}

}
void DoStuff()
{
    // output format
    
    ofstream f("result_cpu.txt"); // Solution Results
    f.setf(ios::fixed | ios::showpoint);
    f << setprecision(5);

    ofstream g("convergence_cpu.txt"); // Convergence history
    g.setf(ios::fixed | ios::showpoint);
    g << setprecision(5);
    cout.setf(ios::fixed | ios::showpoint);
    cout << setprecision(5);

    // Input parameters 
	float Lx = 4.0, Ly = 5.0; // Domain dimensions
	int ni = 18; // Number of nodes per unit length in x direction
	int nj = 18; // Number of nodes per unit length in y direction
	int nx = Lx * ni;
	int ny = Ly * nj; // Number of Nodes in each direction
	float u_wind = 1; // Reference velocity
	float st = 0.00005; // Total variance criteria
	float eps = 0.001; // Pressure convergence criteria
	float tf = 100; // Final time step
	float Pr = 0.5*(0.709 + 0.711); // Prandtl number
	float Re = 300.0; float Fr = 0.3; // Non-dimensional numbers for inflow conditions
	float dx = Lx / (nx - 1);
	float dy = Ly / (ny - 1); // dx and dy
	dx = dy;
	float beta = 0.8f; // Successive over relaxation factor (SOR)
	float t = 0; // Initial time step
	float T_L = 100.0; // Left wall temperature (C)
	float T_R = 50.0; // Right wall temperature (C)
	float T_amb = 25.0; // Ambient air temperature (C)
	float T_0 = 50.0; // Initial air temperature
	T_L = T_L + 273.15; T_0 = T_0 + 273.15; T_amb = T_amb + 273.15; T_R = T_R + 273.15;// Unit conversion to (K)
	int maxiter = 300; // Maximum iteration at each time step
	int counter = 0; // initial row for output monitoring
	int column = 1; // Column number for output display
					// Records number of clicks a step takes

                // Records number of clicks a step takes
    std::map<string, uint32_t> stepTimingAccumulator;

    // Vectors

    vector<float> u(nx * (ny + 1), 0);
    vector<float> us(nx*(ny + 1), 0);
    vector<float> uold(nx * (ny + 1), 0);
    float wu = ny + 1;

    vector<float> v((nx + 1) * ny, 0);
    vector<float> vs((nx + 1) * ny, 0);
    vector<float> vold((nx + 1) * ny, 0);
    float wv = ny;

    vector<float> p((nx + 1) * (ny + 1), 0);
    float wp = ny + 1;

    vector<float> T((nx + 1) * (ny + 1), T_0 / T_amb);     // Initializing the flow variable (Temperature)  
                                                           // Boundary conditions for T (Initialization)
    float wT = ny + 1;

    vector<float> Told((nx + 1) * (ny + 1), 0);
    vector<float> om(nx * ny, 0);
    vector<float> vc(nx * ny, 0);
    vector<float> uc(nx * ny, 0);
    vector<float> pc(nx * ny, 0);
    vector<float> Tc(nx*ny, 0);
    float wc = ny;

    // Time step size stability criterion

    float mt1 = 0.25*pow(dx, 2.0) / (1.0 / Re); float Rer = 1.0 / Re; float mt2 = 0.25*pow(dy, 2.0) / (1.0 / Re);
    float dt;

    if (mt1 > Rer)
    {
        dt = Rer;
    }
    else
    {
        dt = mt1;
    }

    if (dt > mt2)
    {
        dt = mt2;
    }

	float start_clock = clock();
    //......................................................................................
    // Step 0 - It can be parallelized

    int step0_start = clock();
    //......................................................................................
    int step0_end = clock();
    stepTimingAccumulator["Step 0, Initializing Temperature"] += step0_end - step0_start;
    //......................................................................................

    // Marching in Time - Outermost loop

    while (t <= tf)
    {

        int iter = 0;

        int stepi1_start = clock();
        //........................................................................................
        // Step i1 - it can be parallelized 
        // boundary conditions for u velocity

        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny + 1; j++)
            {
                if (i == 0 && j > 0 && j < ny)
                {
                    if (j*dy < 2.0)
                    {
                        u[i * wu + j] = 0; // left wall - Final
                    }
                    else
                    {
                        u[i * wu + j] = u_wind; // left inlet - Final
                    }
                }
                else if (i == nx - 1 && j>0 && j < ny)
                {
                    if (j*dy < 2.0)
                    {
                        u[i * wu + j] = 0; // Right wall has 0 horizontal velocity - Final
                    }
                    else
                    {
                        u[i * wu + j] = u[(i - 1) * wu + j]; // right outlet - no velocity change
                    }
                }
                else if (j == 0)
                {
                    u[i * wu + j] = -u[i * wu + j + 1]; // bottom ghost - Final
                }
                else if (j == ny)
                {
                    u[i * wu + j] = u[i * wu + j - 1]; // upper ghost - Final
                }
            } // end for j
        } // end for i
        int stepi1_end = clock();
        stepTimingAccumulator["Step i1 - Set Horizontal Velocity Boundary Conditions"] += stepi1_end - stepi1_start;
        //...............................................................................................


        //.........................................................................................
        // Step i2 - it can be parallelized
        // boundary conditions for v velocity
        int stepi2_start = clock();

        for (int i = 0; i < nx + 1; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                if (j == 0 && i > 0 && i < nx)
                {
                    v[i * wv + j] = 0; // bottom wall - Final
                }
                else if (j == ny - 1 && i > 0 && i < nx)
                {
                    v[i * wv + j] = v[i * wv + j - 1]; // upper wall - Final
                }
                else if (i == 0)
                {
                    v[i * wv + j] = -v[(i + 1) * wv + j]; // left ghost (Left Wall and inlet has 0 vertical velocity) - Final
                }
                else if (i == nx)
                {
                    if (j*dy < 2.0)
                    {
                        v[i * wv + j] = -v[(i - 1) * wv + j]; // right ghost (Right wall has 0 vertical velocity) - Final
                    }
                    else
                    {
                        v[i * wv + j] = v[(i - 1) * wv + j]; // right outlet - no velocity gradient
                    }
                }
            } // end for j
        } // end for I
        int stepi2_end = clock();
        stepTimingAccumulator["Step i2 - Set Vertical Velocity Boundary Conditions"] += stepi2_end - stepi2_start;
        //...............................................................................................

        //...............................................................................................
        int step1_start = clock();
        //.........................................................................................
        // Step 1 - it can be parallelized - Solve for intermediate velocity values

        // u - us - vh - a 

        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny; j++)
            {
                float vh = 1.0 / 4.0*(v[i * wv + j] + v[(i + 1) * wv + j] + v[i * wv + j - 1] + v[(i + 1) * wv + j - 1]); // v hat
                float a = u[i * wu + j] * 1.0 / (2.0*dx)*(u[(i + 1) * wu + j] - u[(i - 1) * wu + j]) + vh*1.0 / (2.0*dy)*(u[i * wu + j + 1] - u[i * wu + j - 1]); // a
                us[i * wu + j] = dt / Re*(1.0 / pow(dx, 2.0)*(u[(i + 1) * wu + j] - 2.0*u[i * wu + j] + u[(i - 1) * wu + j]) + 1.0 / pow(dy, 2.0)*(u[i * wu + j + 1] - 2.0*u[i * wu + j] + u[i * wu + j - 1])) - a*dt + u[i * wu + j]; // u star
            } // end for j
        } // end for i

          //..........................................................................................
          // Step 1 - it can be parallelized
          // v - vs - uh - b
        for (int i = 1; i < nx; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                float uh = 1.0 / 4.0*(u[i * wu + j] + u[i * wu + j + 1] + u[(i - 1) * wu + j] + u[(i - 1) * wu + j + 1]);
                float b = uh*1.0 / (2.0*dx)*(v[(i + 1) * wv + j] - v[(i - 1) * wv + j]) + v[i * wv + j] * 1.0 / (2.0*dy)*(v[i * wv + j + 1] - v[i * wv + j - 1]); // b
                vs[i * wv + j] = dt / Re*(1.0 / pow(dx, 2.0)*(v[(i + 1) * wv + j] - 2.0*v[i * wv + j] + v[(i - 1) * wv + j]) + 1.0 / pow(dy, 2.0)*(v[i * wv + j + 1] - 2.0*v[i * wv + j] + v[i * wv + j - 1])) + dt / pow(Fr, 2.0)*(0.5*(T[i * wT + j] + T[i * wT + j - 1]) - 1) / (0.5*(T[i * wT + j] + T[i * wT + j - 1])) - b*dt + v[i * wv + j]; // v 
            } // end for j
        } // end for i

          //...........................................................................................
          // vs and us on Boundary conditions

        for (int i = 0; i < nx; i++)
        {
            us[i * wu + 0] = -us[i * wu + 1]; // bottom ghost - Final
        } // end for j

          //...........................................................................................
        for (int j = 0; j < ny + 1; j++)
        {
            if (j*dy < 2.0)
            {
                us[0 * wu + j] = 0; // left wall - FInal
                us[(nx - 1) * wu + j] = 0; // right wall - Final
            }
            else
            {
                us[0 * wu + j] = u_wind; // left inlet - Final
            }
        }
        //...........................................................................................

        for (int j = 0; j < ny; j++)
        {
            vs[0 * wv + j] = -vs[1 * wv + j]; // left ghost (Both wall and inlet have 0 vs) - Final
            if (j*dy < 2.0)
            {
                vs[nx * wv + j] = -vs[(nx - 1) * wv + j]; // right ghost (Only the right wall - Final
            }
            else
            {
                vs[nx * wv + j] = vs[(nx - 1) * wv + j]; // right outlet - no flux
            }
        }
        //............................................................................................

        for (int i = 0; i < nx + 1; i++)
        {
            vs[i * wv + 0] = 0; // Bottom wall - Final
        } // end for i
          //............................................................................................

        int step1_end = clock();
        stepTimingAccumulator["Step 1 - Solve for intermediate velocities"] += step1_end - step1_start;

        //...............................................................................................
        // Step 2 - It can be parallelized 
        // This is the most expensive part of the code
        // Poisson equation for pressure
        int step2_start = clock();

        float error = 1; iter = 0;
        float diffp, pold;
		cout << "Time:"<<t;
        // Solve for pressure iteratively until it converges - Using Gauss Seidel SOR 
        while (error > eps)
        {
            error = 0;

            //............................................................................................
            for (int i = 1; i < nx; i++)
            {
                for (int j = 1; j < ny; j++)
                {
                    pold = p[i * wp + j];
                    p[i * wp + j] = beta*pow(dx, 2.0)*pow(dy, 2.0) / (-2.0*(pow(dx, 2.0) + pow(dy, 2.0)))*(-1.0 / pow(dx, 2.0)*(p[(i + 1) * wp + j] + p[(i - 1) * wp + j] + p[i * wp + j + 1] + p[i * wp + j - 1]) + 1.0 / dt*(1.0 / dx*(us[i * wu + j] - us[(i - 1) * wu + j]) + 1.0 / dy*(vs[i * wv + j] - vs[i * wv + j - 1]))) + (1.0 - beta)*p[i * wp + j];
                    diffp = pow((p[i * wp + j] - pold), 2.0);
                    error = error + diffp;
                } // end for j
            } // end for i
              //............................................................................................
              // boundary conditions for pressure

            for (int i = 0; i < nx + 1; i++)
            {
                for (int j = 0; j < ny + 1; j++)
                {
                    if (j == 0)
                    {
                        p[i * wp + j] = p[i * wp + j + 1]; // bottom wall - Final
                    }
                    else if (j == ny)
                    {
                        p[i * wp + j] = p[i * wp + j - 1]; // Upper - no flux
                    }
                    else if (i == 0)
                    {
                        if (j*dy < 2.0)
                        {
                            p[i * wp + j] = p[(i + 1) * wp + j]; // left wall - not the inlet - Final
                        }
                        else
                        {
                            p[i * wp + j] = p[(i + 1) * wp + j];
                        }
                    }
                    else if (i == nx)
                    {
                        if (j*dy < 2.0)
                        {
                            p[i * wp + j] = p[(i - 1) * wp + j]; // right wall - not the outlet - Final
                        }
                        else
                        {
                            p[i * wp + j] = -p[(i - 1) * wp + j]; // pressure outlet - static pressure is zero - Final
                        }
                    }
                } // end for j
            } // end for i
              //................................................................................................

            error = pow(error, 0.5);
            iter = iter + 1;
            if (iter > maxiter)
            {
                break;
            }

        } // end while eps
		
        int step2_end = clock();
		std::cout << "\t Iters:" << iter << "\t Err:" << error << "\tPressure loop time:" << step2_end - step1_end << endl;
        stepTimingAccumulator["Step 2 - Solve for pressure until tolerance or max iterations"] += step2_end - step2_start;
        //...............................................................................................

        //.................................................................................................
        // Step 3 - It can be parallelized 
        // velocity update - projection method
        int step3_start = clock();

        // u

        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny; j++)
            {
                uold[i * wu + j] = u[i * wu + j];
                u[i * wu + j] = us[i * wu + j] - dt / dx*(p[(i + 1) * wp + j] - p[i * wp + j]);
            } // end for j
        } // end for i
          //................................................

          // v

        for (int i = 1; i < nx; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                vold[i * wv + j] = v[i * wv + j];
                v[i * wv + j] = vs[i * wv + j] - dt / dy*(p[i * wp + j + 1] - p[i * wp + j]);
            } // end for j
        } // end for i
        int step3_end = clock();
        stepTimingAccumulator["Step 3 - Velocity Update"] += step3_end - step3_start;
        //...............................................................................................

        //...............................................................................................
        // Step 4 - It can be parallelized
        // Solving for temperature
        int step4_start = clock();
        Told = T;
        for (int i = 1; i < nx; i++)
        {
            for (int j = 1; j < ny; j++)
            {

                T[i * wT + j] = Told[i * wT + j] + dt*(-0.5*(u[i * wu + j] + u[(i - 1) * wu + j])*(1.0 / (2.0*dx)*(Told[(i + 1) * wT + j] - Told[(i - 1) * wT + j])) - 0.5*(v[i * wv + j] + v[i * wv + j - 1])*(1.0 / (2.0*dy)*(Told[i * wT + j + 1] - Told[i * wT + j - 1])) + 1 / (Re*Pr)*(1 / pow(dx, 2.0)*(Told[(i + 1) * wT + j] - 2.0*Told[i * wT + j] + Told[(i - 1) * wT + j]) + 1 / pow(dy, 2.0)*(Told[i * wT + j + 1] - 2 * Told[i * wT + j] + Told[i * wT + j - 1])));
            } // end for j
        } // end for i

        int step4_end = clock();
        stepTimingAccumulator["Step 4 - Solving for temperature"] += step4_end - step4_start;
        //................................................................................................

        //...............................................................................................
        // Step i3 - Initializing boundary conditions for temperature 
        // boundary conditions for Temperature
        int stepi3_start = clock();

        for (int i = 0; i < nx + 1; i++)
        {
            for (int j = 0; j < ny + 1; j++)
            {
                if (j == 0)
                {
                    T[i * wT + j] = T[i * wT + j + 1]; // bottom wall - Insulated - no flux - Final
                }
                else if (j == ny)
                {
                    T[i * wT + j] = 2.0*(T_0) / T_amb - T[i * wT + j - 1]; // upper boundary - lid with ambient temperature (as air) - Final
                }
                else if (i == 0)
                {
                    if (j*dy < 2.0)
                    {
                        T[i * wT + j] = 2.0*T_L / T_amb - T[(i + 1) * wT + j]; // left wall at T_L - Constant Temperature - Final
                    }
                    else
                    {
                        T[i * wT + j] = 2.0*T_0 / T_amb - T[(i + 1) * wT + j]; // left inlet at T_0 (initial temperature) - Final
                    }
                }
                else if (i == nx)
                {
                    if (j*dy < 2.0)
                    {
                        T[i * wT + j] = 2.0*T_R / T_amb - T[(i - 1) * wT + j]; // right wall at T_R - Final
                    }
                }
            } // end for j
        } // end for i
        int stepi3_end = clock();
        stepTimingAccumulator["Step i3 - Initializing boundary conditions for temperature"] += stepi3_end - stepi3_start;
        //...............................................................................................

        //...............................................................................................
        // Step 5 - Checking if solution reached steady state
        // Checking the steady state condition
        int step5_start = clock();

        float TV, diffv; TV = 0;
        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 2; j++)
            {
                diffv = v[i * wv + j] - vold[i * wv + j];
                TV = TV + pow(pow(diffv, 2), 0.5);
            } // end for i
        } // end for j

        TV = TV / ((nx - 1)*(ny - 2));

        if (TV < st && error < eps)
        {
            cout << "Steady state time = " << t << " (s) " << endl;
            break;
        }
        counter = counter + 1;
        if (fmod(counter, 10) == 0 || counter == 1)
        {
            //cout << "" << endl;
            //cout << "Column" << setw(30) << "time(s)" << setw(30) << "Iterations on Pressure" << setw(30) << "Pressure Residual" << setw(30) << "Total Variance" << endl;
        } // end if
        int step5_end = clock();
        stepTimingAccumulator["Step 5 - Check for steady state"] += step5_end - step5_start;
        //...............................................................................................


        //cout << column << setw(30) << t << setw(30) << iter << setw(30) << error << setw(30) << TV << endl;
        g << column << setw(30) << t << setw(30) << iter << setw(30) << error << setw(30) << TV << endl;
        t = t + dt;
        column = column + 1;


		std::vector<float> toDraw(u);
		s_drawData.Init(toDraw, wu, *(std::max_element(toDraw.begin(), toDraw.end())), *(std::min_element(toDraw.begin(), toDraw.end())));
		display();
    } // end while time

      //........................................................................................................

      // Step 6
      // Co-locate the staggered grid points 
    int step6_start = clock();
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            vc[i * wc + j] = 1.0 / 2.0*(v[(i + 1) * wv + j] + v[i * wv + j]);
            pc[i * wc + j] = 1.0 / 4.0*(p[i * wp + j] + p[(i + 1) * wp + j] + p[i * wp + j + 1] + p[(i + 1) * wp + j + 1]);
            uc[i * wc + j] = 1.0 / 2.0*(u[i*wu + j] + u[i * wu + j + 1]);
            om[i * wc + j] = 1.0 / dx*(v[(i + 1) * wv + j] - v[i * wv + j]) - 1.0 / dy*(u[i * wu + j + 1] - u[i * wu + j]);
            Tc[i * wc + j] = 1.0 / 4.0*(T[i * wT + j] + T[(i + 1) * wT + j] + T[i * wT + j + 1] + T[(i + 1) * wT + j + 1]);
        } // end for j
    } // end for i
      //........................................................................................................
    int step6_end = clock();
    stepTimingAccumulator["Step 6 - Co-locate staggered grid points"] += step6_end - step6_start;

    // Steady state results

    for (int j = 0; j < ny; j++)
    {
        for (int i = 0; i < nx; i++)
        {
            f << setw(15) << t - dt << setw(15) << i*dx << setw(15) << j*dy << setw(15) << uc[i * wc + j] << setw(15) << vc[i * wc + j] << setw(15) << pc[i * wc + j] << setw(15) << Tc[i * ny + j] * T_amb - 273.15 << setw(15) << om[i * wc + j] << endl;
        } // end for i
    } // end for j
      //.........................................................................................................

    float end_clock = clock();
    cout << "CPU time = " << (end_clock - start_clock) / CLOCKS_PER_SEC << " (s)" << endl;
    //cout << "Re = " << Re << endl;
    //cout << "Fr = " << Fr << endl;

    for (auto it = stepTimingAccumulator.begin(); it != stepTimingAccumulator.end(); it++)
    {
        float seconds = (float)it->second / CLOCKS_PER_SEC;
        std::cout << it->first << "\t" << seconds << endl;
    }

   

}

void drawOrientedTriangle2D(float u, float v, float x, float y)
{
	float angle = atan2f(v , u) * 180 / 3.1415926f;
	
	glColor3f(1, 1, 1);
	glPushMatrix();
	glTranslatef(x, y, 0);
	glRotatef(angle, 0, 0, 1);
	glScalef(0.04, 0.04, 1);
	glEnable(GL_POLYGON_SMOOTH);
	glBegin(GL_TRIANGLES);
		glVertex2d(-0.8, 0.3);
		glVertex2d(-0.8, -0.3);
		glVertex2d(0.8, 0);
	glEnd();
	glPopMatrix();
	glDisable(GL_POLYGON_SMOOTH);
}

void renderPrimitive()
{
    
    int windowWidth = glutGet(GLUT_WINDOW_WIDTH);
    int windowHeight = glutGet(GLUT_WINDOW_HEIGHT);
    if (s_drawData.m_data.size() > 0)
    {
        glPointSize(windowWidth / s_drawData.m_width);
        glBegin(GL_POINTS);
        int height = s_drawData.m_data.size() / s_drawData.m_width;
        for (int i = 0; i < s_drawData.m_data.size(); ++i)
        {
            float x = i / s_drawData.m_width;
            float y = i - x*s_drawData.m_width;
            rgb color = s_drawData.m_data[i];
            glColor3f(color.r, color.g, color.b);

			x = (2 * x / height) - 1;
			y = (2 * y / s_drawData.m_width) - 1;
            glVertex2d(x, y);
        }
        glEnd();
    }

}

void drawOrientedTriangles(thrust::host_vector<float>& u, int wu, thrust::host_vector<float>&  v, int wv, float occurence_rate /* 0 to 1 */)
{
	//assert(u.size() == v.size());
	
	for (int i = 0; i < u.size(); ++i)
	{
	
			float x = i / wu;
			float y = i - x*wu;
			int height = u.size() / wu;
			if (((int)x % 6) == 0 && ((int)y % 6) == 0)
			{
				x = (2 * x / height) - 1;
				y = (2 * y / wu) - 1;
				drawOrientedTriangle2D(u[i], v[i], x, y);
			}

	}
}

void display()
{
    glClearColor(0.3f, 0.3f, 0.3f, 0.3f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-2, 2, -2, 2, -1, 1);

    glPushMatrix();
    glTranslatef(0.0f, 0.0f, -0.5f);
    renderPrimitive();
	//drawOrientedTriangles(u_global, wu_global, v_global, wv_global, 0.17);
    glPopMatrix();

    glutSwapBuffers();
}
