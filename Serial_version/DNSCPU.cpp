#include "DNSCPU.h"

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


// Define variables in global space (bad practice but we'll go with this for now)
// Enclosed with empty namespace to avoid linking errors with the other implementation of the algorithm
namespace 
{
	using namespace std;

	DNSCPU::SIMSTATES s_nextState = DNSCPU::SIMSTATES::INIT;

	// output format
	float start_clock = clock();

	ofstream f("result_cpu.txt"); // Solution Results
	
	ofstream g("convergence_cpu.txt"); // Convergence history


	// Input parameters 
	//float Re, Pr, Fr, T_L, T_0, T_amb, ni, nj, dx, dy, t, ny, nx, eps, /*beta,*/ iter, maxiter, tf, st, counter, column, u_wind, T_R, Lx, Ly;
	
	float Lx = 4.0, Ly = 5.0; // Domain dimensions
	int ni = 10; // Number of nodes per unit length in x direction
	int nj = 10; // Number of nodes per unit length in y direction
	int nx = Lx * ni; 
	int ny = Ly * nj; // Number of Nodes in each direction
	float u_wind = 1; // Reference velocity
	float st = 0.00005; // Total variance criteria
	float eps = 0.001; // Pressure convergence criteria
	float tf = 100; // Final time step
	float Pr = 0.5*(0.709 + 0.711); // Prandtl number
	float Re = 200.0; float Fr = 0.3; // Non-dimensional numbers for inflow conditions
	float dx = Lx / (nx - 1); 
	float dy = Ly / (ny - 1); // dx and dy
											//beta = 1; // Successive over relaxation factor (SOR)
	float t = 0; // Initial time step
	float T_L = 100.0; // Left wall temperature (C)
	float T_R = 50.0; // Right wall temperature (C)
	float T_amb = 25.0; // Ambient air temperature (C)
	float T_0 = 50.0; // Initial air temperature
	
	int maxiter = 1000; // Maximum iteration at each time step
	int counter = 0; // initial row for output monitoring
	int column = 1; // Column number for output display
				// Records number of clicks a step takes
	std::map<string, uint32_t> stepTimingAccumulator;

	// Vectors

	vector<float> u(nx * (ny + 1), 0);
	vector<float> us(nx*(ny + 1), 0);
	vector<float> uold(nx * (ny + 1), 0);
	int wu = ny + 1;

	vector<float> v((nx + 1) * ny, 0);
	vector<float> vs((nx + 1) * ny, 0);
	vector<float> vold((nx + 1) * ny, 0);
	int wv = ny;

	vector<float> p((nx + 1) * (ny + 1), 0);
	vector<float> pold((nx + 1) * (ny + 1), 0);
	int wp = ny + 1;

	vector<float> T;     // Initializing the flow variable (Temperature)  
														   // Boundary conditions for T (Initialization)
	int wT = ny + 1;

	// These are initialized in the Init() function
	vector<float> Told;
	vector<float> om;
	vector<float> vc;
	vector<float> uc;
	vector<float> pc;
	vector<float> Tc;
	int wc = ny;

	// Time step size stability criterion

	float mt1 = 0.25*pow(dx, 2.0) / (1.0 / Re); float Rer = 1.0 / Re; float mt2 = 0.25*pow(dy, 2.0) / (1.0 / Re);
	float dt;

}


void DNSCPU::RunSimulation()
{
	switch (s_nextState)
	{
	case INIT: Init(); break;
	case STEP: Step(); break;
	case FINISH: Finish(); break;
	case END: break;
	}


}

std::vector<float>& DNSCPU::getPressure()
{
	return p;
}
float DNSCPU::getPressureWidth()
{
	return wp;
}

std::vector<float>& DNSCPU::getTemperature()
{
	return T;
}
float DNSCPU::getTemperatureWidth()
{
	return wT;
}

std::vector<float>& DNSCPU::getU()
{
    return u;
}
float DNSCPU::getUWidth()
{
    return wu;
}

// Call this to initialize shit
void DNSCPU::Init()
{
	f.setf(ios::fixed | ios::showpoint);
	f << setprecision(5);

	g.setf(ios::fixed | ios::showpoint);
	g << setprecision(5);
	cout.setf(ios::fixed | ios::showpoint);
	cout << setprecision(5);

	T_L = T_L + 273.15; T_0 = T_0 + 273.15; T_amb = T_amb + 273.15; T_R = T_R + 273.15;// Unit conversion to (K)

	T.resize((nx + 1) * (ny + 1), T_0 / T_amb);
	Told.resize((nx + 1) * (ny + 1), 0);
	om.resize(nx * ny, 0);
	vc.resize(nx * ny, 0);
	uc.resize(nx * ny, 0);
	pc.resize(nx * ny, 0);
	Tc.resize(nx*ny, 0);

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
	cout << "dt = " << dt << endl;

	s_nextState = STEP;
}

void DNSCPU::Step()
{
	// dt is defined statically 
	if(t<tf)
	{
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

		float error = 1; int iter = 0;
		float diffp;
		// Solve for pressure iteratively until it converges - Using Gauss Seidel SOR 
		while (error > eps)
		{
			error = 0;

			//............................................................................................


			// Fill pold with p

			//	for (int i = 0; i < nx + 1; i++){
			//		for (int j = 0; j < ny + 1; j++){
			//pold[i * wp + j] = p [i * wp + j];
			pold = p;
			//		}
			//	}

			for (int i = 1; i < nx; i++)
			{
				for (int j = 1; j < ny; j++)
				{
					//pold = p[i * wp + j];

					p[i * wp + j] = /*beta**/pow(dx, 2.0)*pow(dy, 2.0) / (-2.0*(pow(dx, 2.0) + pow(dy, 2.0)))*(-1.0 / pow(dx, 2.0)*(pold[(i + 1) * wp + j] + pold[(i - 1) * wp + j] + pold[i * wp + j + 1] + pold[i * wp + j - 1]) + 1.0 / dt*(1.0 / dx*(us[i * wu + j] - us[(i - 1) * wu + j]) + 1.0 / dy*(vs[i * wv + j] - vs[i * wv + j - 1])))/* + (1.0 - beta)*pold[i * wp + j]*/;
					diffp = pow((p[i * wp + j] - pold[i * wp + j]), 2.0);
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
			s_nextState = FINISH;
			return;
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
	}
	else
	{
		s_nextState = FINISH;
	}
}

void DNSCPU::Finish()
{
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

	s_nextState = END;
}
DNSCPU::DNSCPU()
{
}


DNSCPU::~DNSCPU()
{
}
