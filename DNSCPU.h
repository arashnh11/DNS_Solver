#pragma once

#include <vector>

class DNSCPU
{
public:
	DNSCPU();
	~DNSCPU();

	enum SIMSTATES
	{
		INIT,
		STEP,
		FINISH,
		END
	};

	static void RunSimulation();

	static std::vector<float>& getPressure();
	static float getPressureWidth();

	static std::vector<float>& getTemperature();
	static float getTemperatureWidth();

    static std::vector<float>& getU();
    static float getUWidth();

private:

	// Call this to initialize shit
	static void Init();

	// Call this to move ahead in time. dt is defined in the .cpp file
	static void Step();
	
	// Call this to Finish 
	static void Finish();



};

