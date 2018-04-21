#pragma once

#include <random>

class RandGen
{
public:
	RandGen (unsigned int seed=0)
	{
		engine = new std::mt19937(seed);
	}

	double InRangeF(double l, double u)
	{
		std::uniform_real_distribution<double> dist(l, u);
#if defined HAVE_TBB
		mutex.lock();
#endif
		double n = dist(*engine);
#if defined HAVE_TBB
		mutex.unlock();
#endif
		return n;
	}

	int InRangeI(int l, int u)
	{
		std::uniform_int_distribution<int> dist(l, u);
#if defined HAVE_TBB
		mutex.lock();
#endif
		int n = dist(*engine);;
#if defined HAVE_TBB
		mutex.unlock();
#endif
		return n;
	}

private:
	std::mt19937* engine;
#if defined HAVE_TBB
	tbb::spin_mutex mutex;
#endif
};
