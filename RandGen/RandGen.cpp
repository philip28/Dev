#include "RandGen.h"

std::mt19937* RandGen::engine = 0;

void RandGen::Init(unsigned int seed = 0)
{
	engine = new std::mt19937(seed);
}

double RandGen::InRangeF(double l, double u)
{
	std::uniform_real_distribution<double> dist(l, u);
	return dist(*engine);
}
int RandGen::InRangeI(int l, int u)
{
	std::uniform_int_distribution<int> dist(l, u);
	return dist(*engine);
}
