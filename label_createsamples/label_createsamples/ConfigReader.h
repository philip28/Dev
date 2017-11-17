#pragma once

#include <string>
#include <map>
using namespace std;

class ConfigReader
{
public:
	bool Create(string filename)
	{
		string line;
		size_t div;

		ifstream file(filename.c_str());
		if (!file.is_open())
			return false;

		while (!file.eof())
		{
			getline(file, line);
			line.erase(line.find_last_not_of(" \n\r\t") + 1);
			if (line.empty()) continue;
			if (line.at(0) == '#') continue; /* comment */

			if (!(div = line.find_first_of('='))) continue;
			if (div == line.size()) continue;
			params[line.substr(0, div)] = line.substr(div + 1, line.size() - div);
		}

		return true;
	}

	void GetParamValue(string param, int &value) { 
		auto search = params.find(param);
		if (search != params.end())
			value = atoi(search->second.c_str()); }
	void GetParamValue(string param, double &value) { auto search = params.find(param); if (search != params.end()) value = atof(search->second.c_str()); }
	void GetParamValue(string param, string &value) { auto search = params.find(param); if (search != params.end()) value = search->second; }
	void GetParamValue(string param, bool &value) { auto search = params.find(param); if (search != params.end()) value = atoi(search->second.c_str()); }

private:
	map<string, string> params;
};