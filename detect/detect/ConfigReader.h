#pragma once

#include <fstream>
#include <string>
#include <map>

class ConfigReader
{
public:
	bool Create(std::string filename)
	{
		std::string line;
		size_t div;

		std::ifstream file(filename.c_str());
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

	void GetParamValue(const std::string& param, int& value) const { auto search = params.find(param); if (search != params.end()) value = atoi(search->second.c_str()); }
	void GetParamValue(const std::string& param, double& value) const { auto search = params.find(param); if (search != params.end()) value = atof(search->second.c_str()); }
	void GetParamValue(const std::string& param, std::string& value) const { auto search = params.find(param); if (search != params.end()) value = search->second; }
	void GetParamValue(const std::string& param, bool& value) const { auto search = params.find(param); if (search != params.end()) value = atoi(search->second.c_str()); }

private:
	std::map<std::string, std::string> params;
};