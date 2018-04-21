#pragma once

#include <fstream>
#include <string>
#include <map>

typedef struct
{
	std::string value;
	std::string desc;
} value_cnt;

typedef std::map<std::string, value_cnt> params_type;

class ConfigReader
{
public:
	bool Create(std::string filename)
	{
		std::string line;
		std::basic_string<char>::size_type div, rem;
		value_cnt val;

		std::ifstream file(filename.c_str());
		if (!file.is_open())
			return false;

		while (!file.eof())
		{
			getline(file, line);

			if ((rem = line.find_first_of('#')) != std::string::npos) /* comment */
				line.erase(rem, line.length() - rem);

			line.erase(line.find_last_not_of(" \n\r\t") + 1);
			if (line.empty()) continue;

			if ((div = line.find_first_of('=')) == std::string::npos) continue;
			if (div == line.size()) continue;

			val.value = line.substr(div + 1, line.size() - div);
			params[line.substr(0, div)] = val;
		}

		return true;
	}

	template <typename T>
	void InitParam(const std::string& name, const std::string& desc, T& value)
	{
		auto search = params.find(name);
		if (search != params.end())
		{
			search->second.desc = desc;
			ValueCast(search->second.value, value);
		}
	}

	void Print()
	{
		for (auto &param : params)
			std::cout << param.second.desc << ":  " << param.second.value << std::endl;
	}


	// deprecated
	void GetParamValue(const std::string& param, int& value) const { auto search = params.find(param); if (search != params.end()) ValueCast(search->second.value, value); }
	void GetParamValue(const std::string& param, double& value) const { auto search = params.find(param); if (search != params.end()) ValueCast(search->second.value, value); }
	void GetParamValue(const std::string& param, std::string& value) const { auto search = params.find(param); if (search != params.end()) ValueCast(search->second.value, value); }
	void GetParamValue(const std::string& param, bool& value) const { auto search = params.find(param); if (search != params.end()) ValueCast(search->second.value, value); }

private:
	params_type params;

	void ValueCast(const std::string& value, int& cast) const { cast = atoi(value.c_str()); }
	void ValueCast(const std::string& value, double& cast) const  { cast = atof(value.c_str()); }
	void ValueCast(const std::string& value, std::string& cast) const { cast = value; }
	void ValueCast(const std::string& value, bool& cast) const
	{
		if (value == "true")
			cast = true;
		else if (value == "false")
			cast = false;
		else
			cast = atoi(value.c_str());
	}
};