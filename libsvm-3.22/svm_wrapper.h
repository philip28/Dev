#pragma once

#include <vector>
#include "opencv2/opencv.hpp" 
#include "svm.h"

typedef struct
{
	int cid;
	std::string path;
	std::vector<cv::Mat> image_list;
	std::vector<cv::Mat> grad_list;
} class_info;

class svm_problem_wrapper
{
public:
	~svm_problem_wrapper()
	{
		if (problem.l)
		{
			delete problem.x;
			delete[] problem.y;
		}
	}

	void make2class(std::vector<cv::Mat>& c1, std::vector<cv::Mat>& c2)
	{
		int num_c1 = (int)c1.size();
		int num_c2 = (int)c2.size();

		problem.l = num_c1 + num_c2;
		problem.y = new double[problem.l];
		problem.x = new struct svm_node*[problem.l];

		for (int i = 0; i < num_c1; i++)
		{
			problem.y[i] = +1;
			mat_to_svm_node(c1[i], problem.x[i]);
		}
		for (int i = 0; i < num_c2; i++)
		{
			problem.y[i + num_c1] = -1;
			mat_to_svm_node(c2[i], problem.x[i + num_c1]);
		}
	};

	void make_mult_class(std::vector<class_info>& pos, std::vector<cv::Mat>& neg)
	{
		int npos = 0;
		for (int i = 0; i < pos.size(); i++)
			npos += (int)pos[i].grad_list.size();
		int nneg = (int)neg.size();

		problem.l = npos + nneg;
		problem.y = new double[problem.l];
		problem.x = new struct svm_node*[problem.l];

		int gpos = 0;
		for (int i = 0; i < pos.size(); i++)
		{
			int nc = (int)pos[i].grad_list.size();
			for (int j = 0; j < nc; j++)
			{
				problem.y[gpos] = pos[i].cid;
				mat_to_svm_node(pos[i].grad_list[j], problem.x[gpos]);
				gpos++;
			}
		}

		int nc = (int)neg.size();
		for (int i = 0; i < nc; i++)
		{
			problem.y[gpos] = -1;
			mat_to_svm_node(neg[i], problem.x[gpos]);
			gpos++;
		}
	};

	svm_problem problem = {};

	static void mat_to_svm_node(cv::Mat& m, struct svm_node* &s)
	{
		s = new struct svm_node[countNonZero(m) + 1];

		cv::MatIterator_<float> it, end;
		int j = 0, nzindex = 0;
		for (it = m.begin<float>(), end = m.end<float>(); it != end; ++it, j++)
		{
			if (*it != 0)
			{
				s[nzindex].index = j;
				s[nzindex].value = *it;
				nzindex++;
			}
		}
		s[nzindex].index = -1; // end of the vector sign
	}
};
