#include "IsolationForest.h"
#include <stdlib.h>
#include <iostream>
#include <inttypes.h>
#include <iostream>
#include <fstream>




using namespace IsolationForest;


void CalculationResults(const char* pfpath,/*const char* out_path,const char* all_outpath,*/string &result,string filename)
{
	FeaturePtrList features;
	Forest forest(100, 256);
	Sample sample(filename);
	FILE *fp = fopen(pfpath, "rb");
	if (!fp)
	{
		printf("fopen %s fail!\n", pfpath);
		result = "fopen fail!";
		return;
	}
	char buff[1024] = "";
	while (fgets(buff, sizeof(buff), fp))
	{
		std::vector<std::string> out;
		split(buff, "\t", out);
		features.push_back(new Feature("_DY_price", atoi(out[3].c_str())));
		features.push_back(new Feature("totalCount", atoi(out[2].c_str())));
		features.push_back(new Feature("goodsQualityScore", atoi(out[4].c_str())));
		sample.AddFeatures(features);
		forest.AddSample(sample);
	}
	fclose(fp);
	// Create the isolation forest.
	forest.Create();

	double score = forest.Score(sample);
	std::cout << "Outlier test sample " << score << std::endl;

	return;
}



int main(int argc, const char * argv[])
{
	std::vector<std::string> files;
	std::vector<std::string> name;
	std::map<int, std::string> mapfile;
	std::string result;
	const char* dir = "C:\\Users\\suning\\PycharmProjects\\data\\train";
	const char* out_path = "C:\\Users\\suning\\PycharmProjects\\data\\trainout\\out1.csv";
	std::fstream f_out;
	f_out.open(out_path, ios::app);
	traverseDir(dir, files, name);


	for (size_t i = 0; i < files.size(); i++)
	{
		string temp = name[i];
		temp = temp.erase(temp.find("."));
		const char * p = files[i].data();
		CalculationResults(p, result,temp);
		if (result.size() != 0)
		{
			cout << temp << ":\tsuccess" << endl;
			mapfile[atoi(temp.c_str())] = result;
		}
	}

	for (auto i = mapfile.begin(); i != mapfile.end(); i++)
	{
		f_out << i->second;
	}
	f_out.close();

	system("pause");
	return 0;
}
