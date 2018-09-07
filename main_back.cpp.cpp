#include "IsolationForest.h"
#include <stdlib.h>
#include <inttypes.h>
#include <iostream>
#include <fstream>

using namespace IsolationForest;

int mainback(int argc, const char * argv[])
{
	const size_t NUM_TRAINING_SAMPLES = 100;
	const size_t NUM_TEST_SAMPLES = 10;
	const size_t NUM_TREES_IN_FORESTA = 10;
	const size_t SUBSAMPLING_SIZEA = 10;

	std::ofstream outStream;

	// Parse the command line arguments.
	for (int i = 1; i < argc; ++i)
	{
		if ((strstr(argv[i], "outfile") == 0) && (i + 1 < argc))
		{
			outStream.open(argv[i + 1]);
		}
	}

	Forest forest(NUM_TREES_IN_FORESTA, SUBSAMPLING_SIZEA);

	srand((unsigned int)time(NULL));

	// Create some training samples.
	for (size_t i = 0; i < NUM_TRAINING_SAMPLES; ++i)
	{
		Sample sample("training");
		FeaturePtrList features;

		uint32_t x = rand() % 25;
		uint32_t y = rand() % 25;

		features.push_back(new Feature("x", x));
		features.push_back(new Feature("y", y));

		sample.AddFeatures(features);
		forest.AddSample(sample);

		if (outStream.is_open())
		{
			outStream << "training," << x << "," << y << std::endl;
		}
	}

	// Create the isolation forest.
	forest.Create();

	// Test samples (similar to training samples).
	std::cout << "Test samples that are similar to the training set." << std::endl;
	std::cout << "--------------------------------------------------" << std::endl;
	double avgNormalScore = (double)0.0;
	for (size_t i = 0; i < NUM_TEST_SAMPLES; ++i)
	{
		Sample sample("normal sample");
		FeaturePtrList features;

		uint32_t x = rand() % 25;
		uint32_t y = rand() % 25;

		features.push_back(new Feature("x", x));
		features.push_back(new Feature("y", y));
		sample.AddFeatures(features);

		// Run a test with the sample that doesn't contain outliers.
		double score = forest.Score(sample);
		std::cout << "Normal test sample " << i << ": " << score << std::endl;

		if (outStream.is_open())
		{
			outStream << "normal," << x << "," << y << std::endl;
		}

		avgNormalScore += score;
	}
	std::cout << std::endl;

	// Outlier samples (different from training samples).
	std::cout << "Test samples that are different from the training set." << std::endl;
	std::cout << "------------------------------------------------------" << std::endl;
	double avgOutlierScore = (double)0.0;
	for (size_t i = 0; i < NUM_TEST_SAMPLES; ++i)
	{
		Sample sample("outlier sample");
		FeaturePtrList features;

		uint32_t x = 20 + (rand() % 25);
		uint32_t y = 20 + (rand() % 25);

		features.push_back(new Feature("x", x));
		features.push_back(new Feature("y", y));
		sample.AddFeatures(features);

		// Run a test with the sample that contains outliers.
		double score = forest.Score(sample);
		std::cout << "Outlier test sample " << i << ": " << score << std::endl;

		if (outStream.is_open())
		{
			outStream << "outlier," << x << "," << y << std::endl;
		}

		avgOutlierScore += score;
	}
	std::cout << std::endl;

	if (outStream.is_open())
	{
		outStream.close();
	}

	system("pause");
	return 0;
}
