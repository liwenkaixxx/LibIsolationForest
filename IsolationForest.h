#pragma once

#include <map>
#include <random>
#include <set>
#include <stdint.h>
#include <string>
#include <time.h>
#include <vector>
#include <random>
#include <math.h>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>
#include <io.h>
#include <stdio.h>
#include <queue>


using namespace::std;

namespace IsolationForest
{
	//自定义函数
	void traverseDir(const char *dir, vector<string> &vfile, vector<string> &vname);
	void split(const std::string& str, const std::string& sp, std::vector<std::string>& out);




	//该类表示一个特征。每个样本具有一个或多个特征。每个特征都有名称和值。
	class Feature
	{
	public:
		Feature(const std::string& name, uint64_t value) { m_name = name; m_value = value; };
		virtual ~Feature() {};

		virtual void Name(std::string& name) { m_name = name; };
		virtual std::string Name() const { return m_name; };

		virtual void Value(uint64_t value) { m_value = value; };
		virtual uint64_t Value() const { return m_value; };

	protected:
		std::string m_name;
		uint64_t m_value;

	private:
		Feature() {};
	};

	typedef Feature* FeaturePtr;
	typedef std::vector<FeaturePtr> FeaturePtrList;


	// 这个类代表一个样本。每个样本都有一个名称和特征列表。
	class Sample
	{
	public:
		Sample() {};
		Sample(const std::string& name) { m_name = name; };
		virtual ~Sample() {};

		virtual void AddFeatures(const FeaturePtrList& features) 
		{ 
			m_features.insert(m_features.end(), features.begin(), features.end());
		};
		virtual void AddFeature(const FeaturePtr feature) { m_features.push_back(feature); };
		virtual FeaturePtrList Features() const { return m_features; };

	private:
		std::string m_name;
		FeaturePtrList m_features;
	};

	typedef Sample* SamplePtr;
	typedef std::vector<SamplePtr> SamplePtrList;

	// 树节点，内部使用。
	class Node
	{
	public:
		Node();
		Node(const std::string& featureName, uint64_t splitValue);
		virtual ~Node();

		virtual std::string FeatureName() const { return m_featureName; };
		virtual uint64_t SplitValue() const { return m_splitValue; };

		Node* Left() const { return m_left; };
		Node* Right() const { return m_right; };

		void SetLeftSubTree(Node* subtree);
		void SetRightSubTree(Node* subtree);

	private:
		std::string m_featureName;
		uint64_t m_splitValue;

		Node* m_left;
		Node* m_right;

		void DestroyLeftSubtree();
		void DestroyRightSubtree();
	};

	typedef Node* NodePtr;
	typedef std::vector<NodePtr> NodePtrList;


	//这个类抽象随机数生成。
	//如果您希望提供自己的随机化器，则继承这个类。
	//使用林：：StRANDMODER用您选择的一个来重写默认的随机化器。
	class Randomizer
	{
	public:
		Randomizer() : m_gen(m_rand()) {} ;
		virtual ~Randomizer() { };

		virtual uint64_t Rand() { return m_dist(m_gen); };
		virtual uint64_t RandUInt64(uint64_t min, uint64_t max) { return min + (Rand() % (max - min + 1)); }

	private:
		std::random_device m_rand;
		std::mt19937_64 m_gen;
		std::uniform_int_distribution<uint64_t> m_dist;
	};

	typedef std::set<uint64_t> Uint64Set;
	typedef std::map<std::string, Uint64Set> FeatureNameToValuesMap;

	// 孤立森林类
	class Forest
	{
	public:
		Forest();
		Forest(uint32_t numTrees, uint32_t subSamplingSize);
		virtual ~Forest();

		void SetRandomizer(Randomizer* newRandomizer);
		void AddSample(const Sample& sample);
		void Create();
		double Score(const Sample& sample);

	private:
		Randomizer* m_randomizer; // 执行随机数生成
		FeatureNameToValuesMap m_featureValues; // 列出每个特征并将其映射到训练集中的所有唯一值
		NodePtrList m_trees; // 构成森林的决策树
		uint32_t m_numTreesToCreate; //创建树的最大数量
		uint32_t m_subSamplingSize; // 树的最大深度

		NodePtr CreateTree(const FeatureNameToValuesMap& featureValues, size_t depth);
		double Score(const Sample& sample, const NodePtr tree);
		void Destroy();
		void DestroyRandomizer();
	};



};
