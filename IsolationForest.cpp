#include "IsolationForest.h"

namespace IsolationForest
{
	Node::Node() :
		m_splitValue(0),
		m_left(NULL),
		m_right(NULL)
	{
	}

	Node::Node(const std::string& featureName, uint64_t splitValue) :
		m_featureName(featureName),
		m_splitValue(splitValue),
		m_left(NULL),
		m_right(NULL)
	{
	}

	Node::~Node()
	{
		DestroyLeftSubtree();
		DestroyRightSubtree();
	}

	void Node::SetLeftSubTree(Node* subtree)
	{
		DestroyLeftSubtree();
		m_left = subtree;
	}

	void Node::SetRightSubTree(Node* subtree)
	{
		DestroyRightSubtree();
		m_right = subtree;
	}

	void Node::DestroyLeftSubtree()
	{
		if (m_left)
		{
			delete m_left;
			m_left = NULL;
		}
	}

	void Node::DestroyRightSubtree()
	{
		if (m_right)
		{
			delete m_right;
			m_right = NULL;
		}
	}

	Forest::Forest() :
		m_randomizer(new Randomizer()),
		m_numTreesToCreate(10),
		m_subSamplingSize(0)
	{
	}

	Forest::Forest(uint32_t numTrees, uint32_t subSamplingSize) :
		m_randomizer(new Randomizer()),
		m_numTreesToCreate(numTrees),
		m_subSamplingSize(subSamplingSize)
	{
	}

	Forest::~Forest()
	{
		DestroyRandomizer();
		Destroy();
	}

	void Forest::SetRandomizer(Randomizer* newRandomizer)
	{
		DestroyRandomizer();
		m_randomizer = newRandomizer;
	}

	//��ÿ��������������ӵ���֪�����б��С�
	//������Ӧ��Ψһֵ����
	void Forest::AddSample(const Sample& sample)
	{
        // ��ֱ�Ӵ洢������ֻ��������
		const FeaturePtrList& features = sample.Features();
		FeaturePtrList::const_iterator featureIter = features.begin();
		while (featureIter != features.end())
		{
			const FeaturePtr feature = (*featureIter);
			const std::string& featureName = feature->Name();
			uint64_t featureValue = feature->Value();

			// �������������ֵ������
			if (m_featureValues.count(featureName) == 0)
			{
				Uint64Set featureValueSet;
				featureValueSet.insert(featureValue);
				m_featureValues.insert(std::make_pair(featureName, featureValueSet));
			}
			else
			{
				Uint64Set& featureValueSet = m_featureValues.at(featureName);
				featureValueSet.insert(featureValue);
			}

			++featureIter;
		}
	}


	//�����ͷ��ص���������Ϊ���ǵݹ麯����
	//���ָʾ�ݹ�ĵ�ǰ��ȡ�
	NodePtr Forest::CreateTree(const FeatureNameToValuesMap& featureValues, size_t depth)
	{
		// Sanity check.
		if (featureValues.size() <= 1)
		{
			return NULL;
		}

		// ����������������ȣ���ֹͣ��
		if ((m_subSamplingSize > 0) && (depth >= m_subSamplingSize))
		{
			return NULL;
		}

		// ���ѡ��һ��������
		size_t selectedFeatureIndex = (size_t)m_randomizer->RandUInt64(0, featureValues.size() - 1);
		FeatureNameToValuesMap::const_iterator featureIter = featureValues.begin();
		std::advance(featureIter, selectedFeatureIndex);
		const std::string& selectedFeatureName = (*featureIter).first;

		// ��ȡֵ�б���в�֡�
		const Uint64Set& featureValueSet = (*featureIter).second;
		if (featureValueSet.size() == 0)
		{
			return NULL;
		}

		// ���ѡ��һ������ֵ.
		size_t splitValueIndex = 0;
		if (featureValueSet.size() > 1)
		{
			splitValueIndex = (size_t)m_randomizer->RandUInt64(0, featureValueSet.size() - 1);
		}
		Uint64Set::const_iterator splitValueIter = featureValueSet.begin();
		std::advance(splitValueIter, splitValueIndex);
		uint64_t splitValue = (*splitValueIter);

		// �������ڵ���������ֵ��
		NodePtr tree = new Node(selectedFeatureName, splitValue);
		if (tree)
		{

			//�����ղ�ʹ�õ�����ֵ���������汾�����һ�������ұ�һ�á�

			FeatureNameToValuesMap tempFeatureValues = featureValues;

			// ������������
			Uint64Set leftFeatureValueSet = featureValueSet;
			splitValueIter = leftFeatureValueSet.begin();
			std::advance(splitValueIter, splitValueIndex);
			leftFeatureValueSet.erase(splitValueIter, leftFeatureValueSet.end());
			tempFeatureValues[selectedFeatureName] = leftFeatureValueSet;
			tree->SetLeftSubTree(CreateTree(tempFeatureValues, depth + 1));

			// ������������
			if (splitValueIndex < featureValueSet.size() - 1)
			{
				Uint64Set rightFeatureValueSet = featureValueSet;
				splitValueIter = rightFeatureValueSet.begin();
				std::advance(splitValueIter, splitValueIndex + 1);
				rightFeatureValueSet.erase(rightFeatureValueSet.begin(), splitValueIter);
				tempFeatureValues[selectedFeatureName] = rightFeatureValueSet;
				tree->SetRightSubTree(CreateTree(tempFeatureValues, depth + 1));
			}
		}

		return tree;
	}

	//��������ָ�������캯���������������֡�
	void Forest::Create()
	{
		m_trees.reserve(m_numTreesToCreate);

		for (size_t i = 0; i < m_numTreesToCreate; ++i)
		{
			NodePtr tree = CreateTree(m_featureValues, 0);
			if (tree)
			{
				m_trees.push_back(tree);
			}
		}
	}


	// ����ָ��������������������
	double Forest::Score(const Sample& sample, const NodePtr tree)
	{
		double depth = (double)0.0;

		const FeaturePtrList& features = sample.Features();

		NodePtr currentNode = tree;
		while (currentNode)
		{
			bool foundFeature = false;

			// ���������ҵ���һ��������
			FeaturePtrList::const_iterator featureIter = features.begin();
			while (featureIter != features.end() && !foundFeature)
			{
				const FeaturePtr currentFeature = (*featureIter);
				if (currentFeature->Name().compare(currentNode->FeatureName()) == 0)
				{
					if (currentFeature->Value() < currentNode->SplitValue())
					{
						currentNode = currentNode->Left();
					}
					else
					{
						currentNode = currentNode->Right();
					}
					++depth;
					foundFeature = true;
				}
				++featureIter;
			}

			//�����������������û�е���������ô�������ߣ��ѷ���ƽ����һ��
			if (!foundFeature)
			{
				double leftDepth = depth + Score(sample, currentNode->Left());
				double rightDepth = depth + Score(sample, currentNode->Right());
				return (leftDepth + rightDepth) / (double)2.0;
			}
		}
		return depth;
	}

	// ������ɭ�ֵ�������ȡ����
	double Forest::Score(const Sample& sample)
	{
		double score = (double)0.0;
		
		if (m_trees.size() > 0)
		{
			NodePtrList::const_iterator treeIter = m_trees.begin();
			while (treeIter != m_trees.end())
			{
				score += (double)Score(sample, (*treeIter));
				++treeIter;
			}
			score /= (double)m_trees.size();
		}
		return score;
	}

	//��������ɭ�ֵ�����
	void Forest::Destroy()
	{
		std::vector<NodePtr>::iterator iter = m_trees.begin();
		while (iter != m_trees.end())
		{
			NodePtr tree = (*iter);
			if (tree)
			{
				delete tree;
			}
			++iter;
		}
		m_trees.clear();
	}

	//�ͷ��Զ����������������еĻ�����
	void Forest::DestroyRandomizer()
	{
		if (m_randomizer)
		{
			delete m_randomizer;
			m_randomizer = NULL;
		}
	}

	void traverseDir(const char *dir, vector<string> &vfile, vector<string> &vname)
	{
		//�ж�Ŀ¼�ṹ//
		char trans_dir[256];
		trans_dir[0] = '\0';
		if (dir[strlen(dir) - 1] == '\\')
		{
			int i = 0;
			int dir_len = strlen(dir);
			for (i = 0; i < dir_len; ++i)
			{
				trans_dir[i] = dir[i];
			}
			trans_dir[i++] = '*';
			trans_dir[i++] = 0;
		}
		if (dir[strlen(dir) - 1] != '*' && dir[strlen(dir) - 1] != '\\')
		{
			int i = 0;
			int dir_len = strlen(dir);
			for (i = 0; i < dir_len; ++i)
			{
				trans_dir[i] = dir[i];
			}
			trans_dir[i++] = '\\';
			trans_dir[i++] = '*';
			trans_dir[i++] = 0;
		}
		dir = trans_dir;
		_finddata_t fileDir;
		long lfDir;

		queue<string> queue_dir;////�ö���ʵ�ֵݹ�
		queue_dir.push(string(dir));

		while (!queue_dir.empty())
		{
			string curDir = queue_dir.front();
			if ((lfDir = _findfirst(curDir.c_str(), &fileDir)) != -1l)//is -1l no file found
			{

				while (_findnext(lfDir, &fileDir) == 0)
				{
					if ((fileDir.attrib >= 16 && fileDir.attrib <= 23) || (fileDir.attrib >= 48 && fileDir.attrib <= 55))//��Ŀ¼���������
					{
						if (fileDir.name[0] != '.')//���ǵ�ǰĿ¼* �Ҳ�����һ��Ŀ¼**
						{
							string tmpstr = curDir;//��ȥ���һ��*��
							tmpstr.erase(tmpstr.end() - 1);
							tmpstr.append(string(fileDir.name));
							tmpstr.append("\\*");
							queue_dir.push(tmpstr);//�ѵ�ǰĿ¼�ŵ��������Ա���һ�α���
						}
					}
					else
					{
						vname.push_back(fileDir.name);
						string tmpfilename = curDir;
						tmpfilename.erase(tmpfilename.end() - 1);
						vfile.push_back(tmpfilename.append(string(fileDir.name)));
					}
				}

			}
			queue_dir.pop();
		}
	}
	void split(const std::string& str, const std::string& sp, std::vector<std::string>& out)
	{
		out.clear();
		std::string s = str;
		size_t beg, end;
		while (!s.empty())
		{
			beg = s.find_first_not_of(sp);
			if (beg == std::string::npos)
			{
				break;
			}
			end = s.find(sp, beg);
			out.push_back(s.substr(beg, end - beg));
			if (end == std::string::npos)
			{
				break;
			}
			s = s.substr(end, s.size() - end);
		}
	}

}
