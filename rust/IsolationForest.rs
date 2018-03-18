//	MIT License
//
//  Copyright © 2017 Michael J Simms. All rights reserved.
//
//	Permission is hereby granted, free of charge, to any person obtaining a copy
//	of this software and associated documentation files (the "Software"), to deal
//	in the Software without restriction, including without limitation the rights
//	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//	copies of the Software, and to permit persons to whom the Software is
//	furnished to do so, subject to the following conditions:
//
//	The above copyright notice and this permission notice shall be included in all
//	copies or substantial portions of the Software.
//
//	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//	SOFTWARE.

/// Each feature has a name and value.
pub struct Feature {
    name: String,
    value: u64,
}

impl Feature {
    pub fn new (name: &str, value: u64) -> Feature {
        Feature { name: name.to_string(), value: value }
    }

    pub fn set_name(&self, name: String) {
        self.name = name;
    }

    pub fn get_name(&self) -> String {
        self.name
    }

    pub fn set_value(&self, value: u64) {
        self.value = value;
    }

    pub fn get_value(&self) -> u64 {
        self.value
    }
}

pub type FeatureList = Vec<Feature>;

/// This class represents a sample.
/// Each sample has a name and list of features.
pub struct Sample {
    name: String,
    features: FeatureList,
}

impl Sample {
    pub fn new (name: &str) -> Sample {
        Sample { name: name.to_string() }
    }

    pub fn add_features(&self, features: FeatureList) {
        self.features.append(features);
    }

    pub fn add_feature(&self, feature: Feature) {
        self.features.push(feature);
    }

    pub fn features(&self) -> FeatureList {
        self.features
    }
}

pub type SampleList = Vec<Sample>;

/// Tree node, used internally.
struct Node {
    feature_name: String,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

impl Node {
    pub fn new (feature_name: &str) -> Node {
        Node { feature_name: feature_name.to_string() }
    }

	fn set_left_subtree(&self, subtree: Node)
	{
		self.left = subtree;
	}

	fn set_right_subtree(&self, subtree: Node)
	{
		self.right = subtree;
	}
}

pub type NodeList = Vec<Node>;

/// Isolation Forest implementation.
pub struct Forest {
    //featureValues: FeatureNameToValueMap, // Lists each feature and maps it to all unique values in the training set
    trees: NodeList, // The decision trees that comprise the forest
    numTreesToCreate: u32, // The maximum number of trees to create
    subSamplingSize: u32, // The maximum depth of a tree
}

impl Forest {
    pub fn new (numTreesToCreate: u32, subSamplingSize: u32) -> Forest {
        Forest { numTreesToCreate: numTreesToCreate, subSamplingSize: subSamplingSize }
    }

    pub fn add_sample(&self, sample: Sample) {
    }

    pub fn create(&self) {
    }

    pub fn score(&self, sample: Sample) -> f64 {
    }
}
