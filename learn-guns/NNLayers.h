#pragma once
 
#include "res.h"

struct Data
{
public:
	virtual ~Data() {}

	virtual void loadFromString(string const& data) = 0;
	virtual string dataDims() const = 0;
};

struct ConvData : public Data
{
public:
	//depth, x, y
	vector<vector<vector<float>>> tensor;

	void loadFromString(string const& data) override;
	void loadFromImage(vector<vector<float>> const& data)
	{
		tensor.clear();
		tensor.push_back(data);
	}
	string dataDims() const override
	{
		std::ostringstream oss;
		oss << tensor.size() << "," << tensor[0].size() << "," << tensor[0][0].size();
		return oss.str();
	}

	~ConvData() {}
};

struct DenseData : public Data
{
public:

	vector<float> data;

	void loadFromString(string const& data) override;
	string dataDims() const override
	{
		std::ostringstream oss;
		oss << data.size();
		return oss.str();
	}

	~DenseData() {}
};


class Layer
{
public:
	virtual Data* computeOutput(Data* input) = 0;
	virtual void loadLayer(string const& layerString) = 0;
	virtual void loadLayer(byte* layer, int len) = 0;
};

class Conv2DLayer : public Layer
{
private:
	int x, y, z, k;
	vector<vector<vector<vector<float>>>> kernels;
	vector<float> biases;
	string activation;
public:
	Data* computeOutput(Data* input) override;
	void loadLayer(string const& layerString) override;
	void loadLayer(byte* layer, int len) override;
};

class MaxPooling2DLayer : public Layer
{
private:
	int strideW, strideH;
public:
	Data* computeOutput(Data* input) override;
	void loadLayer(string const& layerString) override;
	void loadLayer(byte* layer, int len) override;
};

class FlattenLayer : public Layer
{
public:
	Data* computeOutput(Data* input) override;
	void loadLayer(string const& layerString) {}
	void loadLayer(byte* layer, int len) {}
};

class DenseLayer : public Layer
{
private:
	//neuron, weights
	vector<vector<float>> weights;
	vector<float> biases;
	string activation;

	int inSize, outSize;
public:
	Data* computeOutput(Data* input) override;
	void loadLayer(string const& layerString) override;
	void loadLayer(byte* layer, int len) override;
};

class Model
{
private:
	vector<Layer*> layers;
	void loadModel_string(string const& modelPath);
	void loadModel_byte(string const& modelPath);
public:
	void loadModel(string const& modelPath, bool byteStream = true);
	Data* predict(Data* input);
	int getNumLayers()
	{
		return layers.size();
	}
};