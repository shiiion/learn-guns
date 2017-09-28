#include "NNLayers.h"
#include <sstream>
#include <algorithm>
#include <ctime>
#include <limits>
#include <iostream>
#include <fstream>

void getFileContents(const char *filename, string& contents)
{
	std::ifstream in(filename, std::ios::in | std::ios::binary);
	if (in)
	{
		in.seekg(0, std::ios::end);
		contents.resize(in.tellg());
		in.seekg(0, std::ios::beg);
		in.read(&contents[0], contents.size());
		in.close();
	}
}

void clearSpaces(string& s)
{
	s.erase(std::remove_if(s.begin(), s.end(), isspace), s.end());
}

void split(string const& s, char delim, vector<string>& out, bool whitespace=false)
{
	std::stringstream ss(s);
	string tok;

	while (std::getline(ss, tok, delim))
	{
		if (!whitespace) clearSpaces(tok);
		if (!tok.empty()) out.push_back(tok);
	}
}

void remove(string& s, char remove)
{
	unsigned int idx;
	while ((idx = s.find(remove)) != string::npos)
	{
		s.erase(s.begin() + idx);
	}
}

void parseArray(string const& arr, vector<float>& out)
{
	vector<string> strValues;
	split(arr, ' ', strValues, true);
	for (string const& s : strValues)
	{
		out.emplace_back(std::stof(s));
	}
}

float logSumExp(vector<float> const& data)
{
	float sum = 0;
	float m = data[0];

	for (int a = 1; a < data.size(); a++) m = max(data[a], m);
	for (int a = 0; a < data.size(); a++)
	{
		sum += exp(data[a] - m);
	}

	return m + log(sum);
}

void activation2d(string const& name, matrix& data)
{
	if (name == "relu")
	{
		for (int a = 0; a < data.size(); a++)
		{
			for (int b = 0; b < data[a].size(); b++)
			{
				data[a][b] = max(0, data[a][b]);
			}
		}
	}
}

void activation1d(string const& name, vector<float>& data)
{
	if (name == "relu")
	{
		for (int a = 0; a < data.size(); a++)
		{
			data[a] = max(0, data[a]);
		}
	}
	else if (name == "softmax")
	{
		float denom = logSumExp(data);

		for (int a = 0; a < data.size(); a++)
		{
			data[a] = exp(data[a] - denom);
		}
	}
}

void convolve(matrix const& kernel, matrix const& data, float bias, matrix& out)
{

	for (int a = 0; a < data.size() - (kernel.size() - 1); a++)
	{
		out.push_back(vector<float>());
		for (int b = 0; b < data[a].size() - (kernel[0].size() - 1); b++)
		{
			float sum = 0;
			for (int c = 0; c < kernel.size(); c++)
			{
				for (int d = 0; d < kernel[c].size(); d++)
				{
					sum += data[a + d][b + c] * kernel[c][d];
				}
			}
			out[a].push_back(sum + bias);
		}
	}
}



void Conv2DLayer::loadLayer(string const& layerString)
{
	std::istringstream iss(layerString);
	string sOut;
	int step = 0;
	int stride = 0;
	
	vector<vector<vector<float>>> kernelsTemp;

	while (std::getline(iss, sOut))
	{
		if (FIND("activation", sOut))
		{
			activation = sOut.substr(11);
		}
		else if (FIND("bias", sOut))
		{
			sOut = sOut.substr(5);

			remove(sOut, '['); remove(sOut, ']');
			parseArray(sOut, biases);
		}
		else if (sOut[0] == '[')
		{
			vector<float> kernel;

			remove(sOut, '['); remove(sOut, ']');
			parseArray(sOut, kernel);

			if (kernelsTemp.size() == stride)
			{
				kernelsTemp.push_back(vector<vector<float>>());
			}

			kernelsTemp[stride].emplace_back(kernel);

			if (step == (x - 1))
			{
				step = -1; stride++;
			}
			step++;
		}
		else
		{
			vector<string> dims;
			split(sOut, ',', dims);
			x = std::stoi(dims[0]); y = std::stoi(dims[1]);
			k = std::stoi(dims[3]);
		}
	}
	for (int a = 0; a < k; a++)
	{
		kernels.push_back(vector<vector<float>>());
		for (int b = 0; b < y; b++)
		{
			kernels[a].push_back(vector<float>());
		}
	}

	for (int a = 0; a < kernelsTemp.size(); a++)
	{
		for (int b = 0; b < kernelsTemp[a].size(); b++)
		{
			for (int c = 0; c < kernelsTemp[a][b].size(); c++)
			{
				kernels[c][b].push_back(kernelsTemp[a][b][c]);
			}
		}
	}
}

Data* Conv2DLayer::computeOutput(Data* input)
{
	ConvData* out = new ConvData();
	ConvData* in = (ConvData*)input;
	for (int a = 0; a < kernels.size(); a++)
	{
		matrix result;
		convolve(kernels[a], in->tensor[0], biases[a], result);
		activation2d(activation, result);
		out->tensor.emplace_back(result);
	}

	return out;
}

void DenseLayer::loadLayer(string const& layerString)
{
	std::istringstream iss(layerString);
	string sOut;

	while (std::getline(iss, sOut))
	{
		if (FIND("activation", sOut))
		{
			activation = sOut.substr(11);
		}
		else if (FIND("bias", sOut))
		{
			sOut = sOut.substr(5);

			remove(sOut, '['); remove(sOut, ']');
			parseArray(sOut, biases);
		}
		else if (sOut[0] == '[')
		{
			vector<float> outputWeights;
			remove(sOut, '['); remove(sOut, ']');
			parseArray(sOut, outputWeights);

			weights.emplace_back(outputWeights);
		}
		else
		{
			vector<string> dims;
			split(sOut, ',', dims, true);
			inSize = std::stoi(dims[0]); outSize = std::stoi(dims[1]);
		}
	}
}

Data* DenseLayer::computeOutput(Data* input)
{
	DenseData* out = new DenseData();
	DenseData* in = (DenseData*)input;

	out->data.reserve(outSize);

	for (int a = 0; a < outSize; a++)
	{
		float neuronIn = 0;
		for (int b = 0; b < inSize; b++)
		{
			neuronIn += weights[b][a] * in->data[b];
		}
		neuronIn += biases[a];
		out->data.push_back(neuronIn);
	}

	activation1d(activation, out->data);
	return out;
}

void MaxPooling2DLayer::loadLayer(string const& data)
{
	vector<string> strides;
	split(data, ',', strides);
	strideW = std::stoi(strides[0]); strideH = std::stoi(strides[1]);
}

Data* MaxPooling2DLayer::computeOutput(Data* input)
{
	ConvData* in = (ConvData*)input;
	ConvData* out = new ConvData();

	for (int a = 0; a < in->tensor.size(); a++)
	{
		out->tensor.push_back(vector<vector<float>>());
		for (int b = 0; b < in->tensor[a].size(); b+=strideW)
		{
			out->tensor[a].push_back(vector<float>());
			for (int c = 0; c < in->tensor[a][b].size(); c+=strideH)
			{
				float maxVal = -std::numeric_limits<float>::infinity();
				for (int d = b; d < b + strideW; d++)
				{
					for (int e = c; e < c + strideH; e++)
					{
						maxVal = max(in->tensor[a][d][e], maxVal);
					}
				}

				out->tensor[a][b / strideW].push_back(maxVal);
			}
		}
	}

	return out;
}

void FlattenLayer::loadLayer(string const& data) {}

Data* FlattenLayer::computeOutput(Data* input)
{
	ConvData* in = (ConvData*)input;
	DenseData* out = new DenseData();

	out->data.reserve(in->tensor.size() * in->tensor[0].size() * in->tensor[0][0].size());

	int curIndex = 0;

	for (int a = 0; a < in->tensor[0].size(); a++)
	{
		for (int b = 0; b < in->tensor[0][0].size(); b++)
		{
			for (int c = 0; c < in->tensor.size(); c++)
			{
				out->data.push_back(in->tensor[c][a][b]);
			}
		}
	}
	return out;
}

void ConvData::loadFromString(string const& data) 
{
	std::istringstream iss(data);
	string sOut;

	tensor.push_back(matrix());

	while (std::getline(iss, sOut))
	{
		if (sOut[0] == '[')
		{
			vector<float> row;
			remove(sOut, '['); remove(sOut, ']');
			parseArray(sOut, row);

			tensor[0].push_back(row);
		}
	}
}
void DenseData::loadFromString(string const& data) { }

void Model::loadModel(string const& modelPath)
{
	string data, line;
	getFileContents(modelPath.c_str(), data);

	std::istringstream iss;
	std::ostringstream layerStrBdr;

	iss.str(data);

	int buildID = -1;

	while (std::getline(iss, line))
	{
		if (FIND("layer", line))
		{
			switch (buildID)
			{
				case 1:
				{
					Conv2DLayer* layer = new Conv2DLayer();
					layer->loadLayer(layerStrBdr.str());
					layers.emplace_back(layer);
					break;
				}
				case 2:
				{
					DenseLayer* layer = new DenseLayer();
					layer->loadLayer(layerStrBdr.str());
					layers.emplace_back(layer);
					break;
				}
				case 3:
				{
					MaxPooling2DLayer* layer = new MaxPooling2DLayer();
					layer->loadLayer(layerStrBdr.str());
					layers.emplace_back(layer);
					break;
				}
				case 4:
				{
					FlattenLayer* layer = new FlattenLayer();
					layers.emplace_back(layer);
					break;
				}
			}
			layerStrBdr.str(string());

			if (line.substr(6) == "Conv2D")
			{
				buildID = 1;
			}
			else if (line.substr(6) == "Dense")
			{
				buildID = 2;
			}
			else if (line.substr(6) == "MaxPooling2D")
			{
				buildID = 3;
			}
			else if (line.substr(6) == "Flatten")
			{
				buildID = 4;
			}
			else
			{
				buildID = -1;
			}
		}
		else
		{
			layerStrBdr << line << '\n';
		}
	}

	switch (buildID)
	{
		case 1:
		{
			Conv2DLayer* layer = new Conv2DLayer();
			layer->loadLayer(layerStrBdr.str());
			layers.emplace_back(layer);
			break;
		}
		case 2:
		{
			DenseLayer* layer = new DenseLayer();
			layer->loadLayer(layerStrBdr.str());
			layers.emplace_back(layer);
			break;
		}
		case 3:
		{
			MaxPooling2DLayer* layer = new MaxPooling2DLayer();
			layer->loadLayer(layerStrBdr.str());
			layers.emplace_back(layer);
			break;
		}
		case 4:
		{
			FlattenLayer* layer = new FlattenLayer();
			layers.emplace_back(layer);
			break;
		}
	}
}

Data* Model::predict(Data* input)
{
	Data* next = input;
	Data* prev = nullptr;
	for (Layer* l : layers)
	{
		next = l->computeOutput((prev = next));
		if (prev != input)
		{
			delete prev;
		}
	}

	return next;
}