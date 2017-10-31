#include "NNLayers.h"
#include <sstream>
#include <algorithm>
#include <ctime>
#include <limits>
#include <iostream>
#include <fstream>

void getFileContents(const char *filename, string& contents)
{
	FILE* fp;
	fp = fopen(filename, "rb");
	
	if (fp != nullptr)
	{
		fseek(fp, 0, SEEK_END);
		long size = ftell(fp);
		rewind(fp);

		contents.resize(size);
		fread(&contents[0], 1, size, fp);
		fclose(fp);
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

void convolve(matrix const& kernel, matrix const& data, matrix& out)
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
			out[a].push_back(sum);
		}
	}
}



void Conv2DLayer::loadLayer(string const& layerString)
{
	std::istringstream iss(layerString);
	string sOut;
	int depthStep = 0;
	int colStride = 0;
	int rowStride = 0;
	
	vector<vector<vector<vector<float>>>> kernelsTemp;

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

			if (kernelsTemp.size() == colStride)
			{
				kernelsTemp.push_back(vector<vector<vector<float>>>());
			}

			if (kernelsTemp[colStride].size() == rowStride)
			{
				kernelsTemp[colStride].push_back(vector<vector<float>>());
			}

			kernelsTemp[colStride][rowStride].emplace_back(kernel);

			if (depthStep == (z - 1))
			{
				depthStep = -1; rowStride++;
			}
			if (rowStride == y)
			{
				rowStride = 0; colStride++;
			}

			depthStep++;
		}
		else
		{
			vector<string> dims;
			split(sOut, ',', dims);
			x = std::stoi(dims[0]); y = std::stoi(dims[1]);
			z = std::stoi(dims[2]); k = std::stoi(dims[3]);
		}
	}
	for (int a = 0; a < k; a++)
	{
		kernels.push_back(vector<vector<vector<float>>>());
		for (int b = 0; b < z; b++)
		{
			kernels[a].push_back(vector<vector<float>>());
			for (int c = 0; c < y; c++)
			{
				kernels[a][b].push_back(vector<float>());
			}
		}
	}
	//std::cout << kernelsTemp.size() << " " << kernelsTemp[0].size() << " " << kernelsTemp[0][0].size() << " " << kernelsTemp[0][0][0].size() << std::endl;

	for (int a = 0; a < kernelsTemp.size(); a++)
	{
		for (int b = 0; b < kernelsTemp[a].size(); b++)
		{
			for (int c = 0; c < kernelsTemp[a][b].size(); c++)
			{
				for (int d = 0; d < kernelsTemp[a][b][c].size(); d++)
				{
					kernels[d][c][b].push_back(kernelsTemp[a][b][c][d]);
				}
			}
		}
	}
	//std::cout << kernels.size() << " " << kernels[0].size() << " " << kernels[0][0].size() << " " << kernels[0][0][0].size() << std::endl;

}

void Conv2DLayer::loadLayer(byte* layer, int len)
{
	int kernelStep = 0;
	int depthStep = 0;
	int colStride = 0;
	int rowStride = 0;

	vector<vector<vector<vector<float>>>> kernelsTemp;

	x = (((int)layer[3] & 0xFF) << 24) | (((int)layer[2] & 0xFF) << 16) | (((int)layer[1] & 0xFF) << 8) | (((int)layer[0] & 0xFF));
	y = (((int)layer[7] & 0xFF) << 24) | (((int)layer[6] & 0xFF) << 16) | (((int)layer[5] & 0xFF) << 8) | (((int)layer[4] & 0xFF));
	z = (((int)layer[11] & 0xFF) << 24) | (((int)layer[10] & 0xFF) << 16) | (((int)layer[9] & 0xFF) << 8) | (((int)layer[8] & 0xFF));
	k = (((int)layer[15] & 0xFF) << 24) | (((int)layer[14] & 0xFF) << 16) | (((int)layer[13] & 0xFF) << 8) | (((int)layer[12] & 0xFF));
	layer += 16;

	char activationName[32];
	strncpy(activationName, (char*)layer, 32);
	activation = activationName;
	layer += 32;

	for (int a = 0; a < len - 48;)
	{
		if (a >= (x * y * z * k * 4))
		{
			for (int start = a; a < (start + (4 * k)); a += 4)
			{
				float val;
				memcpy(&val, (layer + a), 4);
				biases.emplace_back(val);
			}
			break;
		}
		else
		{
			vector<float> kernel;
			for (int start = a; a < (start + (4 * k)); a += 4)
			{
				float val;
				memcpy(&val, (layer + a), 4);
				kernel.emplace_back(val);
			}
			if (kernelsTemp.size() == colStride)
			{
				kernelsTemp.push_back(vector<vector<vector<float>>>());
			}

			if (kernelsTemp[colStride].size() == rowStride)
			{
				kernelsTemp[colStride].push_back(vector<vector<float>>());
			}

			kernelsTemp[colStride][rowStride].emplace_back(kernel);

			if (depthStep == (z - 1))
			{
				depthStep = -1; rowStride++;
			}
			if (rowStride == y)
			{
				rowStride = 0; colStride++;
			}
			depthStep++;
		}
	}

	for (int a = 0; a < k; a++)
	{
		kernels.push_back(vector<vector<vector<float>>>());
		for (int b = 0; b < z; b++)
		{
			kernels[a].push_back(vector<vector<float>>());
			for (int c = 0; c < y; c++)
			{
				kernels[a][b].push_back(vector<float>());
			}
		}
	}

	for (int a = 0; a < kernelsTemp.size(); a++)
	{
		for (int b = 0; b < kernelsTemp[a].size(); b++)
		{
			for (int c = 0; c < kernelsTemp[a][b].size(); c++)
			{
				for (int d = 0; d < kernelsTemp[a][b][c].size(); d++)
				{
					kernels[d][c][b].push_back(kernelsTemp[a][b][c][d]);
				}
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
		matrix finalResult;
		for (int b = 0; b < kernels[a].size(); b++)
		{
			matrix temp;
			convolve(kernels[a][b], in->tensor[b], temp);
			if (finalResult.size() == 0)
			{
				finalResult = temp;
			}
			else
			{
				for (int c = 0; c <temp.size(); c++)
				{
					for (int d = 0; d < temp[c].size(); d++)
					{
						finalResult[c][d] += temp[c][d];
					}
				}
			}
		}
		for (int c = 0; c <finalResult.size(); c++)
		{
			for (int d = 0; d < finalResult[c].size(); d++)
			{
				finalResult[c][d] += biases[a];
			}
		}
		activation2d(activation, finalResult);
		out->tensor.emplace_back(finalResult);
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

void DenseLayer::loadLayer(byte* layer, int len)
{
	memcpy(&inSize, layer, 4);
	memcpy(&outSize, layer + 4, 4);
	layer += 8;

	char activationName[32];
	strncpy(activationName, (char*)layer, 32);
	activation = activationName;
	layer += 32;
	int a = 0;
	for (int neuron = 0; neuron < inSize; neuron++)
	{
		vector<float> outputWeights;
		for (int start = a; a < (start + (outSize * 4)); a += 4)
		{
			float val;
			memcpy(&val, layer + a, 4);
			outputWeights.emplace_back(val);
		}
		weights.emplace_back(outputWeights);
	}
	for (int start = a;a<(start + (outSize * 4)); a += 4)
	{
		float val;
		memcpy(&val, layer + a, 4);
		biases.emplace_back(val);
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

void MaxPooling2DLayer::loadLayer(byte* layer, int len)
{
	memcpy(&strideW, layer, 4);
	memcpy(&strideH, layer + 4, 4);
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

void Model::loadModel(string const& modelPath, bool byteStream)
{
	if (byteStream)
	{
		loadModel_byte(modelPath);
	}
	else
	{
		loadModel_string(modelPath);
	}
}

void Model::loadModel_string(string const& modelPath)
{
	string data, line;
	getFileContents(modelPath.c_str(), data);

	long index = data.rfind("npettisgay") + 10;
	data = data.substr(index);

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

void Model::loadModel_byte(string const& modelPath)
{
	string data, line;
	getFileContents(modelPath.c_str(), data);
	long index = data.rfind("npettisgay") + 10;
	data = data.substr(index);
	int len = data.size();

	byte* dataByte = new byte[len];
	memcpy(dataByte, &data[0], len);
	int segmentSize;
	unsigned int curPos = 0;
	while (curPos < len)
	{
		memcpy(&segmentSize, (dataByte + curPos), 4);
		curPos += 4;

		byte* dataSegmentByte = new byte[segmentSize];
		memcpy(dataSegmentByte, (dataByte + curPos), segmentSize);

		segmentSize -= 16;
		curPos += 16;
		Layer* newLayer = nullptr;

		if (!strncmp((char*)dataSegmentByte, "Conv2D", 16))
		{
			newLayer = new Conv2DLayer();
		}
		else if (!strncmp((char*)dataSegmentByte, "Dense", 16))
		{
			newLayer = new DenseLayer();
		}
		else if (!strncmp((char*)dataSegmentByte, "Flatten", 16))
		{
			newLayer = new FlattenLayer();
		}
		else if (!strncmp((char*)dataSegmentByte, "MaxPooling2D", 16))
		{
			newLayer = new MaxPooling2DLayer();
		}

		if (newLayer != nullptr)
		{
			newLayer->loadLayer((dataSegmentByte + 16), segmentSize);
			layers.emplace_back(newLayer);
		}
		curPos += segmentSize;
		delete[] dataSegmentByte;
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