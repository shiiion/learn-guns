#include "NetworkInterface.h"
#include "NNLayers.h"
#include "ScreenshotMgr.h"
#include <Windows.h>

static Model* gunModel = nullptr;

bool StartupNetwork()
{
	if (gunModel != nullptr)
	{
		delete gunModel;
	}
	gunModel = new Model();

	char fileName[MAX_PATH];
	GetModuleFileName(NULL, fileName, MAX_PATH);
	gunModel->loadModel(fileName);

	return gunModel->getNumLayers() == 5;
}

bool Predict(vector<float>& outNetworkPrediction)
{
	outNetworkPrediction.clear();
	matrix screen;
	vector<vector<HSV>> imageHSV;
	ConvData screenData;

	takeScreenshot(imageHSV, screen);
	if (outOfAmmo(imageHSV))
	{
		return false;
	}

	screenData.loadFromImage(screen);
	DenseData* result = (DenseData*)gunModel->predict(&screenData);
	
	for (float f : result->data)
	{
		outNetworkPrediction.emplace_back(f);
	}

	delete result;
	return true;
}