
#include <iostream>
#include <fstream>
#include <string>
#include "NNLayers.h"
#include <ctime>
#include "ScreenshotMgr.h"
#include <Windows.h>


string nameOfGun(int id)
{
	string names[] = { "AK47", "HK416", "Kar98k", "M16A4", "Scar-L", "SKS", "UMP", "Uzi", "Vector" };
	return names[id];
}

int main()
{
	Model m;
	m.loadModel("dumpedcnn.nn");
	vector<vector<float>> screen;
	ConvData data;
	while (true)
	{
		takeScreenshot(screen);
		data.loadFromImage(screen);
		Data* result = m.predict(&data);
		DenseData* denseResult = (DenseData*)result;
		int gunCounter = 0;
		int predictedgun = 0;
		system("cls");
		float maxx = 0;
		for (float f : denseResult->data)
		{
			std::cout << "Prediction for " << nameOfGun(gunCounter) << ": " << f << std::endl;
			if (f > maxx)
			{
				predictedgun = gunCounter;
				maxx = f;
			}
			gunCounter++;
		}
		if (maxx < 0.95)
		{
			std::cout << "Cant guess the gun";
		}
		else
		{
			std::cout << "I think the gun is " << nameOfGun(predictedgun);
		}
		delete result;
		Sleep(500);

		
		screen.clear();
	}
}