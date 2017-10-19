
#include <iostream>
#include <fstream>
#include <string>
#include "NNLayers.h"
#include <ctime>
#include "ScreenshotMgr.h"
#include <Windows.h>


//string nameOfGun(int id)
//{
//	string names[] = { "AK47", "AWM", "Beretta686", "Glock", "Groza", "HK416", "Kar98k", "M16A4", "M1911", "M249", "M24", "M9", "Mini14", "Mk14", "NagantM1895",
//		"SCAR-L", "SKS", "Saiga12", "Thompson", "UMP", "Uzi", "VSS", "Vector", "Winchester" };
//	return names[id];
//}
//
//int main()
//{
//	Model m;
//	char fileName[MAX_PATH];
//	GetModuleFileName(NULL, fileName, MAX_PATH);
//	m.loadModel(fileName);
//
//	vector<vector<float>> screen;
//	ConvData data;
//	while (true)
//	{
//		auto t1 = std::clock();
//		takeScreenshot(screen);
//		data.loadFromImage(screen);
//		Data* result = m.predict(&data);
//		DenseData* denseResult = (DenseData*)result;
//		int gunCounter = 0;
//		int predictedgun = 0;
//		float maxx = 0;
//		for (float f : denseResult->data)
//		{
//			//std::cout << "Prediction for " << nameOfGun(gunCounter) << ": " << f << std::endl;
//			if (f > maxx)
//			{
//				predictedgun = gunCounter;
//				maxx = f;
//			}
//			gunCounter++;
//		}
//		system("cls");
//		if (maxx < 0.95)
//		{
//			std::cout << "Cant guess the gun";
//		}
//		else
//		{
//			std::cout << "I think the gun is " << nameOfGun(predictedgun) << " guess percent: " << maxx;
//		}
//		delete result;
//		//std::cout << std::clock() - t1 << std::endl;
//		Sleep(500);
//
//		
//		screen.clear();
//	}
//}

#include "networkinterface.h"
#include <ctime>
#include <iostream>
int main()
{
	StartupNetwork();

	vector<float> v;

	while (true)
	{

		if (!Predict(v))
		{
			std::cout << "out of ammo";
		}
		else
		{
			std::cout << "has ammo";
		}
		std::cout << std::endl;
		Sleep(500);
	}
}