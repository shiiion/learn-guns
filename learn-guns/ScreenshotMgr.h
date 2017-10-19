#pragma once

#include "res.h"

struct HSV
{
	float hue, sat, val;

	HSV() { hue = sat = val = 0; }
	HSV(float r, float g, float b);
};

bool takeScreenshot(vector<vector<HSV>>& directDataOut, vector<vector<float>>& dataOut);

//3 channels assumed
void convertScreenshot(byte* screenshotData, int width, int height, vector<vector<HSV>>& actualImg, vector<vector<float>>& output);


bool outOfAmmo(vector<vector<HSV>>& imageHSV);