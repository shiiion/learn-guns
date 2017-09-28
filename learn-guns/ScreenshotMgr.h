#pragma once

#include "res.h"

bool takeScreenshot(vector<vector<float>>& dataOut);

//3 channels assumed
void convertScreenshot(byte* screenshotData, int width, int height, vector<vector<float>>& output);
