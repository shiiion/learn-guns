#pragma once

#include <string>
#include <vector>
#include <sstream>

using std::string;
using std::vector; 

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define FIND(s1, s2) (s2.find(s1) != string::npos)
#define clamp(minval, maxval, val) (max(min((val), (maxval)), (minval)))


typedef vector<vector<float>> matrix;
typedef unsigned __int8 byte;