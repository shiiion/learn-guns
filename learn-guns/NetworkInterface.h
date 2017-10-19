#pragma once
#include "res.h"

//97% suggested (subject to change)
#define PREDICTION_MIN_THRESHOLD 0.97f

enum GunIndex
{
	AK47=0,
	AWM=1,
	Beretta686=2,
	Glock=3,
	Groza=4,
	HK416=5,
	Kar98k=6,
	M16A4=7,
	M1911=8,
	M249=9,
	M24=10,
	M9=11,
	Mini14=12,
	Mk14=13,
	Nagant=14,
	SCARL=15,
	SKS=16,
	Saiga12=17,
	Thompson=18,
	UMP=19,
	Uzi=20,
	VSS=21,
	Vector=22,
	Winchester=23
};

//~5-10 second startup
bool StartupNetwork();

//output: 24 element vector
bool Predict(vector<float>& outNetworkPrediction);