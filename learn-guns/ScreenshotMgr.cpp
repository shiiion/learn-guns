#include "ScreenshotMgr.h"
#include <Windows.h>


HSV::HSV(float r, float g, float b)
{
	float Cmax = max(r, max(g, b));
	float Cmin = min(r, min(g, b));
	
	float delta = Cmax - Cmin;

	if (abs(Cmax - r) < 0.00001f)
	{
		hue = 60.f * fmod((g - b) / delta, 6.f);
	}
	else if (abs(Cmax - g) < 0.00001f)
	{
		hue = 60.f * (((b - r) / delta) + 2);
	}
	else
	{
		hue = 60.f * (((r - g) / delta) + 4);
	}
	hue *= (PI / 180.f);

	sat = (Cmax <= 0 ? 0 : (delta / Cmax));

	val = Cmax;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~helper~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

float grayscale(byte r, byte g, byte b)
{
	float rf = (float)r, gf = (float)g, bf = (float)b;

	float lum = 0.299f * rf + 0.587f * gf + 0.114f * bf;
	return clamp(0.f, 255.f, roundf(lum));
}

PBITMAPINFO CreateBitmapInfoStruct(HBITMAP hBmp)
{
	BITMAP bmp;
	PBITMAPINFO pbmi;
	WORD    cClrBits;

	// Retrieve the bitmap color format, width, and height.  
	if (!GetObject(hBmp, sizeof(BITMAP), (LPSTR)&bmp))
		return nullptr;

	// Convert the color format to a count of bits.  
	cClrBits = (WORD)(bmp.bmPlanes * bmp.bmBitsPixel);
	if (cClrBits == 1)
		cClrBits = 1;
	else if (cClrBits <= 4)
		cClrBits = 4;
	else if (cClrBits <= 8)
		cClrBits = 8;
	else if (cClrBits <= 16)
		cClrBits = 16;
	else if (cClrBits <= 24)
		cClrBits = 24;
	else cClrBits = 32;

	// Allocate memory for the BITMAPINFO structure. (This structure  
	// contains a BITMAPINFOHEADER structure and an array of RGBQUAD  
	// data structures.)  

	if (cClrBits < 24)
		pbmi = (PBITMAPINFO)LocalAlloc(LPTR,
			sizeof(BITMAPINFOHEADER) +
			sizeof(RGBQUAD) * (1 << cClrBits));

	// There is no RGBQUAD array for these formats: 24-bit-per-pixel or 32-bit-per-pixel 

	else
		pbmi = (PBITMAPINFO)LocalAlloc(LPTR,
			sizeof(BITMAPINFOHEADER));

	// Initialize the fields in the BITMAPINFO structure.  

	pbmi->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	pbmi->bmiHeader.biWidth = bmp.bmWidth;
	pbmi->bmiHeader.biHeight = bmp.bmHeight;
	pbmi->bmiHeader.biPlanes = bmp.bmPlanes;
	pbmi->bmiHeader.biBitCount = bmp.bmBitsPixel;
	if (cClrBits < 24)
		pbmi->bmiHeader.biClrUsed = (1 << cClrBits);

	// If the bitmap is not compressed, set the BI_RGB flag.  
	pbmi->bmiHeader.biCompression = BI_RGB;

	// Compute the number of bytes in the array of color  
	// indices and store the result in biSizeImage.  
	// The width must be DWORD aligned unless the bitmap is RLE 
	// compressed. 
	pbmi->bmiHeader.biSizeImage = ((pbmi->bmiHeader.biWidth * cClrBits + 31) & ~31) / 8
		* pbmi->bmiHeader.biHeight;
	// Set biClrImportant to 0, indicating that all of the  
	// device colors are important.  
	pbmi->bmiHeader.biClrImportant = 0;
	return pbmi;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~real code~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void convertScreenshot(byte* screenshotData, int width, int height, vector<vector<HSV>>& actualImg, vector<vector<float>>& output)
{
	vector<int> aa;
	for (int a = 0; a < width * height * 4; a++)
	{
		aa.push_back(screenshotData[a]);
	}

	//setup
	output.reserve(64);
	byte*** dst = new byte**[64];
	for (int a = 0; a < 64; a++)
	{
		dst[a] = new byte*[64];
		for (int b = 0; b < 64; b++)
		{
			dst[a][b] = new byte[3];
		}
	}
	actualImg.reserve(height);
	for (int a = 0; a < height; a++)
	{
		actualImg.emplace_back(vector<HSV>());
		for (int b = 0; b < width; b++)
		{
			byte r = screenshotData[((height - 1 - a) * width * 4) + (b * 4) + 2];
			byte g = screenshotData[((height - 1 - a) * width * 4) + (b * 4) + 1];
			byte bb = screenshotData[((height - 1 - a) * width * 4) + (b * 4)];

			HSV hsv = HSV((float)r / 255.f, (float)g / 255.f, (float)bb / 255.f);
			actualImg[a].emplace_back(hsv);
		}
	}
	//nearest-neighbor downsample

	float xScale = (float)width / 64.0f, yScale = (float)height / 64.0f;

	for (int a = 0; a < 64; a++)
	{
		for (int b = 0; b < 64; b++)
		{
			int srcYIndex = (int)roundf((float)a * yScale);
			int srcXIndex = (int)roundf((float)b * xScale);

			srcYIndex = clamp(0, height - 1, srcYIndex);
			srcXIndex = clamp(0, width - 1, srcXIndex);

			int i1d = ((height - 1 - srcYIndex) * width * 4) + (srcXIndex * 4);
			byte r = screenshotData[i1d + 2];
			byte g = screenshotData[i1d + 1];
			byte bb = screenshotData[i1d];
			dst[a][b][0] = screenshotData[i1d + 2];
			dst[a][b][1] = screenshotData[i1d + 1];
			dst[a][b][2] = screenshotData[i1d];
		}
	}

	//grayscale
	for (int a = 0; a < 64; a++)
	{
		output.push_back(vector<float>());
		for (int b = 0; b < 64; b++)
		{
			byte* px = dst[a][b];
			output[a].push_back(grayscale(px[0], px[1], px[2]));
		}
	}

	//cleanup
	for (int a = 0; a < 64; a++)
	{
		for (int b = 0; b < 64; b++)
		{
			delete[] dst[a][b];
		}
		delete[] dst[a];
	}

	delete[] dst;
}

bool takeScreenshot(vector<vector<HSV>>& directDataOut, vector<vector<float>>& dataOut)
{
	RECT desktopSize;
	HWND hDesktop = GetDesktopWindow();
	GetWindowRect(hDesktop, &desktopSize);

	int w = desktopSize.right - desktopSize.left;
	int h = desktopSize.bottom - desktopSize.top;
	double wRatio = (double)(w) / 1920.0;
	double hRatio = (double)(h) / 1080.0;
	int scaledW = (int)roundf(wRatio * 200.f);
	int scaledH = (int)roundf(hRatio * 66.f);
	int scaledX = (w / 2) - (scaledW / 2);
	int scaledY = h - (int)roundf(136.0f * hRatio);


	HDC desktopDC = GetDC(hDesktop);
	HDC captureDC = CreateCompatibleDC(desktopDC);
	HBITMAP captureBMP = CreateCompatibleBitmap(desktopDC, scaledW, scaledH);
	SelectObject(captureDC, captureBMP);

	BitBlt(captureDC, 0, 0, w, h, desktopDC, scaledX, scaledY, SRCCOPY | CAPTUREBLT);

	auto bmpInfo = CreateBitmapInfoStruct(captureBMP);
	auto bmpHdr = (PBITMAPINFOHEADER)bmpInfo;
	if (bmpInfo == nullptr) return false;
	
	LPBYTE image = (LPBYTE)GlobalAlloc(GMEM_FIXED, bmpHdr->biSizeImage);
	if (image == nullptr) return false;

	if (!GetDIBits(captureDC, captureBMP, 0, (WORD)bmpHdr->biHeight, image, (PBITMAPINFO)bmpInfo, DIB_RGB_COLORS))
	{
		GlobalFree((HGLOBAL)image);
		return false;
	}

	convertScreenshot((byte*)image, bmpHdr->biWidth, bmpHdr->biHeight, directDataOut, dataOut);

	GlobalFree((HGLOBAL)image);

	ReleaseDC(hDesktop, desktopDC);
	DeleteDC(captureDC);
	DeleteObject(captureBMP);

	return true;
}

bool outOfAmmo(vector<vector<HSV>>& imageHSV)
{
	const float MIN_PERCENT = 1.f / 11.f;
	int pixelCounter = 0;

	for (int a = 0; a < imageHSV.size(); a++)
	{
		for (int b = 0; b < imageHSV[a].size(); b++)
		{
			HSV const& hsv = imageHSV[a][b];
			float clampedRedHue = max(pow(hsv.hue - PI, 2) - (PI * PI - 1), 0);

			if (clampedRedHue > 0 && hsv.sat > 0.8f && hsv.val > 0.3f)
			{
				pixelCounter++;
			}
		}
	}
	
	float percentage = (float)pixelCounter / (float)(imageHSV.size() * imageHSV[0].size());
	return percentage > MIN_PERCENT;
}