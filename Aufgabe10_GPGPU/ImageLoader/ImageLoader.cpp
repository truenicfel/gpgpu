#include "ImageLoader.h"
#include <FreeImage.h>
#include <stdexcept>
namespace img
{
	ImageLoader::ImageLoader()
	{
#ifdef FREEIMAGE_LIB
		FreeImage_Initialise();
#endif
	}
	ImageLoader::~ImageLoader()
	{
#ifdef FREEIMAGE_LIB
		FreeImage_DeInitialise();
#endif
	}
	unsigned char* ImageLoader::LoadTextureFromFile(const std::string& fileName, int * widthOut, int* heightOut, bool topDown)
	{
		//image format
		FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
		//pointer to the image, once loaded
		FIBITMAP *dib(0);
		//pointer to the image data
		BYTE* bits(0);
		//image width and height
		unsigned int width(0), height(0);

		//check the file signature and deduce its format
		fif = FreeImage_GetFileType(fileName.c_str(), 0);
		//if still unknown, try to guess the file format from the file extension
		if(fif == FIF_UNKNOWN) 
			fif = FreeImage_GetFIFFromFilename(fileName.c_str());
		//if still unkown, return failure
		if(fif == FIF_UNKNOWN)
			throw std::runtime_error("Unknown image file!");;

		//check that the plugin has reading capabilities and load the file
		if(FreeImage_FIFSupportsReading(fif))
			dib = FreeImage_Load(fif, fileName.c_str());
		//if the image failed to load, return failure
		if(!dib)
			throw std::runtime_error("Failed to load image!");

		//retrieve the image data
		bits = FreeImage_GetBits(dib);
		//get the image width and height
		width = FreeImage_GetWidth(dib);
		height = FreeImage_GetHeight(dib);
		//if this somehow one of these failed (they shouldn't), return failure
		if((bits == 0) || (width == 0) || (height == 0))
			throw std::runtime_error("Failed to load image!");

		unsigned bpp = FreeImage_GetBPP(dib) ;
		int pitch = FreeImage_GetPitch(dib) ;

		unsigned char* data = new unsigned char[width*height*bpp] ;

		// this is necessary to store the texture data in the reverse
		// order. FreeImage lib loads the texture in wired bottum up style.
		FreeImage_ConvertToRawBits((BYTE*)data, dib, pitch, bpp, 
			0,0,0, topDown) ;

		//Free FreeImage's copy of the data
		FreeImage_Unload(dib);
		if(widthOut)
			*widthOut = width;
		if(heightOut)
			*heightOut = height;
		return data;
	}
	
}
