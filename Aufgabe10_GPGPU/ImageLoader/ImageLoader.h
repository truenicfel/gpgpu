#pragma once

#include <string>
namespace img
{
	/**
	* Class to load images from file.
	Example:

	img::ImageLoader imgLoader;
	int width,height;
	//Textur einlesen. Bei JPG muss das letzte flag auf true gestellt werden
	unsigned char* data = imgLoader.LoadTextureFromFile("test.jpg",&width,&height,true);
	//Textur hochladen, JPG hat internes format BGR!
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,	0, GL_BGR, GL_UNSIGNED_BYTE, data);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR ) ;
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR ) ;
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE ) ;
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE ) ;
	delete[] data;
	*/
	class ImageLoader
	{
	public:
		ImageLoader();
		~ImageLoader();
		unsigned char* LoadTextureFromFile(const std::string& fileName, int* width = NULL, int* heigth = NULL, bool topDown = true);
	
	};
}