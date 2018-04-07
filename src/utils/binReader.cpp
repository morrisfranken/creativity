/*
 * BinReader.cpp
 *
 *  Created on: 12 May 2014
 *      Author: morris
 */

#include <stdexcept>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv2/imgcodecs.hpp>

#include "binReader.h"

using namespace std;
using namespace cv;

// After several benchmarks, this function was chosen as the fastest way to read a file at once
char *readFile(char *filename, int &size) {
	ifstream file (filename, ios::in | ios::binary | ios::ate);
	size = (int)file.tellg();
	file.seekg (0, ios::beg);
	char *data = (char *)malloc(size);
	file.read (data, size);
	file.close();
	return data;
}

// BinReader
BinReader::~BinReader() {
	free(data_begin_);
}

BinReader::BinReader(string path) {
	data_begin_ = readFile(&path[0], size_);
	data_ = data_begin_;
	data_end_ = data_ + size_;
	if (size_ <= 0)
		throw(runtime_error("could not read "+ path));
}

uint32_t BinReader::readInt32() {
	uint32_t output = *((uint32_t *)data_);
	movePointer(sizeof(uint32_t));
	return output;
}

bool BinReader::readBool() {
	bool output = *((bool *)data_);
	movePointer(sizeof(bool));
	return output;
}

double BinReader::readDouble() {
	double number = strtod(data_, &data_);
	movePointer(1);
	return number;
}

vector<uchar> BinReader::readBytes() {
	uint32_t size = *((uint32_t *)data_);
	movePointer(sizeof(size) + size);
	vector<uchar> bytes(data_ - size, data_);
	return bytes;
}

std::string BinReader::readString() {
	string output(data_);
	movePointer(output.size() + 1);
	return output;
}

Mat BinReader::readMat() {
	uint32_t type = readInt32();
	uint32_t rows = readInt32();
	uint32_t cols = readInt32();

	Mat mat(rows, cols, type, data_); // TODO(morris) : validate size with height/width * sizeof element? --> mat.total() * mat.elemSize();
	movePointer((char *)mat.dataend - data_);

	return mat.clone();  // copy is necessary because we expect the bytes read from the file to be freed, otherwise it would result in memory leak
}
