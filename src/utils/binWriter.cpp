/*
 * BinWriter.cpp
 *
 *  Created on: 12 May 2014
 *      Author: morris
 */

#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv2/imgcodecs.hpp>

#include "binWriter.h"

using namespace std;
using namespace cv;

BinWriter::~BinWriter() {
	output_stream_.close();
}

BinWriter::BinWriter(string path) {
	output_stream_.open(&path[0], ios::binary);
}

void BinWriter::close() {
	output_stream_.close();
}

void BinWriter::appendInt32(const uint32_t &number) {
	output_stream_.write((char *)&number, sizeof(number));
}

void BinWriter::appendDouble(const double &number) {
	output_stream_ << number << '\0';
}

void BinWriter::appendBytes(const char* data, const uint32_t size) {
	appendInt32(size);
	output_stream_.write(data, size);
}

void BinWriter::appendString(const std::string &str) {
//	appendInt32(str.size())		// this would make it incompatible with existing saves
//	output_stream_.write(str.c_str(), str.size() + 1);
	output_stream_ << str << '\0';
}

void BinWriter::appendMat(const Mat &mat) {
	Mat temp = mat.isContinuous()? mat : mat.clone(); // ensure Mat is continuous, TODO(morris) : this can be optimised by writing line by line instead of cloning the entire mat
	uint32_t type = temp.type();
	uint32_t rows = temp.rows;
	uint32_t cols = temp.cols;
	uint32_t sizeofMat = (temp.dataend - temp.data);

	appendInt32(type);
	appendInt32(rows);
	appendInt32(cols);
	output_stream_.write((char *)temp.data, sizeofMat); // elements from the OpenCV matrix are the same on all platforms: CV_8U, CV_32S, CV_32F, etc... all have a predefined size regardless of platform, therefore we can simply dump the memory to file and retrieve it later
}
