/*
 * BinReader.h
 *
 * Read files that have been saved using the BinWriter class.
 *
 *  Created on: 12 May 2014
 *      Author: morris
 */

#ifndef BIN_READER_H
#define BIN_READER_H

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <cstdint>

char *readFile(char *filename, int &size);

class BinReader {
protected:
	char *data_begin_;
	char *data_;
	char *data_end_;
	int size_;

	inline void movePointer(const size_t amount) {
		data_ += amount;
		if (data_ > data_end_)
			throw(std::runtime_error("BinReader : Reached end of file prematurely")); // This can happen when the file is corrupted, or incorrectly read
	}

public:
	virtual ~BinReader();
	BinReader(std::string path);
	BinReader(const BinReader &me) = delete;	// Do not allow copy constructors, as this might result in data being deleted twice!

	uint32_t readInt32();
	bool readBool();
	double readDouble();
	std::vector<uchar> readBytes();
	std::string readString();
	cv::Mat readMat();      // Read OpenCV matrix without any compression

	/* Read data that was saved using the binWriter appendUnsafe function
	 */
	template <class T> T readUnsafe() {
		uint32_t size_of = readInt32();
		if (sizeof(T) != size_of) {
			throw(std::runtime_error("BinReader::readUnsafe : size of template argument does not match the size of stored vector elements"));
		}
		T res;
		memcpy((uchar *)&res, (uchar *)data_, sizeof(T));
		movePointer(sizeof(T));
		return res;
//			return *(T*)(data_ - sizeof(T)); // Since Eigen does not work well using this method due to memory not being aligned (and Eigen overloads copy operator), we are forced to either use memcpy or create a separate function for loading Eigen matrices, which will also add the Eigen dependency. Therefore it is better to use memcpy, did not show any differences after empirical testings.
	}

	// The difference with readVector<uchar>() and readBytes is that the suzeof(uchar) is also stored
	// Note that T should be serializable!
	template <class T> std::vector<T> readVector() {
		uint32_t size_of = readInt32();
		uint32_t size = readInt32();
		if (sizeof(T) != size_of)
			throw(std::runtime_error("BinReader::readVector : size of template argument does not match the size of stored vector elements"));
		movePointer(size * sizeof(T));
		std::vector<T> data((T *)data_ - size, (T *)data_);
		return data;
	}
};

#endif /* BIN_READER_H */
