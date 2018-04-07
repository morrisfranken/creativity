/*
 * BinWriter.h
 *
 * BinWriter can be used to store data in a binary format. This is an efficient way of storing numbers, as there is no need to convert anything when writing to disk, or reading. It is basically a memory dump.
 * BinWriter offers a number of functions for saving default data types (and these will work among all platforms), but it can also be used to store custom serialised data types using the appendUnsafe and appendVector functions
 * The files that have been saved using BinWriter can be read using the BinReader class
 *  Created on: 12 May 2014
 *      Author: morris
 *
 *  For saving c++ data to a binary file
 */

#ifndef BIN_WRITER_H_
#define BIN_WRITER_H_

#include <fstream>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <cstdint>

class BinWriter {
protected:
	std::ofstream output_stream_;
public:
	virtual ~BinWriter();
	BinWriter(std::string path);
	BinWriter(const BinWriter &me) = delete;	// std::ofstream may not be copied, use reference for passing this as argument

	void close();

	void appendInt32(const uint32_t &number);	// uint32_t will always be 4 bytes, regardless of platform
	void appendDouble(const double &number);	// store double as string, so it is 100% platform independent -> actually, according to http://stackoverflow.com/questions/1101463/the-double-byte-size-in-32-bit-and-64-bit-os, double is ALWAYS 64 bits
	void appendBytes(const char* data, const uint32_t size);
	void appendString(const std::string &str);
	void appendMat(const cv::Mat &mat);

	// Warning! This might behave different on different platforms, use with caution!
	// Warning! Use only serialised input!
	template <class T>  void appendUnsafe(const T &input) {
		appendInt32(sizeof(T)); // for validating compatibility -> should I keep this? It is practically useless, and bloats the filesize if used often... only advantage is to show a dedicated error message when reading using wrong template
		output_stream_.write((char *)&input, sizeof(input));
	}

	// Warning!, this might behave different on different platforms, use with caution!
	template <class T> void appendVector(const std::vector<T> &vec) {
		const uint32_t size = vec.size();
		appendInt32(sizeof(T)); // for validating compatibility -> should I keep this? (if not I can merge readBytes with readVector<char>)
		appendInt32(size);
		output_stream_.write((char *)&vec[0], sizeof(T) * size);
	}
};

#endif /* BIN_WRITER_H_ */
