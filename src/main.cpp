
#include <iostream>
#include "average.h"
#include "dominantColour.h"

using namespace std;

int main(int argc, char* argv[]) {
//	dominant_colour::run("/home/morris/Downloads/images", "../cache");
    dominant_colour::runCached("../cache");
	return 0;
}
