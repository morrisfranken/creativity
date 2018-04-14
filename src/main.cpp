
#include <iostream>
#include "average.h"
#include "dominantColour.h"
#include "dominantTransition.h"

using namespace std;

int main(int argc, char* argv[]) {
//	dominant_colour::run("/home/morris/Downloads/images", "../cache");
    dominant_colour::runCached("../cache");
//    dominantTransition::run("../results/dominant");
	return 0;
}
