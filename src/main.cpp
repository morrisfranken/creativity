
#include <iostream>
#include "average.h"
#include "dominantColour.h"
#include "dominantTransition.h"

using namespace std;

int main(int argc, char* argv[]) {
//	dominant_colour::computeColors("/home/morris/Downloads/images", "../cache");
    dominant_colour::runDominantExtaction("../cache");
//    dominantTransition::computeColors("../results/dominant");
	return 0;
}
