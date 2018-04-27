
#include <iostream>
#include <boost/filesystem.hpp>

#include "average.h"
#include "dominantColour.h"
#include "cu/kmeans.h"
#include "eigenArt.h"

using namespace std;

int main(int argc, char* argv[]) {
//	dominant_colour::computeColors("/home/morris/Downloads/images", "../cache");
//    dominant_colour::runDominantExtaction("../cache");
//    dominantTransition::computeColors("../results/dominant");
//    cu::test_kmeans();
    eigen_art::run("../data/mini", "../cache");
	return 0;
}