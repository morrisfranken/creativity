//
// Created by morris on 2/28/18.
//

#include <iostream>
#include <dirent.h>

#include "utils.h"

using namespace std;

vector<string> my_utils::listdir(const std::string &path) {
    struct dirent *ent;
    DIR *dir = opendir (path.c_str());
    vector<string> files;

    if (dir) {
        while ((ent = readdir(dir))) {
            if (ent->d_type == DT_REG)
                files.emplace_back(path + "/" + ent->d_name);
        }
        closedir (dir);
    } else {
        cerr << "no such directory: " << path << endl;
    }

    return files;
}