#include "vegaDataframe.h"
#include <iostream>

int main() {
    try {
        vegaDataframe vd;
        vd.read_csv("/Users/sriramkurnella/CLionProjects/vega/data.csv");
        auto shape = vd.shape();
        std::cout << shape.first << " " << shape.second << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

}