/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include <traact/traact.h>
#include <traact/vision.h>
#include <traact/cereal/CerealVision.h>
#include <traact/cereal/CerealSpatial.h>
#include <traact/vision/bundle_adjustment/BundleAdjustment.h>
#include "BAProblemLoader.h"
int main(int argc, char **argv) {

    using namespace traact;
    using namespace traact::vision;
    using namespace traact::vision::bundle_adjustment;
    using namespace traact::facade;

    util::initLogging(spdlog::level::debug);


    int camera_count = 6;

    std::string result_file_pattern =           "/home/frieder/data/recording_20210611_calib1/data/{0}.json";
    std::string result_camera_file_pattern =    "/home/frieder/data/recording_20210611_calib1/data/{0}_{1:02d}.json";

    std::string config_file = "/home/frieder/projects/traact_workspace/app_local/misc/inm_ba.yml";

    traact::BAProblemLoader problem_loader;
    if(!problem_loader.LoadConfig(config_file)){
        return -1;
    }


    auto bundle_adjustment = problem_loader.GetBundleAdjustment();

    if(!bundle_adjustment->CheckData()){
        return -1;
    }
    if(!bundle_adjustment->Optimize()) {
        return -1;
    }

    problem_loader.SaveResults();




    return 0;
}