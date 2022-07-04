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

    const std::string keys =
        "{help h usage ? |      | print this message   }"
        "{@config        |<none>| config file          }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Outside-In multi camera bundle adjustment v0.0.1");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    if (!parser.has("@config")) {
        SPDLOG_ERROR("missing config file");
        parser.printMessage();
        return 0;
    }


    std::string config_file = parser.get<std::string>(0);

    traact::BAProblemLoader problem_loader;
    if(!problem_loader.LoadConfig(config_file)){
        return 1;
    }


    auto bundle_adjustment = problem_loader.GetBundleAdjustment();

    if(!bundle_adjustment->CheckData()){
        return 1;
    }
    if(!bundle_adjustment->Optimize()) {
        return 1;
    }

    problem_loader.SaveResults();




    return 0;
}