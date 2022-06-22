/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef APP_LOCAL_SRC_BAPROBLEMLOADER_H_
#define APP_LOCAL_SRC_BAPROBLEMLOADER_H_

#include <traact/vision/bundle_adjustment/BundleAdjustment.h>
#include <yaml-cpp/yaml.h>

namespace traact {

class BAProblemLoader {
 public:
    bool LoadConfig(std::string config_file);
    bool LoadConfig(YAML::Node config);
    std::shared_ptr<traact::vision::bundle_adjustment::BundleAdjustment> GetBundleAdjustment();

    void SaveResults();

 protected:
     std::shared_ptr<vision::bundle_adjustment::BundleAdjustment> ba_;
     std::vector<std::shared_ptr<vision::bundle_adjustment::BACamera> > cameras_;
     std::map<size_t, std::string> artekmed_result_files_;

};

} // traact

#endif //APP_LOCAL_SRC_BAPROBLEMLOADER_H_
