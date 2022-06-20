/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "YAMLUtil.h"
namespace traact::util  {
    bool HasValue(std::string parameter, const YAML::Node& node) {
        if(!node[parameter]){
            spdlog::error("Missing parameter {0}", parameter);
            return false;
        }
        return true;
    }
}
