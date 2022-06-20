/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/


#ifndef TRAACTMULTI_YAMLUTIL_H
#define TRAACTMULTI_YAMLUTIL_H

#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>
namespace traact::util  {
    template<typename T>
    bool SetValue(T& value, std::string parameter, const YAML::Node& local_value, const YAML::Node& global_value){
        bool valueSet= false;
        if(global_value[parameter]){
            valueSet = true;
            value = global_value[parameter].as<T>();
        }
        if(local_value[parameter]){
            valueSet = true;
            value = local_value[parameter].as<T>();
        }
        if(!valueSet){
            spdlog::error("missing parameter value for {0}", parameter);
        }
        return valueSet;
    }

    bool HasValue(std::string parameter, const YAML::Node& node) ;
}



#endif //TRAACTMULTI_YAMLUTIL_H
