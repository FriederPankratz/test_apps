/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "BAProblemLoader.h"
#include "util/YAMLUtil.h"
#include <traact/util/FileUtil.h>
#include <cppfs/FilePath.h>
#include <fstream>
#include <cereal/archives/json.hpp>
#include <traact/cereal/CerealSpatial.h>
#include <traact/cereal/CerealVision.h>

namespace traact {

using namespace vision::bundle_adjustment;

bool missingKey(YAML::Node node, std::string key) {
    if (node[key])
        return false;

    spdlog::error("config invalid: missing '{0}' parameter", key);
    return true;
}

bool BAProblemLoader::LoadConfig(std::string config_file) {
    auto config = YAML::LoadFile(config_file);
    return LoadConfig(config);
}

bool BAProblemLoader::LoadConfig(YAML::Node config) {
    ba_ = std::make_shared<BundleAdjustment>();
    if (!util::HasValue("ba", config)) {
        return false;
    }
    auto ba_config = config["ba"];
    if (!util::HasValue("cameras", ba_config)) {
        return false;
    }
    if (!util::HasValue("target", ba_config)) {
        return false;
    }
    if (!util::HasValue("default", ba_config)) {
        return false;
    }
    auto default_parameter = ba_config["default"];
    if (!util::HasValue("tracking_root_folder", default_parameter))
        return false;
    std::string tracking_root = default_parameter["tracking_root_folder"].as<std::string>();
    cppfs::FilePath tracking_root_fp(tracking_root);

    auto ba_cameras = ba_config["cameras"];
    auto ba_target_config = ba_config["target"];

    if (!util::HasValue("model_file", ba_target_config)) {
        return false;
    }
    if (!util::HasValue("target_pose_file", ba_target_config)) {
        return false;
    }
    if (!util::HasValue("target_feature_file", ba_target_config)) {
        return false;
    }
    if (!util::HasValue("point_3D_feature_file", ba_target_config)) {
        return false;
    }
    if(!util::HasValue("target_to_origin_file", ba_target_config)) {
        return false;
    }
    target_to_origin_file_ = ba_target_config["target_to_origin_file"].as<std::string>();
    if(!util::fileExists(target_to_origin_file_, "target_to_origin_file")){
        return false;
    }

    std::string model_file = ba_target_config["model_file"].as<std::string>();
    if (!util::fileExists(model_file, "model_file")) {
        return false;
    }

    std::string target_pose_file = ba_target_config["target_pose_file"].as<std::string>();
    target_pose_file = tracking_root_fp.resolve(target_pose_file).fullPath();
    if (!util::fileExists(target_pose_file, "target_pose_file")) {
        return false;
    }
    std::string target_feature_file = ba_target_config["target_feature_file"].as<std::string>();
    target_feature_file = tracking_root_fp.resolve(target_feature_file).fullPath();
    if (!util::fileExists(target_feature_file, "target_feature_file")) {
        return false;
    }
    std::string point_3D_feature_file = ba_target_config["point_3D_feature_file"].as<std::string>();
    point_3D_feature_file = tracking_root_fp.resolve(point_3D_feature_file).fullPath();
    if (!util::fileExists(model_file, "point_3D_feature_file")) {
        return false;
    }



    //----------------------------------------------------------------------------------

    BATarget::Ptr ba_target = std::make_shared<BATarget>();
    std::map<std::uint64_t, spatial::Pose6D> target_pose_data;
    std::map<std::uint64_t, vision::FeatureList> target_feature_data;
    std::map<std::uint64_t, vision::FeatureList> point_3D_feature_data;
    try {
        {
            std::ifstream stream;
            stream.open(model_file);
            cereal::JSONInputArchive archive(stream);

            vision::Position3DList data;
            archive(data);
            ba_target->SetTargetData(data);
        }
        {
            std::ifstream stream;
            stream.open(target_pose_file);
            cereal::JSONInputArchive archive(stream);
            archive(target_pose_data);
        }
        {
            std::ifstream stream;
            stream.open(target_feature_file);
            cereal::JSONInputArchive archive(stream);
            archive(target_feature_data);
        }
        {
            std::ifstream stream;
            stream.open(point_3D_feature_file);
            cereal::JSONInputArchive archive(stream);
            archive(point_3D_feature_data);
        }

    } catch (std::exception e) {
        spdlog::spdlog_ex(e.what());
        return false;
    }

    for (const auto &ts_pose : target_pose_data) {
        Timestamp ts = traact::AsTimestamp(ts_pose.first);
        Eigen::Affine3d pose = ts_pose.second.cast<double>();
        ba_target->SetMeasurement(ts, pose);
    }

    if (bool use_target_residual = ba_target_config["use_target_residual"].as<bool>()) {
        ba_target->SetUseTargetResidual(use_target_residual);
    }
    if (double target_residual_stddev = ba_target_config["target_residual_stddev"].as<double>()) {
        ba_target->SetStdDev(target_residual_stddev);
    }

    spdlog::info("Loaded Target: \n{0}", ba_target->toString());

    ba_->SetTarget(ba_target);
    int target_point_count = ba_target->GetTargetData().size();
    std::unordered_map<uint64_t ,std::unordered_map<vision::FeatureID, size_t> > all_point_2D_feature_to_model_index;
    for(const auto& ts_feature_data : target_feature_data){
        auto current_ts = ts_feature_data.first;
        const auto& feature_data = ts_feature_data.second;
        const auto& point_features = point_3D_feature_data.at(current_ts);
        auto& point_2D_feature_to_model_index = all_point_2D_feature_to_model_index[current_ts];
        for (int model_index = 0; model_index < feature_data.constructed_from.size(); ++model_index) {

            auto model_feature = feature_data.feature_id[model_index];
            auto find_point_3D_feature = feature_data.constructed_from.find(model_feature);
            if(find_point_3D_feature == feature_data.constructed_from.end()){
                SPDLOG_DEBUG("model feature {0} currently not found", model_index);
                continue;
            }
            if(find_point_3D_feature->second.size() != 1) {
                SPDLOG_ERROR("model feature should be reconstructed from exactly one 3d point feature");
                return true;
            }
            auto& point_3D_feature = find_point_3D_feature->second.front();
            auto find_point_2D_feature = point_features.constructed_from.find(point_3D_feature);
            if(find_point_2D_feature == point_features.constructed_from.end()){
                SPDLOG_ERROR("could not find point 2D features a 3D point was reconstructed from");
                return true;
            }
            for(auto point_2D_feature : find_point_2D_feature->second){
                point_2D_feature_to_model_index.emplace(point_2D_feature, model_index);
            }

        }
    }



    //----------------------------------------------------------------------------------








    for (const auto &camera : ba_config["cameras"]) {
        std::string camera_name = camera.first.as<std::string>();
        auto parameter = camera.second;

        if (!util::HasValue("result_file", parameter))
            return false;
        if (!util::HasValue("result_internal_file", parameter))
            return false;
        if (!util::HasValue("intrinsic_file", parameter))
            return false;
        if (!util::HasValue("extrinsic_file", parameter))
            return false;
        if (!util::HasValue("measurement_file", parameter))
            return false;
        if (!util::HasValue("feature_file", parameter))
            return false;
        if (!util::HasValue("static_position", parameter))
            return false;
        if (!util::HasValue("static_rotation", parameter))
            return false;

        auto measurement_file = parameter["measurement_file"].as<std::string>();
        auto feature_file = parameter["feature_file"].as<std::string>();
        auto result_file = parameter["result_file"].as<std::string>();
        auto result_file_internal = parameter["result_internal_file"].as<std::string>();
        auto intrinsic_file = parameter["intrinsic_file"].as<std::string>();
        auto extrinsic_file = parameter["extrinsic_file"].as<std::string>();

        result_file = tracking_root_fp.resolve(result_file).fullPath();
        result_file_internal = tracking_root_fp.resolve(result_file_internal).fullPath();
        intrinsic_file = tracking_root_fp.resolve(intrinsic_file).fullPath();
        extrinsic_file = tracking_root_fp.resolve(extrinsic_file).fullPath();
        measurement_file = tracking_root_fp.resolve(measurement_file).fullPath();
        feature_file = tracking_root_fp.resolve(feature_file).fullPath();

        if (!util::fileExists(measurement_file, "measurement_file")) {
            return false;
        }
        if (!util::fileExists(feature_file, "feature_file")) {
            return false;
        }
        if (!util::fileExists(extrinsic_file, "extrinsic_file")) {
            return false;
        }
        if (!util::fileExists(intrinsic_file, "intrinsic_file")) {
            return false;
        }

        bool static_position = parameter["static_position"].as<bool>();
        bool static_rotation = parameter["static_rotation"].as<bool>();

        BACamera::Ptr ba_cam = std::make_shared<BACamera>(camera_name);

        ba_cam->setStaticPosition(static_position);
        ba_cam->setStaticRotation(static_rotation);
        ba_cam->setResultfile(result_file_internal);

        try {
            {
                std::ifstream stream;
                stream.open(intrinsic_file);
                cereal::JSONInputArchive archive(stream);
                vision::CameraCalibration data;
                archive(data);
                ba_cam->setIntrinsic(data);
            }
            {
                std::ifstream stream;
                stream.open(extrinsic_file);
                cereal::JSONInputArchive archive(stream);
                spatial::Pose6D data;
                archive(data);
                ba_cam->setExtrinsic(data);
            }

            {
                std::ifstream stream_data;
                stream_data.open(measurement_file);
                cereal::JSONInputArchive archive_data(stream_data);
                std::ifstream stream_features;
                stream_features.open(feature_file);
                cereal::JSONInputArchive archive_features(stream_features);

                std::map<std::uint64_t, vision::KeyPointList> data;
                std::map<std::uint64_t, vision::FeatureList> features;
                archive_data(data);
                archive_features(features);



                for (const auto &tmp : data) {
                    auto current_ts = tmp.first;
                    std::vector<Eigen::Vector2d> image_points;

                    auto find_point_2D_feature_to_model_index = all_point_2D_feature_to_model_index.find(current_ts);
                    if(find_point_2D_feature_to_model_index == all_point_2D_feature_to_model_index.end()){
                        continue;
                    }

                    const std::unordered_map<vision::FeatureID, size_t>& point_2D_feature_to_model_index = find_point_2D_feature_to_model_index->second;
                    const auto& point_list = tmp.second;
                    const auto& point_list_feature = features.at(current_ts);
                    image_points.resize(target_point_count);
                    int point_count{0};
                    for (size_t point_index = 0; point_index < point_list.size(); ++point_index) {
                        auto was_used = point_2D_feature_to_model_index.find(point_list_feature.feature_id[point_index]);
                        if(was_used != point_2D_feature_to_model_index.end()){
                            image_points[was_used->second]= Eigen::Vector2d (point_list[point_index].pt.x,point_list[point_index].pt.y);
                            ++point_count;
                        }
                    }
//                    if(image_points.size() < target_point_count) {
//                        auto target_world_points = ba_target->GetMeasurement(AsTimestamp(current_ts));
//                        ba_cam->tryFindingImagePoints(target_world_points, image_points, point_list);
//                    }
                    if(point_count == target_point_count){
                        ba_cam->SetMeasurement(AsTimestamp(current_ts), image_points);
                    } else {
                        //SPDLOG_WARN("not all target points in camera image for timestamp {0} {1} pf {2}", current_ts, image_points.size(), target_point_count);

                    }

                }

                ba_->AddCamera(ba_cam);
                artekmed_result_files_.emplace(cameras_.size(), result_file);
                cameras_.emplace_back(ba_cam);

                SPDLOG_INFO(ba_cam->toString());
            }

        } catch (std::exception& e) {
            spdlog::spdlog_ex(e.what());
            return false;
        }

        spdlog::info("Loaded Camera: \n{0}", ba_cam->toString());

    }


    return true;
}

std::shared_ptr<BundleAdjustment> BAProblemLoader::GetBundleAdjustment() {
    return ba_;
}

void BAProblemLoader::SaveResults() {

    const auto results = ba_->getResult();
    for (int i = 0; i < cameras_.size() ; ++i) {
        auto& camera = cameras_[i];
        try{
            {
                std::ofstream stream;
                stream.open(camera->getResultfile());
                cereal::JSONOutputArchive archive(stream);
                archive(results[i]);
            }
            {
                auto world2camera_opencv = results[i].inverse();
                spatial::Pose6D target2origin_ubitrack;
                {
                    std::ifstream stream;
                    stream.open(target_to_origin_file_);
                    cereal::JSONInputArchive archive(stream);
                    archive(target2origin_ubitrack);


                    //target2origin_ubitrack = target2origin_ubitrack * Eigen::AngleAxisf(0.25*M_PI, Eigen::Vector3f::UnitY());
                    target2origin_ubitrack.rotate( Eigen::AngleAxisf(0.25*M_PI, Eigen::Vector3f::UnitX()) * Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitY()));

                }

                // are artekmed poses not in opengl format?

//                world2camera_opencv.matrix().row(1) = world2camera_opencv.matrix().row(1) * -1;
//                world2camera_opencv.matrix().row(2) = world2camera_opencv.matrix().row(2) * -1;
//                spatial::Pose6D final_pose = world2camera_opencv;

//                spatial::Translation3D position_opencv = spatial::Translation3D(world2camera_opencv.translation());
//                spatial::Rotation3D  rotation_opencv = spatial::Rotation3D(world2camera_opencv.rotation());
//                spatial::Rotation3D  opencv_to_opengl(0,1,0,0);
//                spatial::Pose6D position_opengl = opencv_to_opengl * position_opencv * opencv_to_opengl;
//                spatial::Rotation3D rotation_opengl = opencv_to_opengl * rotation_opencv;
//                spatial::Pose6D final_pose = position_opengl * rotation_opengl;

                spatial::Pose6D pose_tmp = spatial::Pose6D::Identity();
                pose_tmp.rotate( spatial::Rotation3D (0,1,0,0));

                spatial::Pose6D pose_tmp2 = spatial::Pose6D::Identity();
                pose_tmp2.rotate(spatial::Rotation3D(0.7071,-0.7071,0,0) * spatial::Rotation3D(0.7071,0,0,0.7071));

                spatial::Pose6D world2camera_artekmed = pose_tmp2 * world2camera_opencv * pose_tmp;

                spatial::Pose6D final_pose = target2origin_ubitrack.inverse() * world2camera_artekmed;

                auto final_filename = artekmed_result_files_.at(i);
                std::ofstream stream;
                stream.open(final_filename);
                cereal::JSONOutputArchive archive(stream);
                archive(final_pose);
            }

        }catch(std::exception& e){
            SPDLOG_ERROR(e.what());
        }
    }
}

} // traact