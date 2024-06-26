# /usr/bin/python3
import os
from conan import ConanFile
from conan.tools.build import can_run

class TraactPackage(ConanFile):
    python_requires = "traact_base/0.0.0@traact/latest"
    python_requires_extend = "traact_base.TraactPackageCmake"

    name = "test_apps"
    version = "0.0.0"
    description = ""
    url = ""
    license = "MIT"
    author = "Frieder Pankratz#"
    
    settings = "os", "compiler", "build_type", "arch"
    compiler = "cppstd"

    options = {
        "shared": [True, False]
    }

    default_options = {
        "shared": True
    }

    exports_sources = "CMakeLists.txt", "src/*"

    def requirements(self):
        self.requires("traact_base/0.0.0@traact/latest")
        self.requires("traact_core/0.0.0@traact/latest")
        self.requires("traact_spatial/0.0.0@traact/latest")
        self.requires("traact_vision/0.0.0@traact/latest")
        self.requires("traact_gui/0.0.0@traact/latest")
        self.requires("traact_component_basic/0.0.0@traact/latest")
        self.requires("traact_component_kinect_azure/0.0.0@traact/latest")
        self.requires("traact_component_cereal/0.0.0@traact/latest")
        self.requires("traact_component_aruco/0.0.0@traact/latest")
        self.requires("traact_pcpd/0.0.0@traact/latest")
        self.requires("traact_pointcloud/0.0.0@traact/latest")
        self.requires("yaml-cpp/0.7.0")
        self.requires("cpp-httplib/0.14.0", transitive_libs=True)

    def configure(self):
        self.options['traact_core'].shared = self.options.shared
        self.options['traact_facade'].shared = self.options.shared
        self.options['traact_spatial'].shared = self.options.shared
        self.options['traact_vision'].shared = self.options.shared
