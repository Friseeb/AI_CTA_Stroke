cmake_minimum_required(VERSION 3.16.3)
project(CTAVertebralSOP)

# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/Friseeb/AI_CTA_Stroke")
set(EXTENSION_CATEGORY "Cardiac")
set(EXTENSION_CONTRIBUTORS "Sebastian Fridman (NYU)")
set(EXTENSION_DESCRIPTION "CTA Vertebral SOP tools: guided segmentation wizard and finalize/save utility.")
set(EXTENSION_ICONURL "")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "")

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

add_subdirectory(CTAVertebralWizard)
add_subdirectory(CTAFinalizeSOP)

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
