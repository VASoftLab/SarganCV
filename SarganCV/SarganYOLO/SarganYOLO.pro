TEMPLATE = app

CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp \
        neuralnetdetector.cpp

# Include OpenCV
INCLUDEPATH += c:\OpenCV-480\build\include\
LIBS += -Lc:\OpenCV-480\build\x64\vc16\bin
LIBS += -Lc:\OpenCV-480\build\x64\vc16\lib

OPENCV_VER = 480
#LIBS += -lopencv_world$${OPENCV_VER}
LIBS += -lopencv_world$${OPENCV_VER}d

HEADERS += \
    neuralnetdetector.h
