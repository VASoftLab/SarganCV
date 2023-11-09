TEMPLATE = app

CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp

OPENCV_VER = 480

win32 {
    INCLUDEPATH += c:\OpenCV-$${OPENCV_VER}\build\include\
    LIBS += -Lc:\OpenCV-$${OPENCV_VER}\build\x64\vc16\bin
    LIBS += -Lc:\OpenCV-$${OPENCV_VER}\build\x64\vc16\lib
    LIBS += -lopencv_world$${OPENCV_VER}d
}
unix {
    INCLUDEPATH += /usr/include/opencv4
    LIBS += -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_features2d -lopencv_calib3d -lopencv_videoio -lopencv_imgproc
}
