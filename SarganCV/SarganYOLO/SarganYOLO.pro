TEMPLATE = app

CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp \
        neuralnetdetector.cpp


OPENCV_VER = 480

win32 {
    INCLUDEPATH += c:\OpenCV-$${OPENCV_VER}\build\include\
    LIBS += -Lc:\OpenCV-$${OPENCV_VER}\build\x64\vc16\bin
    LIBS += -Lc:\OpenCV-$${OPENCV_VER}\build\x64\vc16\lib    
    Debug:LIBS += -lopencv_world$${OPENCV_VER}d
    Release:LIBS += -lopencv_world$${OPENCV_VER}
}
unix {
    INCLUDEPATH += /usr/include/opencv4
    INCLUDEPATH += /usr/local/include/opencv4
    LIBS += -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lopencv_imgproc -lopencv_dnn -pthread
}

HEADERS += \
    neuralnetdetector.h
