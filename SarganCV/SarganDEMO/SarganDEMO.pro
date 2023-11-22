QT += core gui network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    mainwindow.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    resources.qrc

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
