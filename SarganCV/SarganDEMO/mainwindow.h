#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QUdpSocket>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

const int PACKET_SIZE = 5;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void OnVideoTimer();
    void OnBlinkTimer();

private slots:
    void on_btnConnect_clicked();
    void processDatagram();
    void onConnected();
    void onDisconnected();

private:
    Ui::MainWindow *ui;
    QTimer *videoTimer;
    QTimer *blinkTimer;

    cv::VideoCapture *webcam;
    cv::Mat source;
    cv::Mat destination;
    QImage imgcam;

    bool cvConnected = false;

    QUdpSocket udpSocket;
    QString ip;
    QHostAddress host;
    int port;
    bool isConnected = false;

    int8_t direction;
    float angle;
};
#endif // MAINWINDOW_H
