#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QScreen>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // Фиксируем размер окна и убираем иконку ресайза
    setFixedSize(QSize(800, 800));

    // Центрируем окно в пределах экрана
    move(screen()->geometry().center() - frameGeometry().center());
    ui->lbCopyright->setStyleSheet("QLabel { color: grey; }");
    ui->lblCamera->setStyleSheet("QLabel {"
                                 "border-style: solid;"
                                 "border-width: 1px;"
                                 "border-color: silver; "
                                 "}");

    ui->lbArrowLeft->setEnabled(false);
    ui->lbTarget->setEnabled(false);
    ui->lbArrowRight->setEnabled(false);
    ui->edAngleLeft->setVisible(false);
    ui->edAngleRight->setVisible(false);

    videoTimer = new QTimer(this);
    connect(videoTimer, &QTimer::timeout, this, &MainWindow::OnVideoTimer);
    blinkTimer = new QTimer(this);
    connect(blinkTimer, &QTimer::timeout, this, &MainWindow::OnBlinkTimer);

    connect(&udpSocket, SIGNAL(readyRead()), this, SLOT(processDatagram()));
    connect(&udpSocket, SIGNAL(connected()), this, SLOT(onConnected()));
    connect(&udpSocket, SIGNAL(disconnected()), this, SLOT(onDisconnected()));
}

MainWindow::~MainWindow()
{
    if (udpSocket.isOpen())
        udpSocket.close();

    if (videoTimer->isActive())
        videoTimer->stop();
    if (blinkTimer->isActive())
        blinkTimer->stop();

    delete videoTimer;
    delete blinkTimer;

    delete webcam;

    delete ui;
}

void MainWindow::OnVideoTimer()
{
    webcam->read(source);

    if (source.empty())
        return;

    cv::Mat resized;
    cv::resize(source, resized, cv::Size(640, 480));

    // Image preprocessing
    cv::cvtColor(resized, destination, cv::COLOR_BGR2RGB);

    imgcam = QImage((uchar*) destination.data,
                    destination.cols,
                    destination.rows,
                    destination.step,
                    QImage::Format_RGB888);

    // Show QImage using QLabel
    ui->lblCamera->setPixmap(QPixmap::fromImage(imgcam));
}

void MainWindow::OnBlinkTimer()
{
    if (direction == -1)
    {
        if (ui->lbArrowLeft->isEnabled())
            ui->lbArrowLeft->setEnabled(false);
        else
            ui->lbArrowLeft->setEnabled(true);
    }
    else if (direction == 1)
    {
        if (ui->lbArrowRight->isEnabled())
            ui->lbArrowRight->setEnabled(false);
        else
            ui->lbArrowRight->setEnabled(true);
    }
    else if (direction == 0)
    {
        if (ui->lbTarget->isEnabled())
            ui->lbTarget->setEnabled(false);
        else
            ui->lbTarget->setEnabled(true);
    }
}


void MainWindow::on_btnConnect_clicked()
{
    if (!cvConnected)
    {
        ui->btnConnect->setText("РАЗОРВАТЬ СОЕДИНЕНИЕ");
        cvConnected = true;

        webcam = new cv::VideoCapture(ui->edIPCamera->text().toStdString());

        //udpSocket.connectToHost(QHostAddress(ui->edIP->text()),
        //                        ui->edPort->text().toInt(),
        //                        QIODeviceBase::ReadWrite);

        udpSocket.bind(QHostAddress(ui->edIP->text()),
                                ui->edPort->text().toInt());

        if (!videoTimer->isActive())
            videoTimer->start(100);
    }
    else
    {
        ui->btnConnect->setText("УСТАНОВИТЬ СОЕДИНЕНИЕ");
        cvConnected = false;

        if (webcam->isOpened())
            webcam->release();

        if (videoTimer->isActive())
            videoTimer->stop();

        if (udpSocket.isOpen())
            udpSocket.close();

        QPixmap pixmap;
        QColor color;

        // Очистка изображения
        color = QColor(255, 255, 255, 255);
        pixmap = QPixmap(ui->lblCamera->size());
        pixmap.fill(color);

        ui->lblCamera->setPixmap(pixmap);
        ui->lblCamera->setText("CAMERA");
    }
}

void MainWindow::onConnected()
{
    isConnected = true;
}

void MainWindow::onDisconnected()
{
    isConnected = false;
}

void MainWindow::processDatagram()
{
    if (udpSocket.pendingDatagramSize() == PACKET_SIZE)
    {
        QByteArray datagram;
        do {
            datagram.resize(udpSocket.pendingDatagramSize());
            udpSocket.readDatagram(datagram.data(), datagram.size());
        } while (udpSocket.hasPendingDatagrams());

        QDataStream in(&datagram, QIODevice::ReadOnly);
        in.setFloatingPointPrecision(QDataStream::SinglePrecision);
        // in.setByteOrder(QDataStream::LittleEndian);
        in.setVersion(QDataStream::Qt_6_3);

        in >> direction;
        in >> angle;

        if (direction == -1)
        {
            ui->edAngleLeft->setVisible(true);
            ui->edAngleRight->setVisible(false);
            ui->lbArrowLeft->setEnabled(true);
            ui->lbArrowRight->setEnabled(false);
            ui->lbTarget->setEnabled(false);
        }
        else if (direction == 1)
        {
            ui->edAngleLeft->setVisible(false);
            ui->edAngleRight->setVisible(true);
            ui->lbArrowLeft->setEnabled(false);
            ui->lbArrowRight->setEnabled(true);
            ui->lbTarget->setEnabled(false);
        }
        else if (direction == 0)
        {
            ui->edAngleLeft->setVisible(false);
            ui->edAngleRight->setVisible(false);
            ui->lbArrowLeft->setEnabled(false);
            ui->lbArrowRight->setEnabled(false);
            ui->lbTarget->setEnabled(true);
        }
        else if (direction == -101)
        {
            ui->edAngleLeft->setVisible(false);
            ui->edAngleRight->setVisible(false);
            ui->lbArrowLeft->setEnabled(false);
            ui->lbArrowRight->setEnabled(false);
            ui->lbTarget->setEnabled(false);
        }

        //if (direction == -101)
        //{
        //    if (blinkTimer->isActive())
        //        blinkTimer->stop();
        //}
        //else
        //{
        //    if (!blinkTimer->isActive())
        //        blinkTimer->start(500);
        //}

        ui->edAngleLeft->setText(QString::number(angle, 'f', 0));
        ui->edAngleRight->setText(QString::number(angle, 'f', 0));

    }
}


