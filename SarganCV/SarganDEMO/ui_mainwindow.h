/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 6.3.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QLabel *lbSevSULogo;
    QLabel *lbCopyright;
    QLabel *lblCamera;
    QLineEdit *edIPCamera;
    QLineEdit *edIP;
    QLineEdit *edPort;
    QPushButton *btnConnect;
    QLabel *lbArrowLeft;
    QLabel *lbArrowRight;
    QLabel *lbTarget;
    QLineEdit *edAngleLeft;
    QLineEdit *edAngleRight;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->setEnabled(true);
        MainWindow->resize(800, 800);
        MainWindow->setMinimumSize(QSize(800, 800));
        MainWindow->setMaximumSize(QSize(800, 800));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        lbSevSULogo = new QLabel(centralwidget);
        lbSevSULogo->setObjectName(QString::fromUtf8("lbSevSULogo"));
        lbSevSULogo->setGeometry(QRect(430, 670, 293, 90));
        lbSevSULogo->setPixmap(QPixmap(QString::fromUtf8(":/images/png/sevsu.png")));
        lbSevSULogo->setScaledContents(true);
        lbCopyright = new QLabel(centralwidget);
        lbCopyright->setObjectName(QString::fromUtf8("lbCopyright"));
        lbCopyright->setGeometry(QRect(520, 750, 202, 16));
        lbCopyright->setAlignment(Qt::AlignCenter);
        lblCamera = new QLabel(centralwidget);
        lblCamera->setObjectName(QString::fromUtf8("lblCamera"));
        lblCamera->setGeometry(QRect(80, 80, 640, 480));
        lblCamera->setAutoFillBackground(true);
        lblCamera->setStyleSheet(QString::fromUtf8(""));
        lblCamera->setAlignment(Qt::AlignCenter);
        edIPCamera = new QLineEdit(centralwidget);
        edIPCamera->setObjectName(QString::fromUtf8("edIPCamera"));
        edIPCamera->setGeometry(QRect(80, 570, 640, 31));
        QFont font;
        font.setPointSize(12);
        edIPCamera->setFont(font);
        edIPCamera->setAlignment(Qt::AlignCenter);
        edIP = new QLineEdit(centralwidget);
        edIP->setObjectName(QString::fromUtf8("edIP"));
        edIP->setGeometry(QRect(80, 610, 121, 31));
        edIP->setFont(font);
        edIP->setAlignment(Qt::AlignCenter);
        edPort = new QLineEdit(centralwidget);
        edPort->setObjectName(QString::fromUtf8("edPort"));
        edPort->setGeometry(QRect(210, 610, 51, 31));
        edPort->setFont(font);
        edPort->setAlignment(Qt::AlignCenter);
        btnConnect = new QPushButton(centralwidget);
        btnConnect->setObjectName(QString::fromUtf8("btnConnect"));
        btnConnect->setGeometry(QRect(270, 610, 451, 32));
        btnConnect->setFont(font);
        lbArrowLeft = new QLabel(centralwidget);
        lbArrowLeft->setObjectName(QString::fromUtf8("lbArrowLeft"));
        lbArrowLeft->setGeometry(QRect(80, 20, 66, 50));
        lbArrowLeft->setPixmap(QPixmap(QString::fromUtf8(":/images/png/arrow-left.png")));
        lbArrowLeft->setScaledContents(true);
        lbArrowRight = new QLabel(centralwidget);
        lbArrowRight->setObjectName(QString::fromUtf8("lbArrowRight"));
        lbArrowRight->setGeometry(QRect(654, 20, 66, 50));
        lbArrowRight->setPixmap(QPixmap(QString::fromUtf8(":/images/png/arrow-right.png")));
        lbArrowRight->setScaledContents(true);
        lbTarget = new QLabel(centralwidget);
        lbTarget->setObjectName(QString::fromUtf8("lbTarget"));
        lbTarget->setGeometry(QRect(375, 20, 50, 50));
        lbTarget->setPixmap(QPixmap(QString::fromUtf8(":/images/png/target.png")));
        lbTarget->setScaledContents(true);
        edAngleLeft = new QLineEdit(centralwidget);
        edAngleLeft->setObjectName(QString::fromUtf8("edAngleLeft"));
        edAngleLeft->setGeometry(QRect(150, 30, 51, 31));
        edAngleLeft->setFont(font);
        edAngleLeft->setAlignment(Qt::AlignCenter);
        edAngleRight = new QLineEdit(centralwidget);
        edAngleRight->setObjectName(QString::fromUtf8("edAngleRight"));
        edAngleRight->setGeometry(QRect(598, 30, 51, 31));
        edAngleRight->setFont(font);
        edAngleRight->setAlignment(Qt::AlignCenter);
        MainWindow->setCentralWidget(centralwidget);
        lblCamera->raise();
        lbSevSULogo->raise();
        lbCopyright->raise();
        edIPCamera->raise();
        edIP->raise();
        edPort->raise();
        btnConnect->raise();
        lbArrowLeft->raise();
        lbArrowRight->raise();
        lbTarget->raise();
        edAngleLeft->raise();
        edAngleRight->raise();

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "Sargan DEMO", nullptr));
        lbSevSULogo->setText(QString());
        lbCopyright->setText(QCoreApplication::translate("MainWindow", "\302\251 2023 \320\241\320\265\320\262\320\223\320\243. \320\222\321\201\320\265 \320\277\321\200\320\260\320\262\320\260 \320\267\320\260\321\211\320\270\321\211\320\265\320\275\321\213.", nullptr));
        lblCamera->setText(QCoreApplication::translate("MainWindow", "CAMERA", nullptr));
        edIPCamera->setText(QCoreApplication::translate("MainWindow", "http://localhost:8080/sargan", nullptr));
        edIP->setText(QCoreApplication::translate("MainWindow", "127.0.0.1", nullptr));
        edPort->setText(QCoreApplication::translate("MainWindow", "3310", nullptr));
        btnConnect->setText(QCoreApplication::translate("MainWindow", "\320\243\320\241\320\242\320\220\320\235\320\236\320\222\320\230\320\242\320\254 \320\241\320\236\320\225\320\224\320\230\320\235\320\225\320\235\320\230\320\225", nullptr));
        lbArrowLeft->setText(QString());
        lbArrowRight->setText(QString());
        lbTarget->setText(QString());
        edAngleLeft->setText(QCoreApplication::translate("MainWindow", "0", nullptr));
        edAngleRight->setText(QCoreApplication::translate("MainWindow", "0", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
