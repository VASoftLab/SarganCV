#include <iostream>
// ============================================================================
// Подключаем библиотеки OpenCV
// ============================================================================
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
// ============================================================================
// Подключаем библиотеки стримера
#include "nadjieb/streamer.hpp"
using MJPEGStreamer = nadjieb::MJPEGStreamer;
// ============================================================================
const bool USE_WEBCAM = true;
using namespace std;

int main()
{
    cv::VideoCapture cap;
    if (USE_WEBCAM)
    {
        // cv::VideoCapture cap(0); // Создаем соединение с камерой по-умолчанию

        cap.open(0, cv::CAP_ANY);
        if (!cap.isOpened())
        {
            std::cerr << "VideoCapture not opened\n";
            exit(EXIT_FAILURE);
        }
    }

    // Задаем качество картинки
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
    // Создаем объект стримера
    MJPEGStreamer streamer;
    // By default "/shutdown" is the target to graceful shutdown the streamer
    // if you want to change the target to graceful shutdown:
    //      streamer.setShutdownTarget("/stop");

    // Запускаем стример
    // By default std::thread::hardware_concurrency() workers is used for streaming
    // if you want to use 4 workers instead:
    //      streamer.start(8080, 4);
    streamer.start(8080);

    int i = 0; // Счетчик времени
    // Visit /shutdown or another defined target to stop the loop and graceful shutdown

    while (streamer.isRunning())
    {
        cv::Mat frame;
        if (USE_WEBCAM)
        {
            cap >> frame;
            if (frame.empty())
            {
                 std::cerr << "frame not grabbed\n";
                 exit(EXIT_FAILURE);
            }
        }

        // Адрес сервера по которому доступен поток
        // http://localhost:8080/bgr

        std::vector<uchar> buff_bgr;
        i++;
        if (!USE_WEBCAM)
            frame = cv::imread( "C:\\TEMP\\test-image.bmp" );
        // Накладываем текст
        cv::putText(frame, // target image
                    std::to_string(i), // text
                    cv::Point(10, frame.rows / 5), // top-left position
                    cv::FONT_HERSHEY_DUPLEX,
                    1.0,
                    CV_RGB(0, 0, 0), // font color
                    2);

        cv::imencode(".jpg", frame, buff_bgr, params);
        // Выгрузка изображения в поток
        streamer.publish("/bgr", std::string(buff_bgr.begin(), buff_bgr.end()));

        // Формат HSV
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        // Адрес сервера с потоком в формате HSV
        // http://localhost:8080/hsv
        std::vector<uchar> buff_hsv;
        cv::imencode(".jpg", hsv, buff_hsv, params);
        streamer.publish("/hsv", std::string(buff_hsv.begin(), buff_hsv.end()));

        // Задержка 100 мс
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Завершение работы стремера
    streamer.stop();
    cout << "Sargan streamer started..." << endl;
    return 0;
}
