#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <cerrno>
#include <filesystem>
#include <chrono>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "nadjieb/streamer.hpp"
using MJPEGStreamer = nadjieb::MJPEGStreamer;

#include "neuralnetdetector.h"

/** Параметры картинки */
static const float IMG_WIDTH = 640;
static const float IMG_HEIGHT  = 640;

///////////////////////////////////////////////////////////////////////////////
// !!!ЗНАЧЕНИЕ УГЛА ОБЗОРА ДОЛЖНО БЫТЬ ИЗМЕНЕНО ПОД КАМЕРУ НА АППАРАТЕ!!!
///////////////////////////////////////////////////////////////////////////////
static const float CAMERA_ANGLE = 80/*60*/;
///////////////////////////////////////////////////////////////////////////////

// Размеры прицела
static const float SIGHT_WIDTH = 50;

// Горизонтальная линейка
static const int RULER_H = 40;

namespace fs = std::filesystem;

/** Функция поиска угла между целью и центром фрейма
 *   @param resolution - разрешение камеры по горизонтали
 *   @param cx - абциса центра цели
 *   @return угол между центром фрейма и центром цели
 */
int findAngleF(double resolution, int cx)
{
    return (int)((cx * CAMERA_ANGLE / resolution) - CAMERA_ANGLE / 2);
}

// https://stackoverflow.com/questions/24686846/get-current-time-in-milliseconds-or-hhmmssmmm-format
std::string time_in_HH_MM_SS_MMM()
{
    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    auto timer = system_clock::to_time_t(now);
    std::tm bt = *std::localtime(&timer);
    std::ostringstream oss;
    oss << std::put_time(&bt, "%H:%M:%S"); // HH:MM:SS
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

int main()
{
    cv::VideoCapture source;

    // Источник изображений по умолчанию
    // cv::VideoCapture source(1, cv::CAP_GSTREAMER);
#ifdef _WIN32
    // source.open(0, cv::CAP_ANY);
    // Disable Error message: getPluginCandidates Found 2 plugin(s) for GSTREAMER
    source.open(0, cv::CAP_DSHOW);
#else
    // source.open(0, cv::CAP_ANY);
    source.open(0, cv::CAP_GSTREAMER);
#endif
    source.set(cv::CAP_PROP_FPS, 30);

    //if (!source.isOpened())
    //{
    //    std::cerr << "ERROR! Left camera not ready!" << std::endl;
    //    std::cin.get();
    //    return -1;
    //}
    //else
    //{
    //    std::cout << "LEFT camera test -- SUCCESS" << std::endl;
    //}

    cv::Mat frame;

    //for (;;)
    //{
    //    source >> frame;
    //    cv::imshow("Web Camera", frame);

    //    if (cv::waitKey(5) >= 0)
    //        break;
    //}
    //cv::destroyAllWindows();

    //return 0;

    ///////////////////////////////////////////////////////////////////////////
    // Подготовка стримера
    ///////////////////////////////////////////////////////////////////////////
    // Задаем качество картинки
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
    // Создаем объект стримера
    MJPEGStreamer streamer;
    // Буфер для работы с потоком
    std::vector<uchar> streamerBuf;
    // Запуск стримера
    streamer.start(8080);
    ///////////////////////////////////////////////////////////////////////////

    // Получить разрешение камеры по горизонтали и вертикали
    double FRAME_WIDTH = source.get(cv::CAP_PROP_FRAME_WIDTH);
    double FRAME_HEIGHT = source.get(cv::CAP_PROP_FRAME_HEIGHT);

    if (DIAGNOSTIC_LOG)
        std::cout << "Camera resolution: " << FRAME_WIDTH << " x " << FRAME_HEIGHT << std::endl;

    // Путь к модели и файлу с классами
    fs::path nn_dir ("nn");
    fs::path nn_onnx ("yolov5s.onnx");
    fs::path nn_names ("coco.names");

    const fs::path model_path = fs::current_path() / nn_dir / nn_onnx;
    const fs::path classes_path = fs::current_path() / nn_dir / nn_names;

    if (DIAGNOSTIC_LOG)
        std::cout << model_path.u8string() << std::endl;

    // TODO -- Разобраться, почему падает код с прямоугольными размерами фрейма
    // NeuralNetDetector detector(model_path.u8string(), classes_path.u8string(), FRAME_WIDTH, FRAME_HEIGHT);
    NeuralNetDetector detector(model_path.u8string(), classes_path.u8string(), (int)IMG_WIDTH, (int)IMG_HEIGHT);

    // cv::Mat frame;

    // Бесконечный цикл с захватом видео и детектором
    while(cv::waitKey(1) < 1)
    {
        // Захват текущего кадра
        source >> frame;
        if (frame.empty())
        {
            cv::waitKey();
            break;
        }

        ///////////////////////////////////////////////////////////////////////
        // Отработка детектора
        ///////////////////////////////////////////////////////////////////////
        cv::Mat img = detector.process(frame);

        // Результаты работы детектора
        std::vector<int> class_ids = detector.get_class_ids();
        std::vector<float> confidences = detector.get_confidences();
        std::vector<cv::Rect> boxes = detector.get_boxes();
        std::vector<std::string> classes = detector.get_classes();
        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////
        // Наложение подложки
        ///////////////////////////////////////////////////////////////////////
        double alpha = 0.5;
        cv::Mat overlay;
        img.copyTo(overlay);

        // Создаем фон под текстом
        cv::rectangle(
            overlay,
            cv::Point(0, (int)FRAME_HEIGHT - 30),
            cv::Point((int)FRAME_WIDTH, (int)FRAME_HEIGHT),
            CV_RGB(255, 255, 255),
            -1);

        cv::addWeighted(overlay, alpha, img, 1 - alpha, 0, img);
        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////
        // Поиск бокса цели с максимальной площадью
        ///////////////////////////////////////////////////////////////////////
        int bigestArea = INT_MIN;
        int bigestIndex = -1;
        int boxIndex = -1;

        for (auto b : boxes)
        {
            boxIndex++;
            if (b.area() > bigestArea)
            {
                bigestIndex = boxIndex;
                bigestArea = b.area();
            }
        }
        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////
        // Отрисовка бокса и прицела цели
        ///////////////////////////////////////////////////////////////////////
        if (boxes.size() > 0)
        {
            cv::Point center = (boxes[bigestIndex].br() + boxes[bigestIndex].tl()) * 0.5;

            // Отрисовка прицела в центре фрейма цели
            cv::Point objectBoxPt1;
            cv::Point objectBoxPt2;

            objectBoxPt1.x = center.x - (int)(SIGHT_WIDTH / 2);
            objectBoxPt1.y = center.y - (int)(SIGHT_WIDTH / 2);
            objectBoxPt2.x = center.x + (int)(SIGHT_WIDTH / 2);
            objectBoxPt2.y = center.y + (int)(SIGHT_WIDTH / 2);
            cv::rectangle(img, objectBoxPt1, objectBoxPt2, CV_RGB(255, 0, 0), 2, 0);

            // Перекрестие (фрейм объекта)
            cv::Point objectCrossPtV1;
            cv::Point objectCrossPtV2;
            cv::Point objectCrossPtH1;
            cv::Point objectCrossPtH2;

            objectCrossPtV1.x = center.x;
            objectCrossPtV1.y = center.y - (int)(SIGHT_WIDTH / 6);
            objectCrossPtV2.x = center.x;
            objectCrossPtV2.y = center.y + (int)(SIGHT_WIDTH / 6);

            objectCrossPtH1.x = center.x - (int)(SIGHT_WIDTH / 6);
            objectCrossPtH1.y = center.y;
            objectCrossPtH2.x = center.x + (int)(SIGHT_WIDTH / 6);
            objectCrossPtH2.y = center.y;

            cv::line(img, objectCrossPtV1, objectCrossPtV2, CV_RGB(255, 0, 0), 2, 0);
            cv::line(img, objectCrossPtH1, objectCrossPtH2, CV_RGB(255, 0, 0), 2, 0);
        }
        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////
        // Расчет центральной точки бортовой системы наведения
        ///////////////////////////////////////////////////////////////////////
        cv::Point boardBoxPt1;
        cv::Point boardBoxPt2;

        // Координаты бокса прицела
        boardBoxPt1.x = (int)(img.cols / 2) - (int)SIGHT_WIDTH;
        boardBoxPt1.y = (int)(img.rows / 2) - (int)SIGHT_WIDTH;
        boardBoxPt2.x = (int)(img.cols / 2) + (int)SIGHT_WIDTH;
        boardBoxPt2.y = (int)(img.rows / 2) + (int)SIGHT_WIDTH;

        std::string direction;
        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////
        // Расчет управления
        ///////////////////////////////////////////////////////////////////////
        if (boxes.size() > 0)
        {
            // Расчет центра бокса с обнаруженной целью
            cv::Point center = (boxes[bigestIndex].br() + boxes[bigestIndex].tl()) * 0.5;

            // Угол между прицелом и целью
            int angle = findAngleF(FRAME_WIDTH, center.x);

            // Команда управления лево / право
            direction = center.x > FRAME_WIDTH / 2 ? "RIGTH" : "LEFT";

            // Если цель находится в границах прицела - удерживаем курс
            //if ((boardBoxPt1.x <= center.x) && (center.x <= boardBoxPt2.x) &&
            //    (boardBoxPt1.y <= center.y) && (center.y <= boardBoxPt2.y))
            //{
            //    direction = "HOLD";
            //}

            if ((boardBoxPt1.x <= center.x) && (center.x <= boardBoxPt2.x))
            {
                direction = "HOLD";
            }

            // Время работы детектора
            std::stringstream ssTime;
            ssTime << std::fixed << std::setprecision(2) << detector.get_inference();
            std::string inference = ssTime.str();

            // Строка инфорации
            std::string textInfo = " CMD: (" + direction + ":" + std::to_string(angle) + ")" +
                                   " TARGET: (" + std::to_string(center.x) + ";" + std::to_string(center.y) + ")" +
                                   " RES: (" + std::to_string((int)FRAME_WIDTH) + "x" + std::to_string((int)FRAME_HEIGHT) + ")" +
                                   " TIME: " + inference;
            cv::putText(img, textInfo, cv::Point(10, img.rows - 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 0, 255), 1);

            if (COMMAND_LOG)
            {
                std::string diagnosticInfo = "CMD:\t(" + direction + ":" + std::to_string(angle) + ")" + "\tTIME: " + inference + "\t" + time_in_HH_MM_SS_MMM();
                std::cout << diagnosticInfo << std::endl;
            }
        }
        else
        {
            std::string textInfo = " RES: (" + std::to_string((int)FRAME_WIDTH) + "x" + std::to_string((int)FRAME_HEIGHT) + ")";
            cv::putText(img, textInfo, cv::Point(10, img.rows - 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 0, 255), 1);
        }
        ///////////////////////////////////////////////////////////////////////
        // Отрисовка бортовых бокса и прицела
        ///////////////////////////////////////////////////////////////////////

        if (direction == "HOLD")
            cv::rectangle(img, boardBoxPt1, boardBoxPt2, CV_RGB(255, 0, 0), 2, 0);
        else
            cv::rectangle(img, boardBoxPt1, boardBoxPt2, CV_RGB(255, 255, 255), 2, 0);

        // Перекрестие (основное изображение)
        cv::Point boardCrossPtV1;
        cv::Point boradCrossPtV2;
        cv::Point boardCrossPtH1;
        cv::Point boardCrossPtH2;

        boardCrossPtV1.x = (int)(img.cols / 2);
        boardCrossPtV1.y = (int)(img.rows / 2) - (int)(SIGHT_WIDTH / 4);
        boradCrossPtV2.x = (int)(img.cols / 2);
        boradCrossPtV2.y = (int)(img.rows / 2) + (int)(SIGHT_WIDTH / 4);

        boardCrossPtH1.x = (int)(img.cols / 2) - (int)(SIGHT_WIDTH / 4);
        boardCrossPtH1.y = (int)(img.rows / 2);
        boardCrossPtH2.x = (int)(img.cols / 2) + (int)(SIGHT_WIDTH / 4);
        boardCrossPtH2.y = (int)(img.rows / 2);

        if (direction == "HOLD")
        {
            cv::line(img, boardCrossPtV1, boradCrossPtV2, CV_RGB(255, 0, 0), 2, 0);
            cv::line(img, boardCrossPtH1, boardCrossPtH2, CV_RGB(255, 0, 0), 2, 0);
        }
        else
        {
            cv::line(img, boardCrossPtV1, boradCrossPtV2, CV_RGB(255, 255, 255), 2, 0);
            cv::line(img, boardCrossPtH1, boardCrossPtH2, CV_RGB(255, 255, 255), 2, 0);
        }
        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////
        // Отрисовка горизонтальной линейки
        ///////////////////////////////////////////////////////////////////////

        int fontFace = cv::FONT_HERSHEY_PLAIN;
        double fontScale = 1;
        int thickness = 1;
        int baseline = 0;

        cv::Point rulerV30N;
        cv::Point rulerV30P;
        cv::Point rulerV20N;
        cv::Point rulerV20P;
        cv::Point rulerV10N;
        cv::Point rulerV10P;
        cv::Point rulerVZer;

        double K = FRAME_WIDTH / FRAME_HEIGHT;
        K = 1;

        double delta30 = (double)RULER_H * tan(30 * M_PI / 180) * K;
        double delta20 = (double)RULER_H * tan(20 * M_PI / 180) * K;
        double delta10 = (double)RULER_H * tan(10 * M_PI / 180) * K;

        // 30
        rulerV30N.x = (int)(FRAME_WIDTH / 2.0) - (int)((double)FRAME_HEIGHT * tan(30 * M_PI / 180) * K) + (int)delta30;
        rulerV30P.x = (int)(FRAME_WIDTH / 2.0) + (int)((double)FRAME_HEIGHT * tan(30 * M_PI / 180) * K) - (int)delta30;
        // 20
        rulerV20N.x = (int)(FRAME_WIDTH / 2.0) - (int)((double)FRAME_HEIGHT * tan(20 * M_PI / 180) * K) + (int)delta20;
        rulerV20P.x = (int)(FRAME_WIDTH / 2.0) + (int)((double)FRAME_HEIGHT * tan(20 * M_PI / 180) * K) - (int)delta20;
        // 10
        rulerV10N.x = (int)(FRAME_WIDTH / 2.0) - (int)((double)FRAME_HEIGHT * tan(10 * M_PI / 180) * K) + (int)delta10;
        rulerV10P.x = (int)(FRAME_WIDTH / 2.0) + (int)((double)FRAME_HEIGHT * tan(10 * M_PI / 180) * K) - (int)delta10;

        rulerVZer.x = (int)(FRAME_WIDTH / 2.0);
        rulerV30N.y = RULER_H;
        rulerV30P.y = RULER_H;
        rulerV20N.y = RULER_H;
        rulerV20P.y = RULER_H;
        rulerV10N.y = RULER_H;
        rulerV10P.y = RULER_H;
        rulerVZer.y = RULER_H;

        int lw = 2;
        int dw = 7;
        int tw = 5;

        cv::line(img, rulerV30N, rulerV30P, CV_RGB(255, 255, 255), lw, 0);
        cv::line(img, rulerV30N, cv::Point(rulerV30N.x, rulerV30N.y - dw), CV_RGB(255, 255, 255), lw, 0);
        cv::line(img, rulerV30P, cv::Point(rulerV30P.x, rulerV30P.y - dw), CV_RGB(255, 255, 255), lw, 0);
        cv::line(img, rulerV20N, cv::Point(rulerV20N.x, rulerV20N.y - dw), CV_RGB(255, 255, 255), lw, 0);
        cv::line(img, rulerV20P, cv::Point(rulerV20P.x, rulerV20P.y - dw), CV_RGB(255, 255, 255), lw, 0);
        cv::line(img, rulerV10N, cv::Point(rulerV10N.x, rulerV10N.y - dw), CV_RGB(255, 255, 255), lw, 0);
        cv::line(img, rulerV10P, cv::Point(rulerV10P.x, rulerV10P.y - dw), CV_RGB(255, 255, 255), lw, 0);
        cv::line(img, cv::Point(rulerVZer.x, rulerVZer.y - dw), cv::Point(rulerVZer.x, rulerVZer.y + dw), CV_RGB(255, 255, 255), lw, 0);

        cv::Size textSize30N = cv::getTextSize("-30", fontFace, fontScale, thickness, &baseline);
        cv::Size textSize30P = cv::getTextSize("+30", fontFace, fontScale, thickness, &baseline);
        cv::Size textSize20N = cv::getTextSize("-20", fontFace, fontScale, thickness, &baseline);
        cv::Size textSize20P = cv::getTextSize("+20", fontFace, fontScale, thickness, &baseline);
        cv::Size textSize10N = cv::getTextSize("-10", fontFace, fontScale, thickness, &baseline);
        cv::Size textSize10P = cv::getTextSize("+10", fontFace, fontScale, thickness, &baseline);
        cv::Size textSizeZer = cv::getTextSize("0.0", fontFace, fontScale, thickness, &baseline);

        cv::Point textOrg30N(rulerV30N.x - textSize30N.width / 2, rulerV30N.y - textSize30N.height - tw);
        cv::Point textOrg30P(rulerV30P.x - textSize30P.width / 2, rulerV30P.y - textSize30P.height - tw);
        cv::Point textOrg20N(rulerV20N.x - textSize20N.width / 2, rulerV20N.y - textSize20N.height - tw);
        cv::Point textOrg20P(rulerV20P.x - textSize20P.width / 2, rulerV20P.y - textSize20P.height - tw);
        cv::Point textOrg10N(rulerV10N.x - textSize10N.width / 2, rulerV10N.y - textSize10N.height - tw);
        cv::Point textOrg10P(rulerV10P.x - textSize10P.width / 2, rulerV10P.y - textSize10P.height - tw);
        cv::Point textOrgZer(rulerVZer.x - textSizeZer.width / 2, rulerVZer.y - textSizeZer.height - tw);

        cv::putText(img, "-30", textOrg30N, fontFace, fontScale, CV_RGB(255, 255, 255), thickness);
        cv::putText(img, "+30", textOrg30P, fontFace, fontScale, CV_RGB(255, 255, 255), thickness);
        cv::putText(img, "-20", textOrg20N, fontFace, fontScale, CV_RGB(255, 255, 255), thickness);
        cv::putText(img, "+20", textOrg20P, fontFace, fontScale, CV_RGB(255, 255, 255), thickness);
        cv::putText(img, "-10", textOrg10N, fontFace, fontScale, CV_RGB(255, 255, 255), thickness);
        cv::putText(img, "+10", textOrg10P, fontFace, fontScale, CV_RGB(255, 255, 255), thickness);
        cv::putText(img, "0.0", textOrgZer, fontFace, fontScale, CV_RGB(255, 255, 255), thickness);

        if (boxes.size() > 0)
        {
            cv::Point center = (boxes[bigestIndex].br() + boxes[bigestIndex].tl()) * 0.5;
            cv::Point centerN(center.x - 10, RULER_H + 12);
            cv::Point centerP(center.x + 10, RULER_H + 12);
            cv::Point centerZ(center.x, RULER_H + 2);

            if ((rulerV30N.x <= centerZ.x) && (rulerV30P.x >= centerZ.x))
            {
                cv::line(img, centerN, centerZ, CV_RGB(255, 0, 0), 2, 0);
                cv::line(img, centerP, centerZ, CV_RGB(255, 0, 0), 2, 0);
            }
        }

        // Вывод результатов (опционально)
        if (true)
        {
            /*
            std::cout << "class_ids: ";
            for (auto element : class_ids)
            {
                std::cout << element << " ";
            }
            std::cout << std::endl;
            std::cout << "classes: ";
            for (auto element : classes)
            {
                std::cout << element << " ";
            }
            std::cout << std::endl;
            std::cout << "confidences: ";
            for (auto element : confidences)
            {
                std::cout << element << " ";
            }
            std::cout << std::endl;
            std::cout << "inference time: " << detector.get_inference() << std::endl;

            std::cout << detector.get_info();
            */

            cv::imshow("SarganYOLO", img);
        }

        // Отправляем результат в поток
        cv::imencode(".jpg", img, streamerBuf, params);

        // Выгрузка изображения в поток http://localhost:8080/sargan
        streamer.publish("/sargan", std::string(streamerBuf.begin(), streamerBuf.end()));
    }

    // Остановка стримера
    streamer.stop();

    // cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
