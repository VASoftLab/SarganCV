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

#include <QSettings>
#include <QUdpSocket>

#include "udppacket.h"

///////////////////////////////////////////////////////////////////////////////
// ГЛОБАЛЬНЫЕ НАСТРОЙКИ ПРИЛОЖЕНИЯ (ЗНАЧЕНИЯ ПО УМОЛЧАНИЮ)
//////////////////////////////////////////////////////////////////////////////////
static float IMG_WIDTH = 640;
static float IMG_HEIGHT  = 640;
static double CAMERA_FPS = 30;        // FPS камеры
static double VIDEO_FPS = 5;          // FPS видеоролика
static double FRAME_SCALE = 0.5;      // Коэф-т масштабирования картинки
static short VIDEO_DURATION_SEC = 10; // Длительность видеоролика
static short VIDEO_FILES_COUNT = 10;  // Максимальное кол-во видеофайлов

static std::string NN_DIR = "nn";     // Папка в которой лежит сеть

// Для отладки
static std::string NN_ONNX = "debug.onnx";    // Файл модели
static std::string NN_NAMES = "debug.names";  // Файл названий классов

// Для релиза
//static const std::string NN_ONNX = "ship.onnx";
//static const std::string NN_NAMES = "ship.names";

///////////////////////////////////////////////////////////////////////////////
// !!!ЗНАЧЕНИЕ УГЛА ОБЗОРА ДОЛЖНО БЫТЬ ИЗМЕНЕНО ПОД КАМЕРУ НА АППАРАТЕ!!!
///////////////////////////////////////////////////////////////////////////////
static float CAMERA_ANGLE = 80;
///////////////////////////////////////////////////////////////////////////////

// Размеры прицела
static float SIGHT_WIDTH = 50;

// Горизонтальная линейка
static int RULER_H = 40;

QUdpSocket udpSocket;
QHostAddress UDP_HOST;
int UDP_PORT;

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

std::string getTimeStamp()
{
    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    auto timer = system_clock::to_time_t(now);
    std::tm bt = *std::localtime(&timer);
    std::ostringstream oss;
    oss << std::put_time(&bt, "%d-%m-%Y %H:%M:%S"); // DD-MM-YYYY HH:MM:SS
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

std::string getVideoFileName()
{
    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    auto timer = system_clock::to_time_t(now);
    std::tm bt = *std::localtime(&timer);
    std::ostringstream oss;
    oss << std::put_time(&bt, "sargan_%d%m%Y%H%M%S.avi");
    return oss.str();
}

int main()
{
    ///////////////////////////////////////////////////////////////////////////
    // Чтение настроек
    ///////////////////////////////////////////////////////////////////////////
    fs::path pathToSettings = fs::current_path() / "settings.ini";
    QSettings settings (QString::fromStdString(pathToSettings.u8string()), QSettings::IniFormat);
    settings.beginGroup("Settings");
    IMG_WIDTH = settings.value("IMG_WIDTH").toFloat();
    IMG_HEIGHT = settings.value("IMG_HEIGHT").toFloat();
    CAMERA_FPS = settings.value("CAMERA_FPS").toDouble();
    VIDEO_FPS = settings.value("VIDEO_FPS").toDouble();
    FRAME_SCALE = settings.value("FRAME_SCALE").toDouble();
    VIDEO_DURATION_SEC = settings.value("VIDEO_DURATION_SEC").toUInt();
    VIDEO_FILES_COUNT = settings.value("VIDEO_FILES_COUNT").toUInt();
    NN_DIR = settings.value("NN_DIR").toString().toStdString();
    NN_ONNX = settings.value("NN_ONNX").toString().toStdString();
    NN_NAMES = settings.value("NN_NAMES").toString().toStdString();
    CAMERA_ANGLE = settings.value("CAMERA_ANGLE").toFloat();
    SIGHT_WIDTH = settings.value("SIGHT_WIDTH").toFloat();
    RULER_H = settings.value("RULER_H").toUInt();

    UDP_HOST = QHostAddress(settings.value("UDP_HOST").toString());
    UDP_PORT = settings.value("UDP_PORT").toUInt();

    std::cout << "IMG_WIDTH: " << IMG_WIDTH << std::endl;
    std::cout << "IMG_HEIGHT: " << IMG_HEIGHT << std::endl;
    std::cout << "CAMERA_FPS: " << CAMERA_FPS << std::endl;
    std::cout << "VIDEO_FPS: " << VIDEO_FPS << std::endl;
    std::cout << "FRAME_SCALE: " << FRAME_SCALE << std::endl;
    std::cout << "VIDEO_DURATION_SEC: " << VIDEO_DURATION_SEC << std::endl;
    std::cout << "VIDEO_FILES_COUNT: " << VIDEO_FILES_COUNT << std::endl;
    std::cout << "NN_DIR: " << NN_DIR << std::endl;
    std::cout << "NN_ONNX: " << NN_ONNX << std::endl;
    std::cout << "NN_NAMES: " << NN_NAMES << std::endl;
    std::cout << "CAMERA_ANGLE: " << CAMERA_ANGLE << std::endl;
    std::cout << "SIGHT_WIDTH: " << SIGHT_WIDTH << std::endl;
    std::cout << "RULER_H: " << RULER_H << std::endl;
    std::cout << "UDP_HOST: " << UDP_HOST.toString().toStdString() << std::endl;
    std::cout << "UDP_PORT: " << UDP_PORT << std::endl;

    cv::VideoCapture source;
    // Источник изображений по умолчанию

#ifdef _WIN32
    // source.open(0, cv::CAP_ANY);
    // source.open(0, cv::CAP_GSTREAMER);
    source.open(1, cv::CAP_DSHOW);
#else
    // source.open(0, cv::CAP_ANY);
    // source.open(0, cv::CAP_GSTREAMER);
    source.open(0, cv::CAP_DSHOW);
#endif

    source.set(cv::CAP_PROP_FPS, CAMERA_FPS);
    cv::Mat frame;

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

    fs::path nn_dir (NN_DIR);
    fs::path nn_onnx (NN_ONNX);
    fs::path nn_names (NN_NAMES);

    const fs::path model_path = fs::current_path() / nn_dir / nn_onnx;
    const fs::path classes_path = fs::current_path() / nn_dir / nn_names;

    if (DIAGNOSTIC_LOG)
        std::cout << model_path.u8string() << std::endl;

    NeuralNetDetector detector(model_path.u8string(), classes_path.u8string(), (int)IMG_WIDTH, (int)IMG_HEIGHT);

    ///////////////////////////////////////////////////////////////////////////
    // Набор глобальных переменных для основного фунционала
    ///////////////////////////////////////////////////////////////////////////
    cv::Mat img;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::string> classes;

    // Подложка
    double alpha = 0.5;
    cv::Mat overlay;

    // Вспомогательные переменные для поиска максимального объекта
    int bigestArea;
    int bigestIndex;
    int boxIndex;

    // Переменные для отрисовки
    cv::Point center;
    cv::Point centerN;
    cv::Point centerP;
    cv::Point centerZ;

    // Отрисовка прицела в центре фрейма цели
    cv::Point objectBoxPt1;
    cv::Point objectBoxPt2;

    // Перекрестие (фрейм объекта)
    cv::Point objectCrossPtV1;
    cv::Point objectCrossPtV2;
    cv::Point objectCrossPtH1;
    cv::Point objectCrossPtH2;

    // Перекрестие (основное изображение)
    cv::Point boardCrossPtV1;
    cv::Point boradCrossPtV2;
    cv::Point boardCrossPtH1;
    cv::Point boardCrossPtH2;

    // Центральная точка бортовой системы наведения
    cv::Point boardBoxPt1;
    cv::Point boardBoxPt2;

    // Направление прицела
    std::string direction;
    int angle;

    // Время работы детектора
    std::stringstream ssTime;
    std::string inference;
    std::string timestamp;

    // Строка инфорации
    std::string textInfo;
    std::string diagnosticInfo;

    // Угловая линейка
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

    double K = 1;
    double delta30;
    double delta20;
    double delta10;

    int lw = 2;
    int dw = 7;
    int tw = 5;

    cv::Size textSize30N;
    cv::Size textSize30P;
    cv::Size textSize20N;
    cv::Size textSize20P;
    cv::Size textSize10N;
    cv::Size textSize10P;
    cv::Size textSizeZer;

    cv::Point textOrg30N;
    cv::Point textOrg30P;
    cv::Point textOrg20N;
    cv::Point textOrg20P;
    cv::Point textOrg10N;
    cv::Point textOrg10P;
    cv::Point textOrgZer;

    // Переменная для сохранения видео
    cv::VideoWriter video;
    fs::path video_dir ("video");
    fs::path video_path;
    cv::Mat videoImg;

    std::vector<std::string> video_files;

    // Таймер записи
    std::chrono::time_point<std::chrono::system_clock> videoStartTime;
    std::chrono::time_point<std::chrono::system_clock> videoEndTime;
    bool isRecordStarted = false;

    // UDP Packet
    UDPPacket packet;

    ///////////////////////////////////////////////////////////////////////////
    // Удаляем старые файлы
    ///////////////////////////////////////////////////////////////////////////

    // Удаляет вместе с папкой video
    // std::filesystem::remove_all(fs::current_path() / video_dir);

    for (const auto& entry : std::filesystem::directory_iterator(fs::current_path() / video_dir))
        std::filesystem::remove_all(entry.path());

    ///////////////////////////////////////////////////////////////////////////
    // Бесконечный цикл с захватом видео и детектором
    ///////////////////////////////////////////////////////////////////////////

    while(cv::waitKey(1) < 1)
    {        
        ///////////////////////////////////////////////////////////////////////
        // Захват текущего кадра
        ///////////////////////////////////////////////////////////////////////
        source >> frame;

        if (frame.empty())
        {
            cv::waitKey();
            break;
        }

        // Создаем объект для записи видео
        if (!isRecordStarted)
        {
            video_path = fs::current_path() / video_dir / getVideoFileName();

            // Если размерность вектора больше допустимой, удаляем первый эл-т
            if (video_files.size() >= VIDEO_FILES_COUNT)
            {
                // Удалить файл
                std::filesystem::remove(video_files.front()); //video_files.at(0)
                // Извлечь имя удаленного файла из вектора
                video_files.erase(video_files.begin());
            }

            // Запоминаем файл в векторе
            video_files.push_back(video_path.u8string());

            std::cout << video_path.u8string() << std::endl;
            video = cv::VideoWriter(video_path.u8string(),
                                    //cv::VideoWriter::fourcc('X','V','I','D'),
                                    cv::VideoWriter::fourcc('D','I','V','X'),
                                    //cv::VideoWriter::fourcc('M','J','P','G'),
                                    VIDEO_FPS,
                                    cv::Size(FRAME_WIDTH * FRAME_SCALE,
                                             FRAME_HEIGHT * FRAME_SCALE));

            // TODO: Разобраться с флагами настройки качества изображения
            // video.set(cv::VIDEOWRITER_PROP_QUALITY, 10);

            // Запоминаем время начала записи
            videoStartTime = std::chrono::system_clock::now();

            // Установка флага - Старт записи
            isRecordStarted = true;
        }


        ///////////////////////////////////////////////////////////////////////
        // Отработка детектора
        ///////////////////////////////////////////////////////////////////////
        img = detector.process(frame);

        // Результаты работы детектора
        class_ids = detector.get_class_ids();
        confidences = detector.get_confidences();
        boxes = detector.get_boxes();
        classes = detector.get_classes();
        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////
        // Наложение подложки
        ///////////////////////////////////////////////////////////////////////        
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
        bigestArea = INT_MIN;
        bigestIndex = -1;
        boxIndex = -1;

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
            center = (boxes[bigestIndex].br() + boxes[bigestIndex].tl()) * 0.5;

            // Отрисовка прицела в центре фрейма цели            

            objectBoxPt1.x = center.x - (int)(SIGHT_WIDTH / 2);
            objectBoxPt1.y = center.y - (int)(SIGHT_WIDTH / 2);
            objectBoxPt2.x = center.x + (int)(SIGHT_WIDTH / 2);
            objectBoxPt2.y = center.y + (int)(SIGHT_WIDTH / 2);
            cv::rectangle(img, objectBoxPt1, objectBoxPt2, CV_RGB(255, 0, 0), 2, 0);

            // Перекрестие (фрейм объекта)
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

        // Координаты бокса прицела
        boardBoxPt1.x = (int)(img.cols / 2) - (int)SIGHT_WIDTH;
        boardBoxPt1.y = (int)(img.rows / 2) - (int)SIGHT_WIDTH;
        boardBoxPt2.x = (int)(img.cols / 2) + (int)SIGHT_WIDTH;
        boardBoxPt2.y = (int)(img.rows / 2) + (int)SIGHT_WIDTH;
        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////
        // Расчет управления
        ///////////////////////////////////////////////////////////////////////
        timestamp = getTimeStamp(); // Временная метка - TimeStamp
        if (boxes.size() > 0)
        {
            // Расчет центра бокса с обнаруженной целью
            center = (boxes[bigestIndex].br() + boxes[bigestIndex].tl()) * 0.5;

            // Угол между прицелом и целью
            angle = findAngleF(FRAME_WIDTH, center.x);

            // Команда управления лево / право
            direction = center.x > FRAME_WIDTH / 2 ? "RIGHT" : "LEFT";

            // Если цель находится в границах прицела - удерживаем курс
            //if ((boardBoxPt1.x <= center.x) && (center.x <= boardBoxPt2.x) &&
            //    (boardBoxPt1.y <= center.y) && (center.y <= boardBoxPt2.y))
            //{
            //    direction = "HOLD";
            //}

            // Алгоритм удержания цели только по оси абцисс
            if ((boardBoxPt1.x <= center.x) && (center.x <= boardBoxPt2.x))
                direction = "HOLD";

            // Время работы детектора
            ssTime.str(std::string()); // Очистка строкового стримера
            ssTime << std::fixed << std::setprecision(2) << detector.get_inference();
            inference = ssTime.str();

            // Строка инфорации
            textInfo = " CMD: (" + direction + ":" + std::to_string(angle) + ")" +
                       " TARGET: (" + std::to_string(center.x) + ";" + std::to_string(center.y) + ")" +
                       // " RES: (" + std::to_string((int)FRAME_WIDTH) + "x" + std::to_string((int)FRAME_HEIGHT) + ")" +
                       " TIME: " + inference + " " + timestamp;
            cv::putText(img, textInfo, cv::Point(10, img.rows - 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 0, 255), 1);

            if (COMMAND_LOG)
            {
                diagnosticInfo = "CMD:\t(" + direction + ":" + std::to_string(angle) + ")" + "\tTIME: " + inference + "\t" + timestamp;
                std::cout << diagnosticInfo << std::endl;
            }

            ///////////////////////////////////////////////////////////////////
            // Формируем пакет для передачи по UDP
            ///////////////////////////////////////////////////////////////////
            if (direction == "LEFT")
                packet.udpCMD = (std::int8_t)(-1);
            else if (direction == "HOLD")
                packet.udpCMD = (std::int8_t)(0);
            else if (direction == "RIGHT")
                packet.udpCMD = (std::int8_t)(1);

            packet.udpANG = angle;

            udpSocket.writeDatagram(packet.toByteArray(), UDP_HOST, UDP_PORT);
            ///////////////////////////////////////////////////////////////////
        }
        else
        {
            // textInfo = " RES: (" + std::to_string((int)FRAME_WIDTH) + "x" + std::to_string((int)FRAME_HEIGHT) + ")";
            textInfo = "TIME: " + timestamp;
            cv::putText(img, textInfo, cv::Point(10, img.rows - 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 0, 255), 1);

            packet.udpCMD = (std::int8_t)(-101);
            packet.udpANG = 0;
            udpSocket.writeDatagram(packet.toByteArray(), UDP_HOST, UDP_PORT);
        }
        ///////////////////////////////////////////////////////////////////////
        // Отрисовка бортовых бокса и прицела
        ///////////////////////////////////////////////////////////////////////

        if (direction == "HOLD")
            cv::rectangle(img, boardBoxPt1, boardBoxPt2, CV_RGB(255, 0, 0), 2, 0);
        else
            cv::rectangle(img, boardBoxPt1, boardBoxPt2, CV_RGB(255, 255, 255), 2, 0);

        // Перекрестие (основное изображение)
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
        delta30 = (double)RULER_H * tan(30 * M_PI / 180) * K;
        delta20 = (double)RULER_H * tan(20 * M_PI / 180) * K;
        delta10 = (double)RULER_H * tan(10 * M_PI / 180) * K;

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

        cv::line(img, rulerV30N, rulerV30P, CV_RGB(255, 255, 255), lw, 0);
        cv::line(img, rulerV30N, cv::Point(rulerV30N.x, rulerV30N.y - dw), CV_RGB(255, 255, 255), lw, 0);
        cv::line(img, rulerV30P, cv::Point(rulerV30P.x, rulerV30P.y - dw), CV_RGB(255, 255, 255), lw, 0);
        cv::line(img, rulerV20N, cv::Point(rulerV20N.x, rulerV20N.y - dw), CV_RGB(255, 255, 255), lw, 0);
        cv::line(img, rulerV20P, cv::Point(rulerV20P.x, rulerV20P.y - dw), CV_RGB(255, 255, 255), lw, 0);
        cv::line(img, rulerV10N, cv::Point(rulerV10N.x, rulerV10N.y - dw), CV_RGB(255, 255, 255), lw, 0);
        cv::line(img, rulerV10P, cv::Point(rulerV10P.x, rulerV10P.y - dw), CV_RGB(255, 255, 255), lw, 0);
        cv::line(img, cv::Point(rulerVZer.x, rulerVZer.y - dw), cv::Point(rulerVZer.x, rulerVZer.y + dw), CV_RGB(255, 255, 255), lw, 0);

        textSize30N = cv::getTextSize("-30", fontFace, fontScale, thickness, &baseline);
        textSize30P = cv::getTextSize("+30", fontFace, fontScale, thickness, &baseline);
        textSize20N = cv::getTextSize("-20", fontFace, fontScale, thickness, &baseline);
        textSize20P = cv::getTextSize("+20", fontFace, fontScale, thickness, &baseline);
        textSize10N = cv::getTextSize("-10", fontFace, fontScale, thickness, &baseline);
        textSize10P = cv::getTextSize("+10", fontFace, fontScale, thickness, &baseline);
        textSizeZer = cv::getTextSize("0.0", fontFace, fontScale, thickness, &baseline);

        textOrg30N = cv::Point(rulerV30N.x - textSize30N.width / 2, rulerV30N.y - textSize30N.height - tw);
        textOrg30P = cv::Point(rulerV30P.x - textSize30P.width / 2, rulerV30P.y - textSize30P.height - tw);
        textOrg20N = cv::Point(rulerV20N.x - textSize20N.width / 2, rulerV20N.y - textSize20N.height - tw);
        textOrg20P = cv::Point(rulerV20P.x - textSize20P.width / 2, rulerV20P.y - textSize20P.height - tw);
        textOrg10N = cv::Point(rulerV10N.x - textSize10N.width / 2, rulerV10N.y - textSize10N.height - tw);
        textOrg10P = cv::Point(rulerV10P.x - textSize10P.width / 2, rulerV10P.y - textSize10P.height - tw);
        textOrgZer = cv::Point(rulerVZer.x - textSizeZer.width / 2, rulerVZer.y - textSizeZer.height - tw);

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
            //std::cout << std::endl << "class_ids: ";
            //for (auto element : class_ids)
            //    std::cout << element << " ";

            //std::cout << std::endl << "classes: ";
            //for (auto element : classes)
            //    std::cout << element << " ";

            //std::cout << std::endl << "confidences: ";
            //for (auto element : confidences)
            //    std::cout << element << " ";

            //std::cout << std::endl << "inference time: " << detector.get_inference() << std::endl;
            //std::cout << std::endl << detector.get_info();

            // Дублируем видео в окне
            cv::imshow("SarganYOLO", img);
        }

        // Отправляем результат в поток
        cv::imencode(".jpg", img, streamerBuf, params);

        // Выгрузка изображения в поток http://localhost:8080/sargan
        streamer.publish("/sargan", std::string(streamerBuf.begin(), streamerBuf.end()));

        // Сохраняем в видеофайл

        // Уменьшаем картинку в два раза
        resize(img, videoImg, cv::Size(), FRAME_SCALE, FRAME_SCALE, cv::INTER_CUBIC);
        video.write(videoImg);

        videoEndTime = std::chrono::system_clock::now();

        // Новый видео файл каждые 10 секунд
        if (std::chrono::duration_cast<std::chrono::milliseconds>(videoEndTime - videoStartTime).count() / 1000 > VIDEO_DURATION_SEC)
        {
            video.release();
            isRecordStarted = false;
        }
    }

    // Остановка стримера
    streamer.stop();

    // Освобождение занятых ресурсов
    source.release();
    video.release();

    cv::destroyAllWindows();
    return 0;
}
