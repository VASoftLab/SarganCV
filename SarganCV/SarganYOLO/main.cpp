#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <cerrno>
#include <filesystem>

#include "nadjieb/streamer.hpp"
using MJPEGStreamer = nadjieb::MJPEGStreamer;

#include "neuralnetdetector.h"

/** Параметры картинки */
static const float IMG_WIDTH = 640;
static const float IMG_HEIGHT  = 640;
///////////////////////////////////////////////////////////////////////////////
// !!!ЗНАЧЕНИЕ УГЛА ОБЗОРА ДОЛЖНО БЫТЬ ИЗМЕНЕНО ПОД КАМЕРУ НА АППАРАТЕ!!!
///////////////////////////////////////////////////////////////////////////////
static const float CAMERA_ANGLE = /*60*/78;
///////////////////////////////////////////////////////////////////////////////

// Размеры прицела
static const float SIGHT_WIDTH = 50;

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

int main()
{
    // Источник изображений по умолчанию
    cv::VideoCapture source(0);

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
    double FRAME_HEIGHT  = source.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::cout << "Camera resolution: " << FRAME_WIDTH << " x " << FRAME_HEIGHT << std::endl;

    // Путь к модели и файлу с классами
    fs::path nn_dir ("nn");
    fs::path nn_onnx ("yolov5s.onnx");
    fs::path nn_names ("coco.names");

    const fs::path model_path = fs::current_path() / nn_dir / nn_onnx;
    const fs::path classes_path = fs::current_path() / nn_dir / nn_names;

    std::cout << model_path.u8string() << std::endl;

    // TODO -- Разобраться, почему падает код с прямоугольными размерами фрейма
    // NeuralNetDetector detector(model_path.u8string(), classes_path.u8string(), FRAME_WIDTH, FRAME_HEIGHT);
    NeuralNetDetector detector(model_path.u8string(), classes_path.u8string(), (int)IMG_WIDTH, (int)IMG_HEIGHT);

    cv::Mat frame;

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

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
