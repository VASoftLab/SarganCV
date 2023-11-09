#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <cerrno>
#include <filesystem>

#include "neuralnetdetector.h"

/** Параметры картинки */
static const float IMG_WIDTH = 640;
static const float IMG_HEIGHT  = 640;
static const float CAMERA_ANGLE = /*60*/78;

// Размеры прицела
static const float SIGHT_WIDTH = 50;

namespace fs = std::filesystem;

/** Функция поиска угла между целью и центром фрейма
 *   @param resolution - разрешение камеры по горизонтали
 *   @param cx - абциса центра цели
 *   @return угол между центром фрейма и центром цели
 */
int findAngleF(float resolution, int cx)
{
    return (cx * CAMERA_ANGLE / resolution) - CAMERA_ANGLE / 2;
}

int main()
{
    // Источник изображений по умолчанию
    cv::VideoCapture source(1);

    // Получить разрешение камеры по горизонтали и вертикали
    float FRAME_WIDTH = source.get(cv::CAP_PROP_FRAME_WIDTH);
    float FRAME_HEIGHT  = source.get(cv::CAP_PROP_FRAME_HEIGHT);

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
    NeuralNetDetector detector(model_path.u8string(), classes_path.u8string(), IMG_WIDTH, IMG_HEIGHT);

    cv::Mat frame;
    // Бесконечный цикл с захватом видео и детектором
    while(cv::waitKey(1) < 1)
    {
        source >> frame;
        if (frame.empty())
        {
            cv::waitKey();
            break;
        }
        // Отработка детектора
        cv::Mat img = detector.process(frame);
        // Результаты работы детектора
        std::vector<int> class_ids = detector.get_class_ids();
        std::vector<float> confidences = detector.get_confidences();
        std::vector<cv::Rect> boxes = detector.get_boxes();
        std::vector<std::string> classes = detector.get_classes();

        ///////////////////////////////////////////////////////////////////////
        // Наложение подложки
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

        // Поиск бокса с максимальной площадью
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

        // Расчет центра бокса с изображением
        if (boxes.size() > 0)
        {
            cv::Point center = (boxes[bigestIndex].br() + boxes[bigestIndex].tl()) * 0.5;
            //std::cout << "X: " << center.x << "; Y: " << center.y << std::endl;
            //std::string textCenter = "X: " + std::to_string(center.x) + "; Y: " + std::to_string(center.y);

            // Накладываем текст c координатами центра картинки
            //cv::putText(img, textCenter, cv::Point(10, img.rows / 2), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 255, 255),  2);

            // Координаты центра фрейма
            //float X2 = FRAME_WIDTH / 2;
            //float Y2 = FRAME_HEIGHT / 2;

            int angle = findAngleF(FRAME_WIDTH, center.x);
            //std::string textAngle = "Angle: " + std::to_string(angle);
            //cv::putText(img, textAngle, cv::Point(10, img.rows / 3), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 255, 255),  2);

            // Команда управления лево / право
            std::string direction = center.x > FRAME_WIDTH / 2 ? "RIGTH" : "LEFT";
            //std::string textCommand = "Command: " + textCMD;
            //cv::putText(img, textCommand, cv::Point(10, img.rows / 5), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 255, 255),  2);

            std::stringstream ssTime;
            ssTime << std::fixed << std::setprecision(2) << detector.get_inference();
            std::string inference = ssTime.str();

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

        // Отрисовка прицела в центре фрейма
        cv::Point sightPt1;
        cv::Point sightPt2;

        sightPt1.x = img.cols / 2 - SIGHT_WIDTH;
        sightPt1.y = img.rows / 2 - SIGHT_WIDTH;
        sightPt2.x = img.cols / 2 + SIGHT_WIDTH;
        sightPt2.y = img.rows / 2 + SIGHT_WIDTH;
        cv::rectangle(img, sightPt1, sightPt2, CV_RGB(255, 255, 255), 2, 0);

        // Перекрестие (основное изображение)
        cv::Point crossPtV1;
        cv::Point crossPtV2;
        cv::Point crossPtH1;
        cv::Point crossPtH2;

        crossPtV1.x = img.cols / 2;
        crossPtV1.y = img.rows / 2 - SIGHT_WIDTH / 4;
        crossPtV2.x = img.cols / 2;
        crossPtV2.y = img.rows / 2 + SIGHT_WIDTH / 4;

        crossPtH1.x = img.cols / 2 - SIGHT_WIDTH / 4;
        crossPtH1.y = img.rows / 2;
        crossPtH2.x = img.cols / 2 + SIGHT_WIDTH / 4;
        crossPtH2.y = img.rows / 2;

        cv::line(img, crossPtV1, crossPtV2, CV_RGB(255, 255, 255), 2, 0);
        cv::line(img, crossPtH1, crossPtH2, CV_RGB(255, 255, 255), 2, 0);

        if (boxes.size() > 0)
        {
            cv::Point center = (boxes[bigestIndex].br() + boxes[bigestIndex].tl()) * 0.5;

            // Отрисовка прицела в центре фрейма
            cv::Point detectPt1;
            cv::Point detectPt2;

            detectPt1.x = center.x - SIGHT_WIDTH / 2;
            detectPt1.y = center.y - SIGHT_WIDTH / 2;
            detectPt2.x = center.x + SIGHT_WIDTH / 2;
            detectPt2.y = center.y + SIGHT_WIDTH / 2;
            cv::rectangle(img, detectPt1, detectPt2, CV_RGB(255, 0, 0), 2, 0);

            // Перекрестие (фрейм объекта)
            cv::Point detectPtV1;
            cv::Point detectPtV2;
            cv::Point detectPtH1;
            cv::Point detectPtH2;

            detectPtV1.x = center.x;
            detectPtV1.y = center.y - SIGHT_WIDTH / 6;
            detectPtV2.x = center.x;
            detectPtV2.y = center.y + SIGHT_WIDTH / 6;

            detectPtH1.x = center.x - SIGHT_WIDTH / 6;
            detectPtH1.y = center.y;
            detectPtH2.x = center.x + SIGHT_WIDTH / 6;
            detectPtH2.y = center.y;

            cv::line(img, detectPtV1, detectPtV2, CV_RGB(255, 0, 0), 2, 0);
            cv::line(img, detectPtH1, detectPtH2, CV_RGB(255, 0, 0), 2, 0);
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
    }
    cv::waitKey(0);
    return 0;
}
