#include "neuralnetdetector.h"

NeuralNetDetector::NeuralNetDetector(const std::string model, const std::string classes)
{
    if (!init_network(model, classes))
    {
        std::cout << "Neural network been inited!" << std::endl;
        std::cout << "Input width: " << input_width << std::endl;
        std::cout << "Input height: " << input_height << std::endl;
    }
    else
    {
        std::cerr << "Failed to init neural network!" << std::endl;
    }
}

NeuralNetDetector::NeuralNetDetector(const std::string model, const std::string classes, int width, int height)
{
    input_width = width;
    input_height = height;
    if (!init_network(model, classes))
    {
        std::cout << "Neural network been inited!" << std::endl;
        std::cout << "Input width: " << input_width << std::endl;
        std::cout << "Input height: " << input_height << std::endl;
    }
    else
    {
        std::cerr << "Failed to init neural network!" << std::endl;
    }
}

#ifdef _WIN32
    errno_t NeuralNetDetector::read_classes(const std::string file_path)
#else
    error_t NeuralNetDetector::read_classes(const std::string file_path)
#endif
{
    std::ifstream classes_file(file_path);
    std::string line;

    if (!classes_file)
    {
        std::cerr << "Failed to open classes names!\n";
        return ENOENT;
    }
    while (std::getline(classes_file, line))
    {
        classes.push_back(line);
    }
    classes_file.close();

    return 0;
}

#ifdef _WIN32
errno_t NeuralNetDetector::init_network(const std::string model_path, const std::string classes_path)
#else
error_t NeuralNetDetector::init_network(const std::string model_path, const std::string classes_path)
#endif
{
#ifdef _WIN32
    errno_t err = read_classes(classes_path);
#else
    error_t err = read_classes(classes_path);
#endif
    if (err == 0)
    {
        network = cv::dnn::readNetFromONNX(model_path);
        if (network.empty())
        {
            return ENETDOWN;
        }
        else
        {
            network.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            network.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
    }

    return err;
}

// Draw the predicted bounding box.
void NeuralNetDetector::draw_label(cv::Mat& img, std::string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseline;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS, &baseline);
    top = std::max(top, label_size.height);
    // Top left corner.
    cv::Point tlc = cv::Point(left, top);
    // Bottom right corner.
    cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseline);
    // Draw black rectangle.
    cv::rectangle(img, tlc, brc, BLACK, cv::FILLED);
    // Put the label on the black rectangle.
    cv::putText(img, label, cv::Point(left, top + label_size.height), cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, YELLOW, THICKNESS);
}

std::vector<cv::Mat> NeuralNetDetector::pre_process(cv::Mat &img, cv::dnn::Net &net)
{
    // Convert to blob.
    cv::Mat blob;
    cv::dnn::blobFromImage(img, blob, 1.0 / 255.0, cv::Size(input_width, input_height), cv::Scalar(), true, false);
    net.setInput(blob);
    // Forward propagate.
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    return outputs;
}

cv::Mat NeuralNetDetector::post_process(cv::Mat &img, std::vector<cv::Mat> &outputs, const std::vector<std::string> &class_name) {
    // Initialize vectors to hold respective outputs while unwrapping detections.
    cv::Mat ret = img.clone();
    classes_id_set.clear();
    confidences_set.clear();
    boxes_set.clear();
    classes_set.clear();
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // Resizing factor.
    float x_factor = img.cols / (float)input_width;
    float y_factor = img.rows / (float)input_height;

    float *data = (float *)outputs[0].data;

    const size_t dimensions = class_name.size() + 5;
    const size_t rows = 25200;
    // Iterate through 25200 detections.
    for (size_t i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float * classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            cv::Mat scores(1, (int)class_name.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire index of best class score.
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD)
            {
                // Store class ID and confidence in the pre-defined respective vectors.
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        // Jump to the next column.
        data += dimensions;
    }

    // Perform Non Maximum Suppression and draw predictions.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    int bigestArea = INT_MIN;
    int bigestIndex = -1;
    int boxIndex = -1;

    for (size_t i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];

        boxIndex++;
        if (box.area() > bigestArea)
        {
            bigestIndex = boxIndex;
            bigestArea = box.area();
        }
    }

    if (bigestIndex > -1)
    {
        int idx = indices[bigestIndex];
        cv::Rect box = boxes[idx];

        boxes_set.push_back(box);
        confidences_set.push_back(confidences[idx]);
        classes_id_set.push_back(class_ids[idx]);
        classes_set.push_back(class_name[class_ids[idx]]);

        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        cv::rectangle(ret, cv::Point(left, top), cv::Point(left + width, top + height), GREEN, 3*THICKNESS);
        // Get the label for the class name and its confidence.
        std::string label = cv::format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ": " + label;
        if (DRAW_LABEL)
        {
            // Draw class labels.
            draw_label(ret, label, left, top);
        }
    }

    return ret;
}

cv::Mat NeuralNetDetector::process(cv::Mat &img)
{
    std::vector<cv::Mat> detections;
    detections = pre_process(img, network);
    cv::Mat res = post_process(img, detections, NeuralNetDetector::classes);
    // Put efficiency information.
    // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency();
    NeuralNetDetector::inference_time = network.getPerfProfile(layersTimes) / (float)freq;
    return res;
}

std::string NeuralNetDetector::get_info(void)
{
    std::string str = "";
    for (size_t i = 0; i < classes_id_set.size(); i++)
    {
        str += classes_set[i];
        str += ": ";
        str += std::to_string(confidences_set[i]);
        str += "\n";
    }
    return str;
}
