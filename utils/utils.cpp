//
// Created by hao on 2021/6/15.
//

#include "utils.h"
#include <unistd.h>
#include "fstream"
#include <string.h>
#include <opencv2/highgui.hpp>
//const swr::imsee_types::Resolution RESOLUTION = swr::imsee_types::Resolution::RES_640X400;
struct Box
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

typedef struct detection{
    Box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
    float *uc; // Gaussian_YOLOv3 - tx,ty,tw,th uncertainty
    int points; // bit-0 - center, bit-1 - top-left-corner, bit-2 - bottom-right-corner
    float *embeddings;  // embeddings for tracking
    int embedding_size;
    float sim;
    int track_id;
} detection;
//char **get_labels_custom(char *filename, int *size)
//{
//    list *plist = get_paths(filename);
//    if(size) *size = plist->size;
//    char **labels = (char **)list_to_array(plist);
//    free_list(plist);
//    return labels;
//}
void ReadFilesFromDir(const std::string &path_to_dir
                      , std::vector<std::string> *image_name_list)
{
    DIR *dir;
    dir = opendir(path_to_dir.c_str());
    struct dirent *ent;
    // CHECK_NOTNULL(dir);
    while ((ent = readdir(dir)) != nullptr)
    {
        auto name = std::string(ent->d_name);
        // ignore "." ".."
        if (name.size() < 4)
        {
            continue;
        }
        auto suffix = name.substr(name.size() - 4, 4);
        if (suffix == ".png" || suffix == ".jpg")
        {
            // filter image
            image_name_list->emplace_back(name);
        }
    }

    closedir(dir);
}

void ReadFile(std::string srcFile, std::vector<std::string> &image_files)
{
    if (not access(srcFile.c_str(), 0) == 0)
    {
        ERROR_PRINT("no such File (" + srcFile + ")");
        return;
    }

    std::ifstream fin(srcFile.c_str());

    if (!fin.is_open())
    {
        ERROR_PRINT("read file error (" + srcFile + ")");
        exit(0);
    }

    std::string s;
    while (getline(fin, s))
    {
        image_files.push_back(s);
    }

    fin.close();
}

box_label *read_boxes(char *filename, int *n)
{
    box_label* boxes = (box_label*)xcalloc(1, sizeof(box_label));
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Can't open label file. (This can be normal only if you use MSCOCO): %s \n", filename);
        //file_error(filename);
        FILE* fw = fopen("bad.list", "a");
        fwrite(filename, sizeof(char), strlen(filename), fw);
        char *new_line = "\n";
        fwrite(new_line, sizeof(char), strlen(new_line), fw);
        fclose(fw);
        if (0) {
            printf("\n Error in read_boxes() \n");
            getchar();
        }

        *n = 0;
        return boxes;
    }
    const int max_obj_img = 4000;// 30000;
    const int img_hash = (custom_hash(filename) % max_obj_img)*max_obj_img;
    //printf(" img_hash = %d, filename = %s; ", img_hash, filename);
    float x, y, h, w;
    int id;
    int count = 0;
    while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5){
        boxes = (box_label*)xrealloc(boxes, (count + 1) * sizeof(box_label));
        boxes[count].track_id = count + img_hash;
        //printf(" boxes[count].track_id = %d, count = %d \n", boxes[count].track_id, count);
        boxes[count].id = id;
        boxes[count].x = x;
        boxes[count].y = y;
        boxes[count].h = h;
        boxes[count].w = w;
        boxes[count].left   = x - w/2;
        boxes[count].right  = x + w/2;
        boxes[count].top    = y - h/2;
        boxes[count].bottom = y + h/2;
        ++count;
    }
    fclose(file);
    *n = count;
    return boxes;
}

box_label *read_boxes1(char *filename, int *n)
{
    box_label* boxes = (box_label*)xcalloc(1, sizeof(box_label));
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Can't open label file. (This can be normal only if you use MSCOCO): %s \n", filename);
        //file_error(filename);
        FILE* fw = fopen("bad.list", "a");
        fwrite(filename, sizeof(char), strlen(filename), fw);
        char *new_line = "\n";
        fwrite(new_line, sizeof(char), strlen(new_line), fw);
        fclose(fw);
        if (0) {
            printf("\n Error in read_boxes() \n");
            getchar();
        }

        *n = 0;
        return boxes;
    }
    const int max_obj_img = 4000;// 30000;
    const int img_hash = (custom_hash(filename) % max_obj_img)*max_obj_img;
    //printf(" img_hash = %d, filename = %s; ", img_hash, filename);
    float x, y, h, w;
    int id;
    int count = 0;
    while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5){
        boxes = (box_label*)xrealloc(boxes, (count + 1) * sizeof(box_label));
        boxes[count].track_id = count + img_hash;
        //printf(" boxes[count].track_id = %d, count = %d \n", boxes[count].track_id, count);
        boxes[count].id = id;
        boxes[count].x = x;
        boxes[count].y = y;
        boxes[count].h = h;
        boxes[count].w = w;
        boxes[count].left   = x - w/2;
        boxes[count].right  = x + w/2;
        boxes[count].top    = y - h/2;
        boxes[count].bottom = y + h/2;
        ++count;
    }
    fclose(file);
    *n = count;
    return boxes;
}

void replace_image_to_label(const char* input_path, char* output_path)
{
    find_replace(input_path, "/images/train2017/", "/labels/train2017/", output_path);    // COCO
    find_replace(output_path, "/images/val2017/", "/labels/val2017/", output_path);        // COCO
    find_replace(output_path, "/JPEGImages/", "/labels/", output_path);    // PascalVOC
    find_replace(output_path, "\\images\\train2017\\", "\\labels\\train2017\\", output_path);    // COCO
    find_replace(output_path, "\\images\\val2017\\", "\\labels\\val2017\\", output_path);        // COCO

    find_replace(output_path, "\\images\\train2014\\", "\\labels\\train2014\\", output_path);    // COCO
    find_replace(output_path, "\\images\\val2014\\", "\\labels\\val2014\\", output_path);        // COCO
    find_replace(output_path, "/images/train2014/", "/labels/train2014/", output_path);    // COCO
    find_replace(output_path, "/images/val2014/", "/labels/val2014/", output_path);        // COCO

    find_replace(output_path, "\\JPEGImages\\", "\\labels\\", output_path);    // PascalVOC
    find_replace(output_path, "/images/", "/labels/", output_path);    // COCO
    find_replace(output_path, "/VOC2007/JPEGImages/", "/VOC2007/labels/", output_path);        // PascalVOC
    find_replace(output_path, "/VOC2012/JPEGImages/", "/VOC2012/labels/", output_path);        // PascalVOC

    if (!strcmp(input_path, output_path))
    {
        find_replace(output_path, "/images/", "/labels/", output_path);
        find_replace(output_path, "\\images\\", "\\labels\\", output_path);
    }
    trim(output_path);

    // replace only ext of files
    find_replace_extension(output_path, ".jpg", ".txt", output_path);
    find_replace_extension(output_path, ".JPG", ".txt", output_path); // error
    find_replace_extension(output_path, ".jpeg", ".txt", output_path);
    find_replace_extension(output_path, ".JPEG", ".txt", output_path);
    find_replace_extension(output_path, ".png", ".txt", output_path);
    find_replace_extension(output_path, ".PNG", ".txt", output_path);
    find_replace_extension(output_path, ".bmp", ".txt", output_path);
    find_replace_extension(output_path, ".BMP", ".txt", output_path);
    find_replace_extension(output_path, ".ppm", ".txt", output_path);
    find_replace_extension(output_path, ".PPM", ".txt", output_path);
    find_replace_extension(output_path, ".tiff", ".txt", output_path);
    find_replace_extension(output_path, ".TIFF", ".txt", output_path);

    // Check file ends with txt:
    if(strlen(output_path) > 4) {
        char *output_path_ext = output_path + strlen(output_path) - 4;
        if( strcmp(".txt", output_path_ext) != 0){
            fprintf(stderr, "Failed to infer label file name (check image extension is supported): %s \n", output_path);
        }
    }else{
        fprintf(stderr, "Label file name is too short: %s \n", output_path);
    }
}


void *xmalloc(size_t size) {
    void *ptr=malloc(size);
    if(!ptr) {
        malloc_error();
    }
    return ptr;
}

void *xrealloc(void *ptr, size_t size) {
    ptr=realloc(ptr,size);
    if(!ptr) {
        realloc_error();
    }
    return ptr;
}

void *xcalloc(size_t nmemb, size_t size) {
    void *ptr=calloc(nmemb,size);
    if(!ptr) {
        calloc_error();
    }
    return ptr;
}

unsigned long custom_hash(char *str)
{
    unsigned long hash = 5381;
    int c;

    while (c = *str++)
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}

void find_replace(const char* str, char* orig, char* rep, char* output)
{
    char* buffer = (char*)calloc(8192, sizeof(char));
    char *p;

    sprintf(buffer, "%s", str);
    if (!(p = strstr(buffer, orig))) {  // Is 'orig' even in 'str'?
        sprintf(output, "%s", buffer);
        free(buffer);
        return;
    }

    *p = '\0';

    sprintf(output, "%s%s%s", buffer, rep, p + strlen(orig));
    free(buffer);
}

void trim(char *str)
{
    char* buffer = (char*)xcalloc(8192, sizeof(char));
    sprintf(buffer, "%s", str);

    char *p = buffer;
    while (*p == ' ' || *p == '\t') ++p;

    char *end = p + strlen(p) - 1;
    while (*end == ' ' || *end == '\t') {
        *end = '\0';
        --end;
    }
    sprintf(str, "%s", p);

    free(buffer);
}

void find_replace_extension(char *str, char *orig, char *rep, char *output)
{
    char* buffer = (char*)calloc(8192, sizeof(char));

    sprintf(buffer, "%s", str);
    char *p = strstr(buffer, orig);
    int offset = (p - buffer);
    int chars_from_end = strlen(buffer) - offset;
    if (!p || chars_from_end != strlen(orig)) {  // Is 'orig' even in 'str' AND is 'orig' found at the end of 'str'?
        sprintf(output, "%s", buffer);
        free(buffer);
        return;
    }

    *p = '\0';
    sprintf(output, "%s%s%s", buffer, rep, p + strlen(orig));
    free(buffer);
}

void malloc_error()
{
    fprintf(stderr, "xMalloc error - possibly out of CPU RAM \n");
    exit(EXIT_FAILURE);
}

void calloc_error()
{
    fprintf(stderr, "Calloc error - possibly out of CPU RAM \n");
    exit(EXIT_FAILURE);
}

void realloc_error()
{
    fprintf(stderr, "Realloc error - possibly out of CPU RAM \n");
    exit(EXIT_FAILURE);
}



std::string getCurrentExePath()
{
    char szPath[256] = {0};
    int ret = readlink("/proc/self/exe", szPath, sizeof(szPath) - 1);
    std::string path(szPath, ret);
    size_t last_ = path.find_last_of('/');
    if (std::string::npos != last_)
    {
        return path.substr(0, last_ + 1);
    }
    return path;
}

std::string getCurrentExeName()
{
    char szPath[256] = {0};
    int ret = readlink("/proc/self/exe", szPath, sizeof(szPath) - 1);
    std::string path(szPath, ret);
    size_t last_ = path.find_last_of('/');
    if (std::string::npos != last_)
    {
        return path.substr(last_ + 1);
    }
    return path;
}
