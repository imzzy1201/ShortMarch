#include "Film.h"

Film::Film(grassland::graphics::Core *core, int width, int height)
    : core_(core), width_(width), height_(height), sample_count_(0) {

    CreateImages();
    Reset();
}

Film::~Film() {
    accumulated_color_image_.reset();
    accumulated_samples_image_.reset();
    output_image_.reset();
    velocity_image_.reset();
}

void Film::CreateImages() {
    // Create accumulated color image (RGBA32F for high precision accumulation)
    core_->CreateImage(width_, height_, grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
                       &accumulated_color_image_);

    // Create accumulated samples image (R32_SINT to count samples)
    core_->CreateImage(width_, height_, grassland::graphics::IMAGE_FORMAT_R32_SINT, &accumulated_samples_image_);

    // Create output image (RGBA32F for final result)
    core_->CreateImage(width_, height_, grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &output_image_);

    // Create velocity image (RG32F for motion vectors)
    core_->CreateImage(width_, height_, grassland::graphics::IMAGE_FORMAT_R32G32_SFLOAT, &velocity_image_);
}

void Film::Reset() {
    // Clear accumulated color to black
    std::unique_ptr<grassland::graphics::CommandContext> cmd_context;
    core_->CreateCommandContext(&cmd_context);
    cmd_context->CmdClearImage(accumulated_color_image_.get(), {{0.0f, 0.0f, 0.0f, 0.0f}});
    cmd_context->CmdClearImage(accumulated_samples_image_.get(), {{0, 0, 0, 0}});
    cmd_context->CmdClearImage(output_image_.get(), {{0.0f, 0.0f, 0.0f, 0.0f}});
    cmd_context->CmdClearImage(velocity_image_.get(), {{0.0f, 0.0f, 0.0f, 0.0f}});
    core_->SubmitCommandContext(cmd_context.get());

    sample_count_ = 0;
    grassland::LogInfo("Film accumulation reset");
}

void Film::DevelopToOutput() {
    // This would ideally be done in a compute shader for efficiency
    // For now, we'll do it on the CPU (simple but potentially slow)

    if (sample_count_ == 0) {
        return;
    }

    // Download accumulated color and samples
    size_t color_size = width_ * height_ * sizeof(float) * 4;
    std::vector<float> accumulated_colors(width_ * height_ * 4);
    accumulated_color_image_->DownloadData(accumulated_colors.data());

    // Divide by sample count to get average
    std::vector<float> output_colors(width_ * height_ * 4);
    for (int i = 0; i < width_ * height_ * 4; i++) {
        output_colors[i] = accumulated_colors[i] / static_cast<float>(sample_count_);
    }

    // Upload to output image
    output_image_->UploadData(output_colors.data());
}

// void Film::DevelopToOutput() {
//     if (sample_count_ == 0) {
//         return;
//     }

//     // Parameters (tweakable)
//     const int   BLUR_SAMPLES = 8;
//     const float MOTION_BLUR_STRENGTH = 2.0f;

//     const int pixel_count = width_ * height_;

//     std::vector<float> accumulated_colors(pixel_count * 4);
//     accumulated_color_image_->DownloadData(accumulated_colors.data());

//     std::vector<float> velocities(pixel_count * 2);
//     velocity_image_->DownloadData(velocities.data());

//     std::vector<float> blur_accum(pixel_count * 4, 0.0f);

//     for (int y = 0; y < height_; ++y) {
//         for (int x = 0; x < width_; ++x) {
//             int idx = y * width_ + x;

//             float r = accumulated_colors[idx * 4 + 0] / sample_count_;
//             float g = accumulated_colors[idx * 4 + 1] / sample_count_;
//             float b = accumulated_colors[idx * 4 + 2] / sample_count_;

//             float vx = velocities[idx * 2 + 0] * MOTION_BLUR_STRENGTH * width_;
//             float vy = velocities[idx * 2 + 1] * MOTION_BLUR_STRENGTH * height_;

//             if (vx * vx + vy * vy < 1e-6f) {
//                 int out = idx * 4;
//                 blur_accum[out + 0] += r;
//                 blur_accum[out + 1] += g;
//                 blur_accum[out + 2] += b;
//                 blur_accum[out + 3] += 1.0f;
//                 continue;
//             }

//             for (int i = 0; i < BLUR_SAMPLES; ++i) {
//                 float t = (i + 0.5f)/ BLUR_SAMPLES;
//                 int sx = int(x + vx * t);
//                 int sy = int(y + vy * t);

//                 if (sx < 0 || sx >= width_ || sy < 0 || sy >= height_)
//                     continue;

//                 int out = (sy * width_ + sx) * 4;
//                 float w = 1.0f / BLUR_SAMPLES;

//                 blur_accum[out + 0] += r * w;
//                 blur_accum[out + 1] += g * w;
//                 blur_accum[out + 2] += b * w;
//                 blur_accum[out + 3] += w;
//             }
//         }
//     }

//     // Normalize and write to output
//     std::vector<float> output_colors(pixel_count * 4, 0.0f);

//     for (int i = 0; i < pixel_count; ++i) {
//         float w = blur_accum[i * 4 + 3];
//         if (w > 0.0f) {
//             output_colors[i * 4 + 0] = blur_accum[i * 4 + 0] / w;
//             output_colors[i * 4 + 1] = blur_accum[i * 4 + 1] / w;
//             output_colors[i * 4 + 2] = blur_accum[i * 4 + 2] / w;
//             output_colors[i * 4 + 3] = 1.0f;
//         }
//     }

//     output_image_->UploadData(output_colors.data());
// }

void Film::Resize(int width, int height) {
    if (width == width_ && height == height_) {
        return;
    }

    width_ = width;
    height_ = height;

    // Recreate images with new dimensions
    accumulated_color_image_.reset();
    accumulated_samples_image_.reset();
    output_image_.reset();

    CreateImages();
    Reset();

    grassland::LogInfo("Film resized to {}x{}", width, height);
}
