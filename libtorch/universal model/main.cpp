#include <torch/torch.h>
#include "Header.h"


// Define a new Module.
struct Discriminator : torch::nn::Module {
    Discriminator() {
        // Construct and register two Linear submodules.
        fc1 = register_module("fc1", torch::nn::Linear(d[0], d[1]));
        fc2 = register_module("fc2", torch::nn::Linear(d[1], d[2]));
        fc3 = register_module("fc3", torch::nn::Linear(d[2], d[3]));
        fc4 = register_module("fc4", torch::nn::Linear(d[3], d[4]));
        fc5 = register_module("fc5", torch::nn::Linear(d[4], d[5]));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x) {
        // Use one of many tensor manipulation functions.
        x = torch::sigmoid(fc1->forward(x.reshape({ x.size(0), d[0] })));
        x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
        x = torch::sigmoid(fc2->forward(x));
        x = torch::sigmoid(fc3->forward(x));
        x = torch::sigmoid(fc4->forward(x));
        x = torch::sigmoid(fc5->forward(x));
        return x;
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr }, fc4{ nullptr }, fc5{ nullptr };
};

struct Generator : torch::nn::Module {
    Generator() {
        // Construct and register two Linear submodules.
        fc1 = register_module("fc1", torch::nn::Linear(g[0], g[1]));
        fc2 = register_module("fc2", torch::nn::Linear(g[1], g[2]));
        fc3 = register_module("fc3", torch::nn::Linear(g[2], g[3]));
        fc4 = register_module("fc4", torch::nn::Linear(g[3], g[4]));
        fc5 = register_module("fc5", torch::nn::Linear(g[4], g[5]));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x) {
        // Use one of many tensor manipulation functions.
        x = torch::relu(fc1->forward(x.reshape({ x.size(0), g[0] })));
        //x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
        x = torch::relu(fc2->forward(x));
        x = torch::relu(fc3->forward(x));
        x = torch::relu(fc4->forward(x));
        x = torch::tanh(fc5->forward(x));
        return x;
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr }, fc4{ nullptr }, fc5{ nullptr };
};

int main() {
    torch::manual_seed(1);


    torch::Device device(torch::kCPU);
    auto discriminator = std::make_shared<Discriminator>();
    auto generator = std::make_shared<Generator>();


    // Create a multi-threaded data loader for the MNIST dataset.
    //auto data_loader = torch::data::make_data_loader(torch::data::datasets::MNIST("D:\\Foton\\xz").map(torch::data::transforms::Stack<>()), Batch);

    auto datasetR = FaceDatasetR().map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(datasetR, torch::data::DataLoaderOptions().batch_size(Batch).enforce_ordering(false));

    // Instantiate an SGD optimization algorithm to update our Net's parameters.
    //torch::optim::SGD optimizer(dis->parameters(), /*lr=*/0.001);
    ///torch::optim::SGD Goptimizer(gen->parameters(), /*lr=*/0.001);

    torch::optim::Adam generator_optimizer(generator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));
    torch::optim::Adam discriminator_optimizer(discriminator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));

    for (size_t epoch = 1; epoch <= 10000; ++epoch) {
        size_t batch_index = 0;
        // Iterate the data loader to yield batches from the dataset.
        for (auto& batch : *data_loader) {
            discriminator->zero_grad();
            torch::Tensor real_images = batch.data.to(device);
            torch::Tensor real_labels = torch::empty(Batch, device).uniform_(0.8, 1.0);            //uniform нормализует дикий рандом, судя по всему это тензор прав результатов для настоящих изобр


            //std::cout << real_labels << std::endl;
            torch::Tensor real_output = discriminator->forward(real_images);
            torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
            d_loss_real.backward();
            torch::Tensor noise = torch::randn({ Batch, g[0], 1, 1 }, device);
            torch::Tensor fake_images = generator->forward(noise);
            torch::Tensor fake_labels = torch::zeros(Batch, device);             //нулевые значения для генератора
            torch::Tensor fake_output = discriminator->forward(fake_images.detach());

            Out(fake_images, batch_index, image_size);
            torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
            d_loss_fake.backward();

            torch::Tensor d_loss = d_loss_real + d_loss_fake;
            discriminator_optimizer.step();

            // Train generator.
            generator->zero_grad();
            fake_labels.fill_(1);
            fake_output = discriminator->forward(fake_images);
            torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
            g_loss.backward();
            generator_optimizer.step();
        }
    }
}
