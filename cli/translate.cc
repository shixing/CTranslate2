#include <chrono>
#include <fstream>
#include <iostream>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>



#include <ctranslate2/translator_pool.h>
#include <ctranslate2/utils.h>
#include <ctranslate2/devices.h>
#include <ctranslate2/profiler.h>

namespace po = boost::program_options;

template <class T>
void print_buffer(const std::vector<std::vector<T>> & buffer){
    std::cout<<"Matrix:\n";
    for (const auto & row : buffer){
        for (const auto & n: row) {
            std::cout << n << "|";
        }
        std::cout<<std::endl;
    }
}

ctranslate2::StorageView* buffer_to_storage_view(const std::vector<std::vector<int>> & buffer, ctranslate2::Device device, bool is_scalar = false) {
    ctranslate2::dim_t n_row = buffer.size();
    ctranslate2::dim_t n_col = 0;
    if (n_row > 0){
        n_col = buffer.front().size();
    }
    ctranslate2::Shape shape = {n_row, n_col};
    if (is_scalar && n_row * n_col == 1){
        shape = {};
    }
    ctranslate2::StorageView* sv = new ctranslate2::StorageView(shape, 0, device);
    int size = n_row * n_col;
    if (size > 0) {
        int* data = new int[n_row * n_col];
        int i = 0;
        for (const auto & v: buffer){
            for (const auto & value: v){
                data[i] = value;
                i += 1;
            }
        }
        sv->copy_from(data,size,ctranslate2::Device::CPU);
        delete data;
    }
    return sv;
}

int main(int argc, char *argv[]) {
    po::options_description desc("CTranslate2 translation client");
    desc.add_options()
            ("help", "Display available options.")
            ("model", po::value<std::string>(),
             "Path to the CTranslate2 model directory.")
            ("compute_type", po::value<std::string>()->default_value("default"),
             "Force the model type as \"float\", \"int16\" or \"int8\"")
            ("src", po::value<std::string>(),
             "Path to the file to translate (read from the standard input if not set).")
            ("tgt", po::value<std::string>(),
             "Path to the output file (write to the standard output if not set.")
            ("use_vmap", po::bool_switch()->default_value(false),
             "Use the vocabulary map included in the model to restrict the target candidates.")
            ("batch_size", po::value<size_t>()->default_value(30),
             "Number of sentences to forward into the model at once.")
            ("beam_size", po::value<size_t>()->default_value(5),
             "Beam search size (set 1 for greedy decoding).")
            ("sampling_topk", po::value<size_t>()->default_value(1),
             "Sample randomly from the top K candidates.")
            ("sampling_temperature", po::value<float>()->default_value(1),
             "Sampling temperature.")
            ("n_best", po::value<size_t>()->default_value(1),
             "Also output the n-best hypotheses.")
            ("with_score", po::bool_switch()->default_value(false),
             "Also output translation scores.")
            ("length_penalty", po::value<float>()->default_value(0),
             "Length penalty to apply during beam search")
            ("max_sent_length", po::value<size_t>()->default_value(250),
             "Maximum sentence length to produce.")
            ("min_sent_length", po::value<size_t>()->default_value(1),
             "Minimum sentence length to produce.")
            ("log_throughput", po::bool_switch()->default_value(false),
             "Log average tokens per second at the end of the translation.")
            ("log_profiling", po::bool_switch()->default_value(false),
             "Log execution profiling.")
            ("inter_threads", po::value<size_t>()->default_value(1),
             "Maximum number of translations to run in parallel.")
            ("intra_threads", po::value<size_t>()->default_value(0),
             "Number of OpenMP threads (set to 0 to use the default value).")
            ("device", po::value<std::string>()->default_value("cpu"),
             "Device to use (can be cpu, cuda, auto).")
            ("device_index", po::value<int>()->default_value(0),
             "Index of the device to use.")
            // for the prefix decoding
            ("prefix_info", po::value<std::string>(), "Path to the matrix file");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cerr << desc << std::endl;
        return 1;
    }
    if (!vm.count("model")) {
        std::cerr << "missing model" << std::endl;
        return 1;
    }

    size_t inter_threads = vm["inter_threads"].as<size_t>();
    size_t intra_threads = vm["intra_threads"].as<size_t>();

    auto model = ctranslate2::models::Model::load(
            vm["model"].as<std::string>(),
            vm["device"].as<std::string>(),
            vm["device_index"].as<int>(),
            vm["compute_type"].as<std::string>());

    ctranslate2::TranslatorPool translator_pool(inter_threads, intra_threads, model);

    auto options = ctranslate2::TranslationOptions();
    options.beam_size = vm["beam_size"].as<size_t>();
    options.length_penalty = vm["length_penalty"].as<float>();
    options.sampling_topk = vm["sampling_topk"].as<size_t>();
    options.sampling_temperature = vm["sampling_temperature"].as<float>();
    options.max_decoding_length = vm["max_sent_length"].as<size_t>();
    options.min_decoding_length = vm["min_sent_length"].as<size_t>();
    options.num_hypotheses = vm["n_best"].as<size_t>();
    options.use_vmap = vm["use_vmap"].as<bool>();

    std::istream *in = &std::cin;
    std::ostream *out = &std::cout;

    if (vm.count("src")) {
        auto path = vm["src"].as<std::string>();
        auto src_file = new std::ifstream(path);
        if (!src_file->is_open())
            throw std::runtime_error("Unable to open input file " + path);
        in = src_file;
    }
    if (vm.count("tgt")) {
        out = new std::ofstream(vm["tgt"].as<std::string>());
    }

    if (vm.count("prefix_info")) {
        auto path = vm["prefix_info"].as<std::string>();
        auto prefix_info_in = std::ifstream(path);
        if (!prefix_info_in.is_open())
            throw std::runtime_error("Unable to open input file " + path);
        std::string line;

        std::vector<std::vector<std::vector<int>>> buffers;
        std::vector<std::vector<int>> buffer;
        while (std::getline(prefix_info_in, line)) {
            if (line == ""){
                buffers.push_back(buffer);
                buffer.clear();
            } else {
                std::vector<std::string> str_row;
                boost::split(str_row, line, boost::is_any_of(" "), boost::token_compress_on);

                auto int_row = std::vector<int>();
                std::transform(str_row.begin(), str_row.end(), std::back_inserter(int_row),
                        [](const std::string& s){return std::stoi(s);});

                buffer.emplace_back(int_row);
            }
        }
        prefix_info_in.close();

        assert(buffers.size() == 4);
        options.decode_with_fsa_prefix = true;
        auto device = model->device();
        options.emission_matrix = std::shared_ptr<ctranslate2::StorageView>(buffer_to_storage_view(buffers[0], device));
        options.transition_matrix = std::shared_ptr<ctranslate2::StorageView>(buffer_to_storage_view(buffers[1], ctranslate2::Device::CPU));
        options.length_matrix = std::shared_ptr<ctranslate2::StorageView>(buffer_to_storage_view(buffers[2], ctranslate2::Device::CPU));
        options.init_state = buffers[3][0][0];
        /*
        std::cout << "options.decode_with_fsa_prefix " << options.decode_with_fsa_prefix << std::endl;
        std::cout << "options.emission_matrix " << *options.emission_matrix << std::endl;
        std::cout << "options.transition_matrix " << *options.transition_matrix << std::endl;
        std::cout << "options.length_matrix " << *options.length_matrix << std::endl;
         */

    }

    auto log_profiling = vm["log_profiling"].as<bool>();
    auto t1 = std::chrono::high_resolution_clock::now();
    if (log_profiling)
        ctranslate2::init_profiling(model->device(), inter_threads);
    auto num_tokens = translator_pool.consume_text_file(*in,
                                                        *out,
                                                        vm["batch_size"].as<size_t>(),
                                                        options,
                                                        vm["with_score"].as<bool>());
    if (log_profiling)
        ctranslate2::dump_profiling(std::cerr);
    auto t2 = std::chrono::high_resolution_clock::now();

    if (in != &std::cin)
        delete in;
    if (out != &std::cout)
        delete out;

    if (vm["log_throughput"].as<bool>()) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        std::cerr << static_cast<double>(num_tokens) / static_cast<double>(duration / 1000) << std::endl;
    }

    return 0;
}
