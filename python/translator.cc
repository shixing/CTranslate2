#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ctranslate2/translator_pool.h>

namespace py = pybind11;

#if PY_MAJOR_VERSION < 3
#  define STR_TYPE py::bytes
#else
#  define STR_TYPE py::str
#endif

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

ctranslate2::StorageView* buffer_to_storage_view(const std::vector<std::vector<int>> & buffer, ctranslate2::Device device) {
    ctranslate2::dim_t n_row = buffer.size();
    ctranslate2::dim_t n_col = 0;
    if (n_row > 0){
        n_col = buffer.front().size();
    }
    ctranslate2::Shape shape = {n_row, n_col};
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








template <typename T>
py::list std_vector_to_py_list(const std::vector<T>& v) {
  py::list l;
  for (const auto& x : v)
    l.append(x);
  return l;
}

template<>
py::list std_vector_to_py_list(const std::vector<std::string>& v) {
  py::list l;
  for (const auto& x : v)
    l.append(STR_TYPE(x));
  return l;
}

template <class T>
std::vector<T> py_list_to_std_vector(const py::object& l) {
  std::vector<T> v;
  v.reserve(py::len(l));
  for (const auto& s : l)
    v.emplace_back(s.cast<T>());
  return v;
}

template <class T>
static std::vector<std::vector<T>> batch_to_vector(const py::object& l) {
  std::vector<std::vector<T>> v;
  if (l.is(py::none()))
    return v;
  v.reserve(py::len(l));
  for (const auto& handle : l)
    v.emplace_back(py_list_to_std_vector<T>(py::cast<py::object>(handle)));
  return v;
}





class TranslatorWrapper
{
public:
  TranslatorWrapper(const std::string& model_path,
                    const std::string& device,
                    int device_index,
                    const std::string& compute_type,
                    size_t inter_threads,
                    size_t intra_threads)
    : _translator_pool(inter_threads,
                       intra_threads,
                       ctranslate2::models::Model::load(model_path,
                                                        device,
                                                        device_index,
                                                        compute_type)) {
    _device = ctranslate2::str_to_device(device);
  }

  py::tuple translate_file(const std::string& in_file,
                           const std::string& out_file,
                           size_t max_batch_size,
                           size_t beam_size,
                           size_t num_hypotheses,
                           float length_penalty,
                           size_t max_decoding_length,
                           size_t min_decoding_length,
                           bool use_vmap,
                           bool with_scores,
                           size_t sampling_topk,
                           float sampling_temperature) {
    auto options = ctranslate2::TranslationOptions();
    options.beam_size = beam_size;
    options.length_penalty = length_penalty;
    options.sampling_topk = sampling_topk;
    options.sampling_temperature = sampling_temperature;
    options.max_decoding_length = max_decoding_length;
    options.min_decoding_length = min_decoding_length;
    options.num_hypotheses = num_hypotheses;
    options.use_vmap = use_vmap;

    size_t num_tokens = 0;
    {
      py::gil_scoped_release release;
      num_tokens = _translator_pool.consume_text_file(in_file,
                                                      out_file,
                                                      max_batch_size,
                                                      options,
                                                      with_scores);
    }

    return py::make_tuple(num_tokens);
  }

  py::list translate_batch(const py::object& source,
                           const py::object& target_prefix,
                           size_t beam_size,
                           size_t num_hypotheses,
                           float length_penalty,
                           size_t max_decoding_length,
                           size_t min_decoding_length,
                           bool use_vmap,
                           bool return_attention,
                           size_t sampling_topk,
                           float sampling_temperature) {
    if (source.is(py::none()) || py::len(source) == 0)
      return py::list();

    auto options = ctranslate2::TranslationOptions();
    options.beam_size = beam_size;
    options.length_penalty = length_penalty;
    options.sampling_topk = sampling_topk;
    options.sampling_temperature = sampling_temperature;
    options.max_decoding_length = max_decoding_length;
    options.min_decoding_length = min_decoding_length;
    options.num_hypotheses = num_hypotheses;
    options.use_vmap = use_vmap;
    options.return_attention = return_attention;

    std::vector<ctranslate2::TranslationResult> results;
    auto future = _translator_pool.post(batch_to_vector<std::string>(source),
                                        batch_to_vector<std::string>(target_prefix),
                                        options);

    {
      py::gil_scoped_release release;
      results = future.get();
    }

    py::list py_results;
    for (const auto& result : results) {
      py::list batch;
      for (size_t i = 0; i < result.num_hypotheses(); ++i) {
        py::dict hyp;
        hyp["score"] = result.scores()[i];
        hyp["tokens"] = std_vector_to_py_list(result.hypotheses()[i]);
        if (result.has_attention()) {
          py::list attn;
          for (const auto& attn_vector : result.attention()[i])
            attn.append(std_vector_to_py_list(attn_vector));
          hyp["attention"] = attn;
        }
        batch.append(hyp);
      }
      py_results.append(batch);
    }

    return py_results;
  }

    py::list translate_batch_with_fsa_prefix(const py::object& source,
                                             const py::object& emission_matrix,
                                             const py::object& transition_matrix,
                                             const py::object& length_matrix,
                                             int init_state,
                                             const py::object& target_prefix,size_t beam_size,
                             size_t num_hypotheses,
                             float length_penalty,
                             size_t max_decoding_length,
                             size_t min_decoding_length,
                             bool use_vmap,
                             bool return_attention,
                             size_t sampling_topk,
                             float sampling_temperature
                             ) {
        if (source.is(py::none()) || py::len(source) == 0)
            return py::list();

        auto options = ctranslate2::TranslationOptions();
        options.beam_size = beam_size;
        options.length_penalty = length_penalty;
        options.sampling_topk = sampling_topk;
        options.sampling_temperature = sampling_temperature;
        options.max_decoding_length = max_decoding_length;
        options.min_decoding_length = min_decoding_length;
        options.num_hypotheses = num_hypotheses;
        options.use_vmap = use_vmap;
        options.return_attention = return_attention;

        options.decode_with_fsa_prefix = true;

        options.emission_matrix = std::shared_ptr<ctranslate2::StorageView>(buffer_to_storage_view(batch_to_vector<int>(emission_matrix), _device));
        options.transition_matrix = std::shared_ptr<ctranslate2::StorageView>(buffer_to_storage_view(batch_to_vector<int>(transition_matrix), ctranslate2::Device::CPU));
        options.length_matrix = std::shared_ptr<ctranslate2::StorageView>(buffer_to_storage_view(batch_to_vector<int>(length_matrix), ctranslate2::Device::CPU));
        options.init_state = init_state;

        std::vector<ctranslate2::TranslationResult> results;
        auto future = _translator_pool.post(batch_to_vector<std::string>(source),
                                            batch_to_vector<std::string>(target_prefix),
                                            options);

        {
            py::gil_scoped_release release;
            results = future.get();
        }

        py::list py_results;
        for (const auto& result : results) {
            py::list batch;
            for (size_t i = 0; i < result.num_hypotheses(); ++i) {
                py::dict hyp;
                hyp["score"] = result.scores()[i];
                hyp["tokens"] = std_vector_to_py_list(result.hypotheses()[i]);
                if (result.has_attention()) {
                    py::list attn;
                    for (const auto& attn_vector : result.attention()[i])
                        attn.append(std_vector_to_py_list(attn_vector));
                    hyp["attention"] = attn;
                }
                batch.append(hyp);
            }
            py_results.append(batch);
        }

        return py_results;
    }


private:
  ctranslate2::TranslatorPool _translator_pool;
  ctranslate2::Device _device;
};

PYBIND11_MODULE(translator, m)
{
  m.def("contains_model", &ctranslate2::models::contains_model, py::arg("path"));

  py::class_<TranslatorWrapper>(m, "Translator")
    .def(py::init<std::string, std::string, int, std::string, size_t, size_t>(),
         py::arg("model_path"),
         py::arg("device")="cpu",
         py::arg("device_index")=0,
         py::arg("compute_type")="default",
         py::arg("inter_threads")=1,
         py::arg("intra_threads")=4)
    .def("translate_batch", &TranslatorWrapper::translate_batch,
         py::arg("source"),
         py::arg("target_prefix")=py::none(),
         py::arg("beam_size")=4,
         py::arg("num_hypotheses")=1,
         py::arg("length_penalty")=0,
         py::arg("max_decoding_length")=250,
         py::arg("min_decoding_length")=1,
         py::arg("use_vmap")=false,
         py::arg("return_attention")=false,
         py::arg("sampling_topk")=1,
         py::arg("sampling_temperature")=1)
    .def("translate_batch_with_fsa_prefix", &TranslatorWrapper::translate_batch_with_fsa_prefix,
        py::arg("source"),
        py::arg("transition_matrix"),
        py::arg("emission_matrix"),
        py::arg("length_matrix"),
        py::arg("init_state"),
        py::arg("target_prefix")=py::none(),
        py::arg("beam_size")=4,
        py::arg("num_hypotheses")=1,
        py::arg("length_penalty")=0,
        py::arg("max_decoding_length")=250,
        py::arg("min_decoding_length")=1,
        py::arg("use_vmap")=false,
        py::arg("return_attention")=false,
        py::arg("sampling_topk")=1,
        py::arg("sampling_temperature")=1)
    .def("translate_file", &TranslatorWrapper::translate_file,
         py::arg("input_path"),
         py::arg("output_path"),
         py::arg("max_batch_size"),
         py::arg("beam_size")=4,
         py::arg("num_hypotheses")=1,
         py::arg("length_penalty")=0,
         py::arg("max_decoding_length")=250,
         py::arg("min_decoding_length")=1,
         py::arg("use_vmap")=false,
         py::arg("with_scores")=false,
         py::arg("sampling_topk")=1,
         py::arg("sampling_temperature")=1)
    ;
}
