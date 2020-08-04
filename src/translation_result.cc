#include "ctranslate2/translation_result.h"

namespace ctranslate2 {

  TranslationResult::TranslationResult(const std::vector<std::vector<std::string>>& hypotheses,
                                       const std::vector<float>& scores,
                                       const std::vector<std::vector<std::vector<float>>>* attention)
    : _hypotheses(hypotheses)
    , _scores(scores) {
    if (attention)
      _attention = *attention;
  }

  TranslationResult::TranslationResult(const std::vector<std::vector<std::string>>& hypotheses,
                                         const std::vector<float>& scores,
                                         const std::vector<int> & n_tokens,
                                         const std::vector<std::vector<std::vector<float>>>* attention)
            : _hypotheses(hypotheses)
            , _scores(scores)
            , _n_tokens(n_tokens)
            {
        if (attention)
            _attention = *attention;
  }

    const std::vector<std::string>& TranslationResult::output() const {
    return _hypotheses[0];
  }

  float TranslationResult::score() const {
    return _scores[0];
  }

  int TranslationResult::n_token() const {
      return _n_tokens[0];
  }

  size_t TranslationResult::num_hypotheses() const {
    return _hypotheses.size();
  }

  const std::vector<std::vector<std::string>>& TranslationResult::hypotheses() const {
    return _hypotheses;
  }

  const std::vector<float>& TranslationResult::scores() const {
    return _scores;
  }

  const std::vector<int>& TranslationResult::n_tokens() const {
      return _n_tokens;
  }

  const std::vector<std::vector<std::vector<float>>>& TranslationResult::attention() const {
    return _attention;
  }

  bool TranslationResult::has_attention() const {
    return !_attention.empty();
  }

}
