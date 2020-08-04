#pragma once

#include <string>
#include <vector>

namespace ctranslate2 {

  class TranslationResult {
  public:
    TranslationResult(const std::vector<std::vector<std::string>>& hypotheses,
                      const std::vector<float>& scores,
                      const std::vector<std::vector<std::vector<float>>>* attention);
    TranslationResult(const std::vector<std::vector<std::string>>& hypotheses,
                        const std::vector<float>& scores,
                        const std::vector<int> & n_tokens,
                        const std::vector<std::vector<std::vector<float>>>* attention);


      const std::vector<std::string>& output() const;
    float score() const;

    size_t num_hypotheses() const;
    const std::vector<std::vector<std::string>>& hypotheses() const;
    const std::vector<float>& scores() const;

    int n_token() const;
    const std::vector<int>& n_tokens() const;

    const std::vector<std::vector<std::vector<float>>>& attention() const;
    bool has_attention() const;

  private:
    std::vector<std::vector<std::string>> _hypotheses;
    std::vector<float> _scores;
    std::vector<std::vector<std::vector<float>>> _attention;
    std::vector<int> _n_tokens;
  };

}
