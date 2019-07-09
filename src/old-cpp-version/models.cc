#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <queue>

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/rnn.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"

#include <data.h>
#include <eval.h>
#include <models.h>

using namespace std;
using namespace dynet;

unsigned LAYERS_FF = 1;
unsigned LAYERS_FF_PAIR = 1;
unsigned LAYERS_FF_REFINE = 1;
unsigned LAYERS_LSTM = 1;
unsigned DIM_INPUT = 128;
unsigned DIM_LSTM_HIDDEN = 64;
unsigned DIM_FF_HIDDEN = 64;
unsigned DIM_FF_HIDDEN_PAIR = 64;
unsigned DIM_FF_HIDDEN_REFINE = 64;
float DROPOUT_INPUT = 0.0;
float DROPOUT_TEXTIN = 0.0;
float DROPOUT_CONTEXTIN = 0.0;
float DROPOUT_FF = 0.0;
float DROPOUT_FF_PAIR = 0.0;
float DROPOUT_FF_REFINE = 0.0;
float DROPOUT_LSTM_H = 0.0;
float DROPOUT_LSTM_C = 0.0;
float DROPOUT_LSTM_I = 0.0;
float SELECTION_PROPORTION = 0.1;
int FIXED_SELECT = 0;
unsigned int FIXED_SELECT_DIST = 25;
int INPUT_HAND_CRAFTED = 0;
int INPUT_STRUCTURE = 0;
int INPUT_TEXT = 0;
int MAX_LINK_LENGTH = 100;
int REFINE_PREDICTION = 0;
int SUBTRACT_AV_SENT = 0;
int CONTEXT_SIZE = 0;
int CONTEXT_TEXT = 0;

int LOSS_CLUSTER = 0;
int LOSS_NO_NORMALIZE = 0;
int LOSS_MAX_HINGE = 0;
int LOSS_USE_MULTIPLIER = 0;
int LOSS_VARY_MARGIN = 0;
float LOSS_RIGHT_CLUSTER = 1.0;
float LOSS_WRONG_CLUSTER = 1.0;
float LOSS_EXTRA_NONE = 1.0;
float LOSS_MISSED_NONE = 1.0;

LossType loss_type = kCrossEntropy;
ostream& operator<<(ostream& os, LossType c) {
  switch (c) {
    case kHinge: os << "kHinge"; break;
    case kCrossEntropy: os << "kCrossEntropy"; break;
    default: os.setstate(std::ios_base::failbit);
  }
  return os;
}
void set_loss_type(char* option) {
  if (strcmp(option, "hinge") ==  0 ||
      strcmp(option, "kHinge") == 0) {
    loss_type = kHinge;
  } else {
    loss_type = kCrossEntropy;
  }
}

NonLinearityType nonlinearity_type = kSeLU;
NonLinearityType nonlinearity_type_pair = kSeLU;
NonLinearityType nonlinearity_type_refine = kSeLU;
ostream& operator<<(ostream& os, NonLinearityType c) {
  switch (c) {
    case kLogistic: os << "kLogistic"; break;
    case kTanh: os << "kTanh"; break;
    case kCube: os << "kCube"; break;
    case kRectify: os << "kRectify"; break;
    case kELU: os << "kELU"; break;
    case kSeLU: os << "kSeLU"; break;
    case kSiLU: os << "kSiLU"; break;
    case kSoftSign: os << "kSoftSign"; break;
    default: os.setstate(std::ios_base::failbit);
  }
  return os;
}
void set_nonlinearity_type(char* option, NonLinearityType* value) {
  if (strcmp(option, "logistic") ==  0 || strcmp(option, "kLogistic") == 0) {
    *value = kLogistic;
  } else if (strcmp(option, "tanh") ==  0 || strcmp(option, "kTanh") == 0) {
    *value = kTanh;
  } else if (strcmp(option, "cube") ==  0 || strcmp(option, "kCube") == 0) {
    *value = kCube;
  } else if (strcmp(option, "elu") ==  0 || strcmp(option, "kELU") == 0) {
    *value = kELU;
  } else if (strcmp(option, "selu") ==  0 || strcmp(option, "kSELU") == 0) {
    *value = kSeLU;
  } else if (strcmp(option, "silu") ==  0 || strcmp(option, "kSiLU") == 0) {
    *value = kSiLU;
  } else if (strcmp(option, "softsign") ==  0 || strcmp(option, "kSoftSign") == 0) {
    *value = kSoftSign;
  } else {
    *value = kRectify;
  }
}

MessageRepresentation message_representation = kAvWord;
ostream& operator<<(ostream& os, MessageRepresentation c) {
  switch (c) {
    case kAvWord: os << "kAvWord"; break;
    case kLSTM: os << "kLSTM"; break;
    case kMaxPool: os << "kMaxPool"; break;
    default: os.setstate(std::ios_base::failbit);
  }
  return os;
}
void set_message_representation(char* option) {
  if (strcmp(option, "lstm") ==  0 || strcmp(option, "kLSTM") == 0) {
    message_representation = kLSTM;
  } else if (
    strcmp(option, "maxpool") ==  0 || strcmp(option, "kMaxPool") == 0
  ) {
    cerr << "kMaxPool is not implemented yet\n";
    assert(false);
    message_representation = kMaxPool;
  } else {
    message_representation = kAvWord;
  }
}

Expression LinkingModel::getLoss(
  ComputationGraph& cg, vector<Expression>& output, bool eval, bool linked
) {
  float margin = eval ? 0.0 : 1.0;
  unsigned int gold = linked ? 1 : -1;
  Expression zero = input(cg, 0.0);
  return max(zero, margin - gold * output.back()); // binary hinge loss
}
Expression LinkingModel::getLoss(
  ComputationGraph& cg, unsigned int source, vector<Expression>& output,
  bool eval, const unordered_set<unsigned int>& gold,
  const unordered_set<unsigned int>* cluster
) {
  unsigned int start = 0;
  if (source > MAX_LINK_LENGTH) start = source - MAX_LINK_LENGTH;
  Expression e_concat_o = log_softmax(concatenate(output));
  vector<float> data; // dense
  int nvalues = e_concat_o.dim().rows();
  int id_for_hinge = -1;
  for (unsigned int i = 0; i < nvalues; i++) {
    float val = 0.0;
    if (LOSS_CLUSTER) {
      if (cluster->find(start + i) != cluster->end()) {
        id_for_hinge = i;
        val = 1.0f;
        if (! LOSS_NO_NORMALIZE) {
          val = 1.0f / static_cast<float>(cluster->size());
        }
      }
    } else {
      if (gold.find(start + i) != gold.end())  {
        id_for_hinge = i;
        val = 1.0f;
        if (! LOSS_NO_NORMALIZE) {
          val = 1.0f / static_cast<float>(gold.size());
        }
      }
    }
    data.push_back(val);
  }

  bool is_correct = false;
  bool same_cluster = false;
  bool is_none = false;
  bool gold_is_none = false;
  vector<unsigned int> targets;
  getPredictionsSelection(cg, source, output, targets);
  for (unsigned int pos : targets) {
    if (pos == nvalues - 1) is_none = true;
    for (int gpos : gold) {
      if (gpos == nvalues - 1) {
        gold_is_none = true;
      }
      if (gpos == pos) {
        is_correct = true;
        if (gpos != nvalues - 1) same_cluster = true;
      }
    }
    for (unsigned int gpos : *cluster) {
      if (gpos == pos && gpos != nvalues - 1)
        same_cluster = true;
    }
  }
  auto scores = as_vector(cg.incremental_forward(concatenate(output)));
  float best = -1e20;
  if (LOSS_MAX_HINGE) {
    if (LOSS_CLUSTER) {
      for (unsigned int gpos : *cluster) {
        if (
            gpos >= start && 
            gpos < nvalues + start &&
            best < scores[gpos - start]
        ) {
          best = scores[gpos - start];
          id_for_hinge = gpos - start;
        }
      }
    } else {
      for (int gpos : gold) {
        if (
            gpos >= start && 
            gpos < nvalues + start &&
            best < scores[gpos - start]
        ) {
          best = scores[gpos - start];
          id_for_hinge = gpos - start;
        }
      }
    }
  }
  // Options:
  // is_correct same_cluster is_none gold_is_none
  // F          F            F       F             linked to other thread
  // F          T            F       F             same thread, wrong line
  // F          F            T       F             none, should be something
  // F          F            F       T             something, should be none
  // T          F            T       T             correct none
  // T          T            F       F             correct link

  double multiplier = 1.0;
  if (LOSS_USE_MULTIPLIER) {
    if (is_correct) {
    } else if (same_cluster) {
      // Wrong link, right cluster
      multiplier = LOSS_RIGHT_CLUSTER;
    } else if (is_none) {
      // Linked to none, should be something
      multiplier = LOSS_EXTRA_NONE;
    } else if (gold_is_none) {
      // Linked to something, should be nothing
      multiplier = LOSS_MISSED_NONE;
    } else {
      // Linked to other thread
      multiplier = LOSS_WRONG_CLUSTER;
    }
  }

  if (loss_type == kHinge) {
    // For hinge loss, just try to get the most recent choice
    if (id_for_hinge < 0) {
      // The cutoff means nothing correct is available
      return input(cg, 0.0);
    } else if (LOSS_VARY_MARGIN) {
      float margin = eval ? 0.0 : multiplier;
      return hinge(e_concat_o, id_for_hinge, margin);
    } else {
      float margin = eval ? 0.0 : 1.0;
      return multiplier * hinge(e_concat_o, id_for_hinge, margin);
    }
  } else {
    // For categorical cross-entropy loss, consider all options
    unsigned int size = data.size();
    Expression gold_expr = input(cg, {1, size}, data);
    Expression loss = - (gold_expr * e_concat_o) * multiplier;
    return loss;
  }
}
Expression LinkingModel::getLossSet(
  ComputationGraph& cg, vector<Expression>& output, bool eval, IRCLog* log,
  unsigned int min_source, unsigned int max_source
) {
  unsigned int output_pos = 0;
  vector<Expression> losses;
  for (unsigned int src = min_source; src < max_source; src++) {
    vector<Expression> local_output;
    unsigned int start = 0;
    if (src > MAX_LINK_LENGTH) start = src - MAX_LINK_LENGTH;
    for (unsigned int target = start; target <= src; target++) {
      local_output.push_back(output[output_pos]);
      output_pos++;
    }
    Expression local_loss = getLoss(cg, src, local_output, eval,
        log->links->at(src), log->clusters->at(src));
    losses.push_back(local_loss);
  }

  return sum(losses);
}

struct ComparePair {
  bool operator()(
    const pair<float, unsigned int>& left,
    const pair<float, unsigned int>& right
  ) const {
    // Note, this is correct - a priority queue that keeps the min value at
    // the top requires a comparator that returns true when left > right.
    return left.first > right.first;
  }
};
bool LinkingModel::getPredictionsPair(
  ComputationGraph& cg, vector<Expression>& output
) {
  float decision = as_scalar(cg.incremental_forward(output.back()));
  return decision > 0.0;
}
void LinkingModel::getPredictionsSelection(
  ComputationGraph& cg, unsigned int source, vector<Expression>& output,
  vector<unsigned int>& targets
) {
  unsigned int start = 0;
  if (source > MAX_LINK_LENGTH) start = source - MAX_LINK_LENGTH;
  // Use a min-heap size 10 to efficiently find the top 10 options
  // in O(n log 10) time. We use a min-heap so we can check if this item is
  // bigger than anything in the heap in constant time.
  priority_queue<pair<float, unsigned int>, vector<pair<float, unsigned int>>, ComparePair> queue;
  auto scores = as_vector(cg.incremental_forward(softmax(concatenate(output))));
  for (unsigned int target = 0; target < scores.size(); target++) {
    pair<float, unsigned int> score(scores[target], target + start);
    if (queue.size() < 10) {
      queue.push(score);
    } else if (queue.top().first < score.first) {
      queue.pop();
      queue.push(score);
    }
  }

  // It's a min-heap, so we need to get everything out and reverse it.
  // TODO: How often does this select "link to nothing else" + something else?
  vector<pair<float, unsigned int>> top_ten;
  while (! queue.empty()) {
    top_ten.push_back(queue.top());
    queue.pop();
  }
  assert(top_ten.size() <= 10);
  float total = 0.0;
  for (
    auto iter = top_ten.rbegin();
    iter != top_ten.rend() && total < SELECTION_PROPORTION;
    iter++
  ) {
    total += iter->first;
    unsigned int pos = iter->second;
    targets.push_back(pos);
  }
}
void LinkingModel::getPredictionsSet(
  ComputationGraph& cg, vector<Expression>& output,
  unsigned int min_source, unsigned int max_source,
  unordered_map<unsigned int, unordered_set<unsigned int>>* links
) {
  unsigned int output_pos = 0;
  for (unsigned int source = min_source; source < max_source; source++) {
    vector<unsigned int> targets;
    vector<Expression> local_output;
    unsigned int start = 0;
    if (source > MAX_LINK_LENGTH) start = source - MAX_LINK_LENGTH;
    for (unsigned int target = start; target <= source; target++) {
      local_output.push_back(output[output_pos]);
      output_pos++;
    }
    getPredictionsSelection(cg, source, local_output, targets);
    for (auto target : targets) {
      auto iter = links->insert({source, unordered_set<unsigned int>()});
      iter.first->second.insert(target);
    }
  }
}

void LinkingModel::updateEval(
  ComputationGraph& cg, vector<Expression>& output, bool linked,
  Evaluator& score
) {
  if (linked) score.total_gold += 1;
  if (getPredictionsPair(cg, output)) {
    score.total_guess += 1;
    if (linked) score.matched += 1;
  }
}
void LinkingModel::updateEval(
  ComputationGraph& cg, vector<Expression>& output,
  unsigned int source, const unordered_set<unsigned int>& gold,
  Evaluator& score
) {
  vector<unsigned int> targets;
  getPredictionsSelection(cg, source, output, targets);
  for (auto target : targets) {
    score.total_guess += 1;
    if (gold.find(target) != gold.end()) {
      score.matched += 1;
    }
  }
  score.total_gold += gold.size();
  if (gold.size() == 1 and gold.find(source) != gold.end()) {
    score.starts_total_gold += 1;
    if (targets.size() == 1 and targets[0] == source)
      score.starts_matched += 1;
  }
  if (targets.size() == 1 and targets[0] == source)
    score.starts_total_guess += 1;
}
void LinkingModel::updateEvalSet(
  ComputationGraph& cg, vector<Expression>& output, IRCLog* log,
  unsigned int min_source, unsigned int max_source, Evaluator& score
) {
  unordered_map<unsigned int, unordered_set<unsigned int>> links;
  getPredictionsSet(cg, output, min_source, max_source, &links);

  for (unsigned int source = min_source; source < max_source; source++) {
    auto gold = log->links->find(source);
    auto guess = links.find(source);

    if (gold != log->links->end() && guess != links.end()) {
      for (auto target : gold->second) {
        if (guess->second.find(target) != guess->second.end())
          score.matched += 1;
      }
    }
    if (gold != log->links->end()) {
      score.total_gold += gold->second.size();
      if (gold->second.size() == 1 &&
          gold->second.find(source) != gold->second.end()) {
        score.starts_total_gold += 1;
        if (guess->second.size() == 1 &&
            guess->second.find(source) != guess->second.end()) {
          score.starts_matched += 1;
        }
      }
    }
    if (guess != links.end()) {
      score.total_guess += guess->second.size();
      if (guess->second.size() == 1 &&
          guess->second.find(source) != guess->second.end()) {
        score.starts_total_guess += 1;
      }
    }
  }
}

void initialize_word_vectors(
  LookupParameter& p_w2v, unordered_map<int, vector<float>>& word_vectors
) {
  for (const auto& pair : word_vectors) {
    p_w2v.initialize(pair.first, pair.second);
  }
}


////// Linear //////

LinearModel::LinearModel(
  ParameterCollection& model, unordered_map<int, vector<float>>& word_vectors
) {
  unsigned int nfeatures = 0;
  if (INPUT_HAND_CRAFTED) nfeatures += N_FEATURES_BASE;
  if (INPUT_STRUCTURE) nfeatures += N_FEATURES_STRUCTURE;
  if (INPUT_TEXT) {
    nfeatures += DIM_INPUT;
    p_w2v = model.add_lookup_parameters(SIMPLE_VOCAB_SIZE, {DIM_INPUT});
    if (word_vectors.size() > 0)
      initialize_word_vectors(p_w2v, word_vectors);
  }
  p_i2o = model.add_parameters({1, nfeatures});
}
void LinearModel::printWeights() {
  unsigned int nfeatures = 0;
  if (INPUT_HAND_CRAFTED) nfeatures += N_FEATURES_BASE;
  if (INPUT_STRUCTURE) nfeatures += N_FEATURES_STRUCTURE;
  cout << "p_i2o " << p_i2o.dim() << " ";
  for (unsigned j = 0; j < nfeatures; j++) {
    cout << j << " " << p_i2o.get_storage().values.v[j] << endl;
  }
}

void LinearModel::preprocessPair(
  IRCLog* log, unsigned int source, unsigned int target
) {
  // no action at the moment
}
void LinearModel::preprocessSelection(IRCLog* log, unsigned int source) {
  // no action at the moment
}
void LinearModel::preprocessSet(
  IRCLog* log, unsigned int min_source, unsigned int max_source
) {
  // no action at the moment
}
Expression LinearModel::makeUtterance(
  ComputationGraph& cg, IRCLog* log, unsigned int sentence, bool eval
) {
  vector<Expression> e_words;
  auto msg = &(log->messages->at(sentence));
  for (auto& token : msg->simple_tokens) {
    e_words.push_back(lookup(cg, p_w2v, token));
  }
  return sum(e_words) / e_words.size();
}

Expression LinearModel::makePair(
  ComputationGraph& cg, IRCLog* log, unsigned int source, unsigned int target,
  unordered_map<unsigned int, unordered_set<unsigned int>>& links, bool eval
) {
  vector<Expression> e_inputs;
  vector<float> contextual_feats;
  auto features = log->get_features(source, target, links,
      contextual_feats);
  if (INPUT_HAND_CRAFTED)
    e_inputs.push_back(input(cg, {N_FEATURES_BASE}, *features));
  if (INPUT_STRUCTURE)
    e_inputs.push_back(input(cg, {N_FEATURES_STRUCTURE},
          contextual_feats));
  if (INPUT_TEXT)
    e_inputs.push_back(makeUtterance(cg, log, target, eval));

  Expression e_in = concatenate(e_inputs);

  // Apply dropout
  if (DROPOUT_INPUT > 0.0 && ! eval) {
    e_in = dropout(e_in, DROPOUT_INPUT);
  }

  return e_in;
}

void LinearModel::addOutputForPair(
  ComputationGraph& cg, IRCLog* log, unsigned int source, unsigned int target,
  Expression& e_i2o,
  unordered_map<unsigned int, unordered_set<unsigned int>>& links,
  vector<Expression>& options, bool eval
) {
  if (source - target > MAX_LINK_LENGTH) {
    Expression e_o = input(cg, {1}, {0.0});
    options.push_back(e_o);
  } else {
    Expression e_in = makePair(cg, log, source, target, links, eval);
    Expression e_o = e_i2o * e_in;
    options.push_back(e_o);
  }
}

void LinearModel::makeGraph(
  ComputationGraph& cg, vector<Expression>& output, IRCLog* log,
  unsigned int source, unsigned int target,
  unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
  bool eval
) {
  Expression e_i2o = parameter(cg, p_i2o);
  addOutputForPair(cg, log, source, target, e_i2o, current_links, output,
      eval);
  if (eval) {
    if (getPredictionsPair(cg, output))
      current_links[source].insert(target);
  }
}
void LinearModel::makeGraph(
  ComputationGraph& cg, vector<Expression>& output, IRCLog* log,
  unsigned int source,
  unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
  bool eval
) {
  unsigned int start = 0;
  if (source > MAX_LINK_LENGTH) start = source - MAX_LINK_LENGTH;
  Expression e_i2o = parameter(cg, p_i2o);
  for (unsigned int target = start; target <= source; target++) {
    addOutputForPair(cg, log, source, target, e_i2o, current_links, output,
        eval);
  }

  if (eval) {
    vector<unsigned int> targets;
    getPredictionsSelection(cg, source, output, targets);
    for (auto num : targets) {
      if (num != source) current_links[source].insert(num);
    }
  }
}
void LinearModel::makeGraphSet(
  ComputationGraph& cg, vector<Expression>& output, IRCLog* log,
  unsigned int min_source, unsigned int max_source,
  unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
  bool eval
) {
  Expression e_i2o = parameter(cg, p_i2o);
  for (unsigned int src = min_source; src < max_source; src++) {
    float best_score = 0.0;
    unsigned int best = 0;
    unsigned int start = 0;
    if (src > MAX_LINK_LENGTH) start = src - MAX_LINK_LENGTH;
    for (unsigned int target = start; target <= src; target++) {
      addOutputForPair(cg, log, src, target, e_i2o, current_links, output,
          eval);

      // Update the best score
      float score = as_scalar(cg.incremental_forward(output.back()));
      if (score > best_score || best == target) {
        best_score = score;
        best = target;
      }
    }

    // Add this link to our set
    if (eval) {
      current_links[src].insert(best);
    }
  }
}


////// Feedforward //////

FeedForwardModel::FeedForwardModel(
  ParameterCollection& model, unordered_map<int, vector<float>>& word_vectors
) :
  lstm_l2r(LAYERS_LSTM, DIM_INPUT, DIM_LSTM_HIDDEN, model),
  lstm_r2l(LAYERS_LSTM, DIM_INPUT, DIM_LSTM_HIDDEN, model)
{
  unsigned int n_input_feats = 0;
  if (INPUT_HAND_CRAFTED) n_input_feats += N_FEATURES_BASE;
  if (INPUT_STRUCTURE) n_input_feats += N_FEATURES_STRUCTURE;

  if (CONTEXT_SIZE > 0) {
    n_input_feats *= (1 + CONTEXT_SIZE * 4);
  }

  if (INPUT_TEXT) {
    p_w2v = model.add_lookup_parameters(SIMPLE_VOCAB_SIZE, {DIM_INPUT});
    if (word_vectors.size() > 0)
      initialize_word_vectors(p_w2v, word_vectors);
    if (message_representation == kLSTM) {
      if (FIXED_SELECT) n_input_feats += DIM_LSTM_HIDDEN * 2;
      else n_input_feats += DIM_LSTM_HIDDEN * 4;

      if (CONTEXT_SIZE > 0 && CONTEXT_TEXT) {
        n_input_feats += (CONTEXT_SIZE * DIM_LSTM_HIDDEN * 8);
      }
    } else if (message_representation == kAvWord) {
      if (FIXED_SELECT) n_input_feats += DIM_INPUT;
      else n_input_feats += DIM_INPUT * 2;

      if (CONTEXT_SIZE > 0 && CONTEXT_TEXT) {
        n_input_feats += (CONTEXT_SIZE * DIM_INPUT * 4);
      }
    }

///    p_tt_matrix = model.add_parameters({DIM_INPUT, DIM_INPUT});
///    p_tt_vector = model.add_parameters({DIM_INPUT, 1});
  }


  p_i2h_pair = model.add_parameters({DIM_FF_HIDDEN_PAIR, n_input_feats});
  p_i2h_b_pair = model.add_parameters({DIM_FF_HIDDEN_PAIR});
  for (int i = 1; i < LAYERS_FF_PAIR; i++) {
    p_h2h_pair.push_back(model.add_parameters({DIM_FF_HIDDEN_PAIR, DIM_FF_HIDDEN_PAIR}));
    p_h2h_b_pair.push_back(model.add_parameters({DIM_FF_HIDDEN_PAIR}));
  }
  p_h2o_pair = model.add_parameters({1, DIM_FF_HIDDEN_PAIR});

  unsigned int input_size = n_input_feats * (FIXED_SELECT_DIST + 1);
  if (LAYERS_FF_PAIR > 0) {
    input_size = DIM_FF_HIDDEN_PAIR * (FIXED_SELECT_DIST + 1);
  }
  p_i2h = model.add_parameters({DIM_FF_HIDDEN, input_size});
  p_i2h_b = model.add_parameters({DIM_FF_HIDDEN});
  for (int i = 1; i < LAYERS_FF; i++) {
    p_h2h.push_back(model.add_parameters({DIM_FF_HIDDEN, DIM_FF_HIDDEN}));
    p_h2h_b.push_back(model.add_parameters({DIM_FF_HIDDEN}));
  }
  p_h2o = model.add_parameters({FIXED_SELECT_DIST + 1, DIM_FF_HIDDEN});

  p_i2h_refine = model.add_parameters({DIM_FF_HIDDEN_REFINE, n_input_feats});
  p_i2h_b_refine = model.add_parameters({DIM_FF_HIDDEN_REFINE});
  for (int i = 1; i < LAYERS_FF_REFINE; i++) {
    p_h2h_refine.push_back(model.add_parameters({DIM_FF_HIDDEN_REFINE, DIM_FF_HIDDEN_REFINE}));
    p_h2h_b_refine.push_back(model.add_parameters({DIM_FF_HIDDEN_REFINE}));
  }
  p_h2o_refine = model.add_parameters({1, DIM_FF_HIDDEN_REFINE});

}

void FeedForwardModel::printWeights() {
///  cout << "p_i2h " << p_i2h.dim() << " ";
///  for (unsigned i = 0; i < DIM_FF_HIDDEN; i++) {
///    for (unsigned j = 0; j < N_FEATURES_BASE; j++) {
///      cout << p_i2h.get()->values.v[i] << " ";
///    }
///  }
///  cout << endl;
///  cout << "p_h2o " << p_h2o.dim() << " ";
///  for (unsigned i = 0; i < DIM_FF_HIDDEN * 2; i++) {
///    cout << p_h2o.get()->values.v[i] << " ";
///  }
///  cout << endl;
}

void FeedForwardModel::printGradient() {
///  cout << "gp_i2h " << p_i2h.dim() << " ";
///  for (unsigned i = 0; i < DIM_FF_HIDDEN; i++) {
///    for (unsigned j = 0; j < N_FEATURES_BASE; j++) {
///      cout << p_i2h.get()->g.v[i] << " ";
///    }
///  }
///  cout << endl;
///  cout << "gp_h2o " << p_h2o.dim() << " ";
///  for (unsigned i = 0; i < DIM_FF_HIDDEN * 2; i++) {
///    cout << p_h2o.get()->g.v[i] << " ";
///  }
///  cout << endl;
}

void FeedForwardModel::updateAverageSentence(IRCLog* log, unsigned int num) {
  ComputationGraph cg;
  SUBTRACT_AV_SENT = 0;
  Expression e_sentence = makeUtterance(cg, log, num, false);
  SUBTRACT_AV_SENT = 1;
  if (sentences_averaged == 0) {
    av_sentence_vector = as_vector(cg.incremental_forward(e_sentence));
  } else {
    vector<float> sentence = as_vector(cg.incremental_forward(e_sentence));
    for (int i = 0; i < sentence.size(); i++) {
      av_sentence_vector[i] =
        (sentence[i] / (sentences_averaged+1)) + 
        (av_sentence_vector[i] * (sentences_averaged / (sentences_averaged + 1)));
    }
  }
  sentences_averaged += 1;
}
void FeedForwardModel::preprocessPair(
  IRCLog* log, unsigned int source, unsigned int target
) {
  if (INPUT_TEXT && SUBTRACT_AV_SENT) {
    if (sentences_done.find(source) == sentences_done.end()) {
      updateAverageSentence(log, source);
      sentences_done.insert(source);
    }
  }
}
void FeedForwardModel::preprocessSelection(IRCLog* log, unsigned int source) {
  if (INPUT_TEXT && SUBTRACT_AV_SENT) {
    updateAverageSentence(log, source);
  }
}
void FeedForwardModel::preprocessSet(
  IRCLog* log, unsigned int min_source, unsigned int max_source
) {
  if (INPUT_TEXT && SUBTRACT_AV_SENT) {
    for (unsigned int source = min_source; source < max_source; source++) {
      updateAverageSentence(log, source);
    }
  }
}
Expression FeedForwardModel::makeUtterance(
  ComputationGraph& cg, IRCLog* log, unsigned int sentence, bool eval
) {
  Expression ans;
  if (message_representation == kLSTM) {
    if (eval) {
      lstm_l2r.disable_dropout();
      lstm_r2l.disable_dropout();
    } else {
      lstm_l2r.set_dropout(DROPOUT_LSTM_I, DROPOUT_LSTM_H, DROPOUT_LSTM_C);
      lstm_r2l.set_dropout(DROPOUT_LSTM_I, DROPOUT_LSTM_H, DROPOUT_LSTM_C);
    }
    lstm_l2r.start_new_sequence();
    lstm_r2l.start_new_sequence();
    lstm_l2r.add_input(lookup(cg, p_w2v, kSOS));
    lstm_r2l.add_input(lookup(cg, p_w2v, kEOS));
    auto utterance = &(log->messages->at(sentence).tokens);
    for (auto& token : (*utterance)) {
      auto e_w = lookup(cg, p_w2v, token);
      lstm_l2r.add_input(e_w);
    }
    for (auto i = utterance->rbegin() ; i != utterance->rend(); ++i) {
      auto e_w = lookup(cg, p_w2v, *i);
      lstm_r2l.add_input(e_w);
    }
    lstm_l2r.add_input(lookup(cg, p_w2v, kEOS));
    lstm_r2l.add_input(lookup(cg, p_w2v, kSOS));
    vector<Expression> parts;
    for (auto h_l : lstm_l2r.final_h()) parts.push_back(h_l);
    for (auto h_l : lstm_r2l.final_h()) parts.push_back(h_l);
    ans = concatenate(parts);
  } else if (message_representation == kAvWord) {
    auto msentence = &(log->messages->at(sentence));
    vector<Expression> e_words;
    for (auto& token : msentence->simple_tokens) {
      auto w_s = lookup(cg, p_w2v, token);
      if (DROPOUT_TEXTIN > 0.0 && ! eval) w_s = dropout(w_s, DROPOUT_TEXTIN);
      e_words.push_back(w_s);
    }
    ans = sum(e_words) / e_words.size();
  } else {
    assert(false);
  }

  if (SUBTRACT_AV_SENT) {
    Expression av = input(cg, ans.dim(), av_sentence_vector);
    ans = ans - av;
  }

  return ans;
}

Expression FeedForwardModel::makePair(
  ComputationGraph& cg, IRCLog* log, unsigned int source, unsigned int target,
  Expression& e_i2h, Expression& e_i2h_b, vector<Expression>& e_h2hs,
  vector<Expression>& e_h2hs_b, Expression& e_h2o, Expression& source_repr,
  unordered_map<unsigned int, unordered_set<unsigned int>>& links, bool eval
) {
  vector<Expression> e_inputs;

  // Feature inputs
  vector<float> contextual_feats;
  if (INPUT_HAND_CRAFTED) {
    auto features = log->get_features(source, target, links, contextual_feats);
    Expression e_hc = input(cg, {N_FEATURES_BASE}, *features);
    if (DROPOUT_INPUT > 0.0 && ! eval) e_hc = dropout(e_hc, DROPOUT_INPUT);
    e_inputs.push_back(e_hc);

    // Also provide the hand-crafted features for messages either side of this
    // one, as a form of context.
    for (int delta = -CONTEXT_SIZE; delta <= CONTEXT_SIZE; delta += 1) {
      if (delta != 0) {
        int pos = target + delta;
        if (pos >= 0 && pos < log->messages->size()) {
          contextual_feats.clear();
          features = log->get_features(source, pos, links, contextual_feats);
          e_hc = input(cg, {N_FEATURES_BASE}, *features);
          if (DROPOUT_CONTEXTIN > 0.0 && ! eval)
            e_hc = dropout(e_hc, DROPOUT_CONTEXTIN);
          e_inputs.push_back(e_hc);
        } else {
          unsigned int dimensions = N_FEATURES_BASE;
          if (INPUT_STRUCTURE) dimensions += N_FEATURES_STRUCTURE;
          Expression e_hc = zeros(cg, {dimensions});
          e_inputs.push_back(e_hc);
        }
        pos = source + delta;
        if (pos >= 0 && pos < log->messages->size()) {
          contextual_feats.clear();
          features = log->get_features(source, pos, links, contextual_feats);
          e_hc = input(cg, {N_FEATURES_BASE}, *features);
          if (DROPOUT_CONTEXTIN > 0.0 && ! eval)
            e_hc = dropout(e_hc, DROPOUT_CONTEXTIN);
          e_inputs.push_back(e_hc);
        } else {
          unsigned int dimensions = N_FEATURES_BASE;
          if (INPUT_STRUCTURE) dimensions += N_FEATURES_STRUCTURE;
          Expression e_hc = zeros(cg, {dimensions});
          e_inputs.push_back(e_hc);
        }
      }
    }
  }
  if (INPUT_STRUCTURE) {
    Expression e_s = input(cg, {N_FEATURES_STRUCTURE}, contextual_feats);
    if (DROPOUT_INPUT > 0.0 && ! eval) e_s = dropout(e_s, DROPOUT_INPUT);
    e_inputs.push_back(e_s);
  }
  if (INPUT_TEXT) {
    Expression target_repr = makeUtterance(cg, log, target, eval);
    e_inputs.push_back(target_repr);

///    Expression e_tt_matrix = parameter(cg, p_tt_matrix);
///    Expression e_tt_vector = parameter(cg, p_tt_vector);
///
///    Expression combined0 = source_repr * transpose(target_repr);
///    Expression combined1 = cmult(combined0, e_tt_matrix);
///    Expression combined2 = selu(combined1);
///    Expression combined3 = reshape(combined2 * e_tt_vector, {100});
///    e_inputs.push_back(combined3);

    if (! FIXED_SELECT) e_inputs.push_back(source_repr);

    // Euclidean distance
///    Expression euclidean = squared_distance(source_repr, target_repr);
///    e_inputs.push_back(euclidean);

    // Difference between vectors
///    e_inputs.push_back(source_repr - target_repr);

    // Also provide the hand-crafted features for messages either side of this
    // one, as a form of context.
    if (CONTEXT_TEXT) {
      for (int delta = -CONTEXT_SIZE; delta <= CONTEXT_SIZE; delta += 1) {
        if (delta != 0) {
          int pos = target + delta;
          if (pos >= 0 && pos < log->messages->size()) {
            Expression context_repr = makeUtterance(cg, log, pos, eval);
            e_inputs.push_back(context_repr);
          } else {
            Expression context_repr = zeros(cg, {DIM_INPUT});
            if (message_representation == kLSTM) 
              context_repr = zeros(cg, {DIM_LSTM_HIDDEN * 2});
            e_inputs.push_back(context_repr);
          }
          pos = source  + delta;
          if (pos >= 0 && pos < log->messages->size()) {
            Expression context_repr = makeUtterance(cg, log, pos, eval);
            e_inputs.push_back(context_repr);
          } else {
            Expression context_repr = zeros(cg, {DIM_INPUT});
            if (message_representation == kLSTM) 
              context_repr = zeros(cg, {DIM_LSTM_HIDDEN * 2});
            e_inputs.push_back(context_repr);
          }
        }
      }
    }
  }

  Expression e_in = concatenate(e_inputs);

  if (LAYERS_FF_PAIR == 0) {
    return e_in;
  } else {
    Expression e_h = affine_transform({e_i2h_b, e_i2h, e_in});
    for (int i = -1; i < e_h2hs.size() ; i++) {
      if (i >= 0) {
        e_h = affine_transform({e_h2hs_b[i], e_h2hs[i], e_h});
      }
      if (nonlinearity_type_pair == kLogistic) e_h = logistic(e_h);
      else if (nonlinearity_type_pair == kTanh) e_h = tanh(e_h);
      else if (nonlinearity_type_pair == kCube) e_h = cube(e_h);
      else if (nonlinearity_type_pair == kRectify) e_h = rectify(e_h);
      else if (nonlinearity_type_pair == kELU) e_h = elu(e_h);
      else if (nonlinearity_type_pair == kSeLU) e_h = selu(e_h);
///      else if (nonlinearity_type_pair == kSiLU) e_h = silu(e_h);
      else if (nonlinearity_type_pair == kSoftSign) e_h = softsign(e_h);

      if (DROPOUT_FF_PAIR > 0.0 && ! eval) {
        e_h = dropout(e_h, DROPOUT_FF_PAIR);
      }
    }
    return e_h;
  }
}

void FeedForwardModel::addOutputForPair(
  ComputationGraph& cg, IRCLog* log, unsigned int source, unsigned int target,
  Expression& e_i2h, Expression& e_i2h_b, vector<Expression>& e_h2hs,
  vector<Expression>& e_h2hs_b, Expression& e_h2o, Expression& source_repr,
  unordered_map<unsigned int, unordered_set<unsigned int>>& links,
  vector<Expression>& options, bool eval
) {
  if (source - target > MAX_LINK_LENGTH) {
    Expression e_o = input(cg, {1}, {0.0});
    options.push_back(e_o);
  } else {
    Expression e_h = makePair(cg, log, source, target, e_i2h, e_i2h_b,
        e_h2hs, e_h2hs_b, e_h2o, source_repr, links, eval);
    Expression e_o = e_h2o * e_h;
    options.push_back(e_o);
  }
}

void FeedForwardModel::makeGraph(
  ComputationGraph& cg, vector<Expression>& output, IRCLog* log,
  unsigned int source, unsigned int target,
  unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
  bool eval
) {
  // Get parameters
  Expression e_i2h_pair = parameter(cg, p_i2h_pair);
  Expression e_i2h_b_pair = parameter(cg, p_i2h_b_pair);
  vector<Expression> e_h2hs_pair;
  vector<Expression> e_h2hs_b_pair;
  for (int i = 0; i < p_h2h_pair.size(); i++) {
    e_h2hs_pair.push_back(parameter(cg, p_h2h_pair[i]));
    e_h2hs_b_pair.push_back(parameter(cg, p_h2h_b_pair[i]));
  }
  Expression e_h2o_pair = parameter(cg, p_h2o_pair);
  Expression e_i2h = parameter(cg, p_i2h);
  Expression e_i2h_b = parameter(cg, p_i2h_b);
  vector<Expression> e_h2hs;
  vector<Expression> e_h2hs_b;
  for (int i = 0; i < p_h2h.size(); i++) {
    e_h2hs.push_back(parameter(cg, p_h2h[i]));
    e_h2hs_b.push_back(parameter(cg, p_h2h_b[i]));
  }
  Expression e_h2o = parameter(cg, p_h2o);
  if (INPUT_TEXT && message_representation == kLSTM) {
    lstm_l2r.new_graph(cg);
    lstm_r2l.new_graph(cg);
  }

  // Form inference graph
  Expression source_repr;
  if (INPUT_TEXT) {
    source_repr = makeUtterance(cg, log, source, eval);
  }

  addOutputForPair(cg, log, source, target, e_i2h_pair, e_i2h_b_pair,
      e_h2hs_pair, e_h2hs_b_pair, e_h2o_pair, source_repr, current_links,
      output, eval);

  if (eval) {
    if (getPredictionsPair(cg, output))
      current_links[source].insert(target);
  }
}

void FeedForwardModel::makeGraph(
  ComputationGraph& cg, vector<Expression>& output, IRCLog* log,
  unsigned int source,
  unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
  bool eval
) {
  // Get parameters and prepare builders
  Expression e_i2h_pair = parameter(cg, p_i2h_pair);
  Expression e_i2h_b_pair = parameter(cg, p_i2h_b_pair);
  vector<Expression> e_h2hs_pair;
  vector<Expression> e_h2hs_b_pair;
  for (int i = 0; i < p_h2h_pair.size(); i++) {
    e_h2hs_pair.push_back(parameter(cg, p_h2h_pair[i]));
    e_h2hs_b_pair.push_back(parameter(cg, p_h2h_b_pair[i]));
  }
  Expression e_h2o_pair = parameter(cg, p_h2o_pair);

  if (INPUT_TEXT && message_representation == kLSTM) {
    lstm_l2r.new_graph(cg);
    lstm_r2l.new_graph(cg);
  }

  // Form inference graph
  unsigned int choice = 0;
  float best = 0.0;
  Expression source_repr;
  if (INPUT_TEXT) {
    source_repr = makeUtterance(cg, log, source, eval);
  }

  if (FIXED_SELECT) {
    // Go back a fixed distance, get sentence representations and make a
    // decision together
    Expression e_i2h = parameter(cg, p_i2h);
    Expression e_i2h_b = parameter(cg, p_i2h_b);
    vector<Expression> e_h2hs;
    vector<Expression> e_h2hs_b;
    for (int i = 0; i < p_h2h.size(); i++) {
      e_h2hs.push_back(parameter(cg, p_h2h[i]));
      e_h2hs_b.push_back(parameter(cg, p_h2h_b[i]));
    }
    Expression e_h2o = parameter(cg, p_h2o);

    vector<Expression> options;
    unsigned int start =
      (FIXED_SELECT_DIST > source ? 0 : source - FIXED_SELECT_DIST);
    for (unsigned int target = start; target <= source; target++) {
      Expression e_h = makePair(cg, log, source, target, e_i2h_pair,
          e_i2h_b_pair, e_h2hs_pair, e_h2hs_b_pair, e_h2o_pair, source_repr,
          current_links, eval);
      options.push_back(e_h);
    }
    Expression e_h = concatenate(options);

    e_h = affine_transform({e_i2h_b, e_i2h, e_h});
    for (int i = -1; i < e_h2hs.size() ; i++) {
      if (i >= 0) {
        e_h = affine_transform({e_h2hs_b[i], e_h2hs[i], e_h});
      }
      if (nonlinearity_type == kLogistic) e_h = logistic(e_h);
      else if (nonlinearity_type == kTanh) e_h = tanh(e_h);
      else if (nonlinearity_type == kCube) e_h = cube(e_h);
      else if (nonlinearity_type == kRectify) e_h = rectify(e_h);
      else if (nonlinearity_type == kELU) e_h = elu(e_h);
      else if (nonlinearity_type == kSeLU) e_h = selu(e_h);
//      else if (nonlinearity_type == kSiLU) e_h = silu(e_h);
      else if (nonlinearity_type == kSoftSign) e_h = softsign(e_h);

      if (DROPOUT_FF > 0.0 && ! eval) {
        e_h = dropout(e_h, DROPOUT_FF);
      }
    }

    // In this case all the options are in one vector. This should still work
    // with the loss code etc
    Expression e_o = e_h2o * e_h;
    // First insert a placeholder for all the options we didn't consider
    Expression earlier = zeros(cg, {start});
    output.push_back(earlier);
    output.push_back(e_o);
  } else {
    // Score each option independently
    unsigned int start = 0;
    if (source > MAX_LINK_LENGTH) start = source - MAX_LINK_LENGTH;
    for (unsigned int target = start; target <= source; target++) {
      addOutputForPair(cg, log, source, target, e_i2h_pair, e_i2h_b_pair,
          e_h2hs_pair, e_h2hs_b_pair, e_h2o_pair, source_repr, current_links,
          output, eval);
    }
  }

  if (REFINE_PREDICTION) {
    // Do an additional FF layer that considers the source message and the
    // distribution at the moment

    Expression e_i2h_refine = parameter(cg, p_i2h_refine);
    Expression e_i2h_b_refine = parameter(cg, p_i2h_b_refine);
    vector<Expression> e_h2hs_refine;
    vector<Expression> e_h2hs_b_refine;
    for (int i = 0; i < p_h2h_refine.size(); i++) {
      e_h2hs_refine.push_back(parameter(cg, p_h2h_refine[i]));
      e_h2hs_b_refine.push_back(parameter(cg, p_h2h_b_refine[i]));
    }
    Expression e_h2o_refine = parameter(cg, p_h2o_refine);

    // TODO:
    // Take the 40 closest (concatenate output, then cut it up)
    // Concatenate with a representation of the source
    // Run through a ff network
    // Add the result to the original distribution, pad with the unchanged
    // ssection, and update output
    
    // TODO:
    // Alternative:
    // - Calculate the actual values here
    // - Take the top 2 or 3
    // - Reweight them using the various messages as input
  }

  if (eval) {
    vector<unsigned int> targets;
    getPredictionsSelection(cg, source, output, targets);
    for (auto choice : targets) {
      current_links[source].insert(choice);
    }
  }
}
void FeedForwardModel::makeGraphSet(
  ComputationGraph& cg, vector<Expression>& output, IRCLog* log,
  unsigned int min_source, unsigned int max_source,
  unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
  bool eval
) {
  // Get parameters
  Expression e_i2h_pair = parameter(cg, p_i2h_pair);
  Expression e_i2h_b_pair = parameter(cg, p_i2h_b_pair);
  vector<Expression> e_h2hs_pair;
  vector<Expression> e_h2hs_b_pair;
  for (int i = 0; i < p_h2h_pair.size(); i++) {
    e_h2hs_pair.push_back(parameter(cg, p_h2h_pair[i]));
    e_h2hs_b_pair.push_back(parameter(cg, p_h2h_b_pair[i]));
  }
  Expression e_h2o_pair = parameter(cg, p_h2o_pair);
  Expression e_i2h = parameter(cg, p_i2h);
  Expression e_i2h_b = parameter(cg, p_i2h_b);
  vector<Expression> e_h2hs;
  vector<Expression> e_h2hs_b;
  for (int i = 0; i < p_h2h.size(); i++) {
    e_h2hs.push_back(parameter(cg, p_h2h[i]));
    e_h2hs_b.push_back(parameter(cg, p_h2h_b[i]));
  }
  Expression e_h2o = parameter(cg, p_h2o);
  if (INPUT_TEXT && message_representation == kLSTM) {
    lstm_l2r.new_graph(cg);
    lstm_r2l.new_graph(cg);
  }

  // Form inference graph
  // TODO: have a set of links at training that reflect the decisions made in
  // this set.
///  unordered_map<unsigned int, unordered_set<unsigned int>> greedy_links;
  for (unsigned int source = min_source; source < max_source; source++) {
    float best_score = 0.0;
    unsigned int best = 0;
    Expression source_repr;
    if (INPUT_TEXT) source_repr = makeUtterance(cg, log, source, eval);
    unsigned int start = 0;
    if (source > MAX_LINK_LENGTH) start = source - MAX_LINK_LENGTH;
    for (unsigned int target = start; target <= source; target++) {
      addOutputForPair(cg, log, source, target, e_i2h_pair, e_i2h_b_pair,
          e_h2hs_pair, e_h2hs_b_pair, e_h2o_pair, source_repr, current_links,
          output, eval);

      // Update the best score
      float score = as_scalar(cg.incremental_forward(output.back()));
      if (score > best_score || best == target) {
        best_score = score;
        best = target;
      }
    }

    // Add this link to our set
    if (eval) current_links[source].insert(best);
  }
}

