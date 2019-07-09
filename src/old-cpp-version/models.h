#ifndef UM_NLP_CONVGRAPH_MODELS_H_
#define UM_NLP_CONVGRAPH_MODELS_H_

#include <stdio.h>
#include <stdlib.h>

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/rnn.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"

#include <data.h>
#include <eval.h>

using namespace std;
using namespace dynet;

extern unsigned LAYERS_FF;
extern unsigned LAYERS_FF_PAIR;
extern unsigned LAYERS_FF_REFINE;
extern unsigned LAYERS_LSTM;
extern unsigned DIM_INPUT;
extern unsigned DIM_LSTM_HIDDEN;
extern unsigned DIM_FF_HIDDEN;
extern unsigned DIM_FF_HIDDEN_PAIR;
extern unsigned DIM_FF_HIDDEN_REFINE;
extern float DROPOUT_INPUT;
extern float DROPOUT_TEXTIN;
extern float DROPOUT_CONTEXTIN;
extern float DROPOUT_FF;
extern float DROPOUT_FF_PAIR;
extern float DROPOUT_FF_REFINE;
extern float DROPOUT_LSTM_H;
extern float DROPOUT_LSTM_C;
extern float DROPOUT_LSTM_I;
extern float SELECTION_PROPORTION;
extern int INPUT_HAND_CRAFTED;
extern int INPUT_STRUCTURE;
extern int INPUT_TEXT;
extern int MAX_LINK_LENGTH;
extern int FIXED_SELECT;
extern unsigned int FIXED_SELECT_DIST;
extern int REFINE_PREDICTION;
extern int SUBTRACT_AV_SENT;
extern int CONTEXT_SIZE;
extern int CONTEXT_TEXT;

extern int LOSS_CLUSTER;
extern int LOSS_NO_NORMALIZE;
extern int LOSS_MAX_HINGE;
extern int LOSS_USE_MULTIPLIER;
extern int LOSS_VARY_MARGIN;
extern float LOSS_RIGHT_CLUSTER;
extern float LOSS_WRONG_CLUSTER;
extern float LOSS_EXTRA_NONE;
extern float LOSS_MISSED_NONE;

enum LossType {
  kHinge,
  kCrossEntropy
};
ostream& operator<<(
  ostream& os, 
  LossType c
);
extern LossType loss_type;
void set_loss_type(char* option);

enum NonLinearityType {
  kLogistic,
  kTanh,
  kCube,
  kRectify,
  kELU,
  kSeLU,
  kSiLU,
  kSoftSign
};
ostream& operator<<(
  ostream& os, 
  NonLinearityType c
);
extern NonLinearityType nonlinearity_type;
extern NonLinearityType nonlinearity_type_pair;
extern NonLinearityType nonlinearity_type_refine;
void set_nonlinearity_type(char* option, NonLinearityType* value);

enum MessageRepresentation {
  kAvWord, // Average the word vectors
  kLSTM, // Use the hidden states of an LSTM
  kMaxPool, // Do some form of convolution (not implemented)
};
ostream& operator<<(
  ostream& os, 
  MessageRepresentation c
);
extern MessageRepresentation message_representation;
void set_message_representation(char* option);


class LinkingModel {
 public:
  void printGradient() { }
  virtual void printWeights() = 0;

  virtual void preprocessPair(
    IRCLog* log,
    unsigned int source,
    unsigned int target
  ) = 0;
  virtual void preprocessSelection(
    IRCLog* log,
    unsigned int source
  ) = 0;
  virtual void preprocessSet(
    IRCLog* log,
    unsigned int min_source,
    unsigned int max_source
  ) = 0;

  // Note, the instance cannot be const because we cache features in the log
  // (and so may need to modify it to add to the cache).
  virtual void makeGraph(
    ComputationGraph& cg,
    vector<Expression>& output,
    IRCLog* log,
    unsigned int source,
    unsigned int target,
    unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
    bool eval
  ) = 0;
  virtual void makeGraph(
    ComputationGraph& cg,
    vector<Expression>& output,
    IRCLog* log,
    unsigned int source,
    unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
    bool eval
  ) = 0;
  virtual void makeGraphSet(
    ComputationGraph& cg,
    vector<Expression>& output,
    IRCLog* log,
    unsigned int min_source,
    unsigned int max_source,
    unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
    bool eval
  ) = 0;

  Expression getLoss(
    ComputationGraph& cg,
    vector<Expression>& output,
    bool eval,
    bool linked
  );
  Expression getLoss(
    ComputationGraph& cg,
    unsigned int source,
    vector<Expression>& output,
    bool eval,
    const unordered_set<unsigned int>& gold,
    const unordered_set<unsigned int>* cluster
  );
  Expression getLossSet(
    ComputationGraph& cg,
    vector<Expression>& output,
    bool eval,
    IRCLog* log,
    unsigned int min_source,
    unsigned int max_source
  );

  bool getPredictionsPair(
    ComputationGraph& cg,
    vector<Expression>& output
  );
  void getPredictionsSelection(
    ComputationGraph& cg,
    unsigned int source,
    vector<Expression>& output,
    vector<unsigned int>& targets
  );
  void getPredictionsSet(
    ComputationGraph& cg,
    vector<Expression>& output,
    unsigned int min_source,
    unsigned int max_source,
    unordered_map<unsigned int, unordered_set<unsigned int>>* links
  );

  void updateEval(
    ComputationGraph& cg,
    vector<Expression>& output,
    bool linked,
    Evaluator& score
  );
  void updateEval(
    ComputationGraph& cg,
    vector<Expression>& output,
    unsigned int source,
    const unordered_set<unsigned int>& gold,
    Evaluator& score
  );
  void updateEvalSet(
    ComputationGraph& cg,
    vector<Expression>& output,
    IRCLog* log,
    unsigned int min_source,
    unsigned int max_source,
    Evaluator& score
  );
};

class LinearModel : public LinkingModel {
 private:
  Parameter p_i2o;
  LookupParameter p_w2v;

  Expression makeUtterance(
    ComputationGraph& cg,
    IRCLog* log,
    unsigned int src,
    bool eval
  );

  void printWeights();

  Expression makePair(
    ComputationGraph& cg, 
    IRCLog* log,
    unsigned int source,
    unsigned int target,
    unordered_map<unsigned int, unordered_set<unsigned int>>& links,
    bool eval
  );

  void addOutputForPair(
    ComputationGraph& cg, 
    IRCLog* log,
    unsigned int source,
    unsigned int target,
    Expression& e_i2o,
    unordered_map<unsigned int, unordered_set<unsigned int>>& links,
    vector<Expression>& options,
    bool eval
  );

 public:
  LinearModel(
    ParameterCollection& model,
    unordered_map<int, vector<float>>& word_vectors
  );

  void preprocessPair(
    IRCLog* log,
    unsigned int source,
    unsigned int target
  );
  void preprocessSelection(
    IRCLog* log,
    unsigned int source
  );
  void preprocessSet(
    IRCLog* log,
    unsigned int min_source,
    unsigned int max_source
  );

  void makeGraph(
    ComputationGraph& cg,
    vector<Expression>& output,
    IRCLog* log,
    unsigned int source,
    unsigned int target,
    unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
    bool eval
  );
  void makeGraph(
    ComputationGraph& cg,
    vector<Expression>& output,
    IRCLog* log,
    unsigned int source,
    unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
    bool eval
  );
  void makeGraphSet(
    ComputationGraph& cg,
    vector<Expression>& output,
    IRCLog* log,
    unsigned int min_source,
    unsigned int max_source,
    unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
    bool eval
  );
};

class FeedForwardModel : public LinkingModel {
 private:
  bool first_query = true;

  // Word representation
  LookupParameter p_w2v; // sparse access to vectors for words
  CoupledLSTMBuilder lstm_l2r;
  CoupledLSTMBuilder lstm_r2l;
  vector<float> av_sentence_vector;
  unsigned int sentences_averaged = 0;
  unordered_set<unsigned int> sentences_done; // for the pair case
///  Parameter p_tt_matrix;
///  Parameter p_tt_vector;
  

  // Pair processing
  // i - input
  // h - hidden
  // o - output
  Parameter p_h2o_pair;
  Parameter p_i2h_pair;
  Parameter p_i2h_b_pair; // bias
  vector<Parameter> p_h2h_pair;
  vector<Parameter> p_h2h_b_pair; // bias

  // Fixed selection processing
  Parameter p_h2o;
  Parameter p_i2h;
  Parameter p_i2h_b; // bias
  vector<Parameter> p_h2h;
  vector<Parameter> p_h2h_b; // bias

  // Refinement of the distribution
  Parameter p_h2o_refine;
  Parameter p_i2h_refine;
  Parameter p_i2h_b_refine; // bias
  vector<Parameter> p_h2h_refine;
  vector<Parameter> p_h2h_b_refine; // bias

  void updateAverageSentence(
    IRCLog* log,
    unsigned int num
  );

  Expression makeUtterance(
    ComputationGraph& cg,
    IRCLog* log,
    unsigned int sentence,
    bool eval
  );

  Expression makePair(
    ComputationGraph& cg, 
    IRCLog* log,
    unsigned int source,
    unsigned int target,
    Expression& e_i2h,
    Expression& e_i2h_b,
    vector<Expression>& e_h2hs,
    vector<Expression>& e_h2hs_b,
    Expression& e_h2o,
    Expression& source_repr,
    unordered_map<unsigned int, unordered_set<unsigned int>>& links,
    bool eval
  );

  void addOutputForPair(
    ComputationGraph& cg, 
    IRCLog* log,
    unsigned int source,
    unsigned int target,
    Expression& e_i2h,
    Expression& e_i2h_b,
    vector<Expression>& e_h2hs,
    vector<Expression>& e_h2hs_b,
    Expression& e_h2o,
    Expression& source_repr,
    unordered_map<unsigned int, unordered_set<unsigned int>>& links,
    vector<Expression>& options,
    bool eval
  );

 public:
  FeedForwardModel(
    ParameterCollection& model,
    unordered_map<int, vector<float>>& word_vectors
  );

  void printWeights();
  void printGradient();

  void preprocessPair(
    IRCLog* log,
    unsigned int source,
    unsigned int target
  );
  void preprocessSelection(
    IRCLog* log,
    unsigned int source
  );
  void preprocessSet(
    IRCLog* log,
    unsigned int min_source,
    unsigned int max_source
  );

  void makeGraph(
    ComputationGraph& cg,
    vector<Expression>& output,
    IRCLog* log,
    unsigned int source,
    unsigned int target,
    unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
    bool eval
  );
  void makeGraph(
    ComputationGraph& cg,
    vector<Expression>& output,
    IRCLog* log,
    unsigned int source,
    unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
    bool eval
  );
  void makeGraphSet(
    ComputationGraph& cg,
    vector<Expression>& output,
    IRCLog* log,
    unsigned int min_source,
    unsigned int max_source,
    unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
    bool eval
  );
};

#endif // UM_NLP_CONVGRAPH_MODELS_H_
