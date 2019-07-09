#ifndef UM_NLP_CONVGRAPH_DATA_H_
#define UM_NLP_CONVGRAPH_DATA_H_

#include <stdio.h>
#include <stdlib.h>

#include "dynet/dynet.h"
#include "dynet/dict.h"

#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>

#include <eval.h>

using namespace std;
using namespace dynet;

// Forward declarations
struct LinkingModel;

// Globals
extern dynet::Dict word_dict;
extern dynet::Dict simple_word_dict;
extern dynet::Dict user_dict;
extern int kSOS;
extern int kEOS;
extern unsigned VOCAB_SIZE;
extern unsigned SIMPLE_VOCAB_SIZE;
extern string feature_config;
extern unsigned int N_FEATURES_BASE;
extern unsigned int N_FEATURES_STRUCTURE;
extern unsigned int QUERY_SET_SIZE;

void set_feature_config(const char* option);

enum InstanceType {
  kFile,
  kPair,
  kSelection
};
ostream& operator<<(
  ostream& os, 
  InstanceType c
);

extern InstanceType instance_type;
void set_instance_type(char* option);

enum Inference {
  kSingleBinary,
  kMultiTargetBinary,
  kMultiTargetSelection,
  kAllBinary,
  kAllSelection,
  kAllStructured
};

class IRCMessage {
 public:
  tm time{0, 0, 0, 1, 0, 0, 0, 0, 0};
  time_t time_in_sec;
  unordered_set<int> targets;
  int user = 0;
  string message = "";
  vector<int> tokens;
  vector<int> simple_tokens; // See tokenise.py for ddescription
  bool is_normal_message = true;

  IRCMessage(
    const string& text,
    const string& simple_text,
    const string& filename,
    const tm& last_time
  );

  void add_targets(
    const unordered_set<int>& seen_users
  );
};

string to_string(const IRCMessage& m);

class IRCLog {
 public:
  const string filename;
  const vector<IRCMessage>* const messages;
  const unsigned int min_source = -1;
  const unsigned int max_source = -1;
  unordered_map<unsigned int, unordered_set<unsigned int>>* links;
  unordered_map<unsigned int, unordered_set<unsigned int>*>* clusters;

  IRCLog(
    string in_filename,
    vector<IRCMessage>* in_messages,
    unsigned int in_min_source,
    unsigned int in_max_source,
    unordered_map<unsigned int, unordered_set<unsigned int>>* in_links
  );

  const vector<float>* get_features(
    unsigned int src_num,
    unsigned int target_num,
    unordered_map<unsigned int, unordered_set<unsigned int>>& links,
    vector<float>& contextual_feats
  );

  bool are_linked(
    unsigned int src_num,
    unsigned int target_num
  ) const;

 private:
  vector<vector<float>*> feature_cache;

  unsigned int pair_to_position(
    unsigned int src_num,
    unsigned int target_num
  );
};

struct Instance {
  IRCLog* log;

  Instance(
    IRCLog* log_in
  ) :
    log(log_in)
  {}

  virtual void preprocess(
    LinkingModel& model
  ) = 0;

  virtual void makeGraph(
    ComputationGraph& cg,
    vector<Expression>& output,
    LinkingModel& model,
    unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
    bool eval
  ) = 0;

  virtual Expression getLoss(
    ComputationGraph& cg,
    vector<Expression>& output,
    LinkingModel& model,
    bool eval
  ) = 0;

  virtual void updateEval(
    ComputationGraph& cg,
    vector<Expression>& output,
    LinkingModel& model,
    Evaluator& score
  ) = 0;

  virtual void printPrediction(
    ComputationGraph& cg, 
    vector<Expression>& output,
    LinkingModel& model
  ) = 0;

  virtual string to_string() const = 0;
};

struct InstancePair : Instance {
  const unsigned int src_num;
  const unsigned int target_num;

  InstancePair(
    IRCLog* log_in,
    unsigned int src,
    unsigned int target
  ) :
    Instance(log_in),
    src_num(src),
    target_num(target)
  {}

  void preprocess(
    LinkingModel& model
  );

  void makeGraph(
    ComputationGraph& cg,
    vector<Expression>& output,
    LinkingModel& model,
    unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
    bool eval
  );

  Expression getLoss(
    ComputationGraph& cg,
    vector<Expression>& output,
    LinkingModel& model,
    bool eval
  );

  void updateEval(
    ComputationGraph& cg,
    vector<Expression>& output,
    LinkingModel& model,
    Evaluator& score
  );

  void printPrediction(
    ComputationGraph& cg, 
    vector<Expression>& output,
    LinkingModel& model
  );

  string to_string() const;
};

struct InstanceSelection : Instance {
  const unsigned int src_num;

  InstanceSelection(
    IRCLog* log_in, 
    unsigned int src
  ) :
    Instance(log_in),
    src_num(src)
  {}

  void preprocess(
    LinkingModel& model
  );

  void makeGraph(
    ComputationGraph& cg,
    vector<Expression>& output,
    LinkingModel& model,
    unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
    bool eval
  );

  Expression getLoss(
    ComputationGraph& cg,
    vector<Expression>& output,
    LinkingModel& model,
    bool eval
  );

  void updateEval(
    ComputationGraph& cg,
    vector<Expression>& output,
    LinkingModel& model,
    Evaluator& score
  );

  void printPrediction(
    ComputationGraph& cg, 
    vector<Expression>& output,
    LinkingModel& model
  );

  string to_string() const;
};

struct InstanceSet : Instance {
  const unsigned int min_source;
  const unsigned int max_source;

  InstanceSet(
    IRCLog* log_in,
    unsigned int min_source_in,
    unsigned int max_source_in
  ) :
    Instance(log_in),
    min_source(min_source_in),
    max_source(max_source_in)
  {}

  void preprocess(
    LinkingModel& model
  );

  void makeGraph(
    ComputationGraph& cg,
    vector<Expression>& output,
    LinkingModel& model,
    unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
    bool eval
  );

  Expression getLoss(
    ComputationGraph& cg,
    vector<Expression>& output,
    LinkingModel& model,
    bool eval
  );

  void updateEval(
    ComputationGraph& cg,
    vector<Expression>& output,
    LinkingModel& model,
    Evaluator& score
  );

  void printPrediction(
    ComputationGraph& cg, 
    vector<Expression>& output,
    LinkingModel& model
  );

  string to_string() const;
};

IRCLog* read_file(
  const string& text_file,
  const string& annotation_file,
  const string& simple_text_file
);

void read_data(
  vector<IRCLog*>& storage,
  vector<Instance*>& instances,
  const string& file_list
);

void read_word_vectors(
  string& filename,
  unordered_map<int, vector<float>>& word_vectors
);

#endif // UM_NLP_CONVGRAPH_DATA_H_
