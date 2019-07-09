#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <sys/types.h>
#include <unistd.h>
#include <assert.h>

#include <ctime>
#include <tuple>
#include <iostream>
#include <fstream>
#include <sstream>

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/dict.h"
#include "dynet/io.h"

#include <models.h>
#include <data.h>
#include <eval.h>

using namespace std;
using namespace dynet;

float CLIP_THRESHOLD = 1.0;

// Consider switching to gflags - https://gflags.github.io/gflags/
int main(int argc, char** argv) {
  // Interpret command line options
  // First, let DyNet get its args (this modifies argc and argv in the process)
  initialize(argc, argv);
  static struct option long_opts[] = {
    {"dim-lstm-hidden", required_argument, nullptr, 1},
    {"dim-input", required_argument, nullptr, 2},
    {"layers-ff", required_argument, nullptr, 3},
    {"layers-lstm", required_argument, nullptr, 4},
    {"model", required_argument, nullptr, 5},
    {"dim-ff-hidden", required_argument, nullptr, 6},
    {"input-text", no_argument, &INPUT_TEXT, 1},
    {"input-structure", no_argument, &INPUT_STRUCTURE, 1},
    {"input-hand-crafted", no_argument, &INPUT_HAND_CRAFTED, 1},
    {"model-file", required_argument, nullptr, 7},
    {"prefix", required_argument, nullptr, 8},
    {"feature-config", required_argument, nullptr, 9},
    {"nonlinearity-pair", required_argument, nullptr, 10},
    {"message-representation", required_argument, nullptr, 11},
    {"fixed-select", no_argument, &FIXED_SELECT, 1},
    {"nonlinearity", required_argument, nullptr, 13},
    {"selection-proportion", required_argument, nullptr, 14},
    {"fixed-select-dist", required_argument, nullptr, 15},
    {"layers-ff-pair", required_argument, nullptr, 16},
    {"refine-prediction", no_argument, &REFINE_PREDICTION, 1},
    {"dim-ff-hidden-pair", required_argument, nullptr, 18},
    {"layers-ff-refine", required_argument, nullptr, 19},
    {"dim-ff-hidden-refine", required_argument, nullptr, 20},
    {"nonlinearity-refine", required_argument, nullptr, 21},
    {"subtract-av-sent", no_argument, &SUBTRACT_AV_SENT, 1},
    {"context-size", required_argument, nullptr, 23},
    {"context-text", no_argument, &CONTEXT_TEXT, 1},

    {"data-train", required_argument, nullptr, 100},
    {"data-dev", required_argument, nullptr, 101},
    {"data-eval", required_argument, nullptr, 102},
    {"query-set-size", required_argument, nullptr, 103},

    {"trainer", required_argument, nullptr, 200},
    {"learning-rate", required_argument, nullptr, 201},
    {"log-freq", required_argument, nullptr, 202},
    {"dev-freq", required_argument, nullptr, 203},
    {"instance-type", required_argument, nullptr, 204},
    {"loss-type", required_argument, nullptr, 205},
    {"no-improvement-cutoff", required_argument, nullptr, 206},
    {"dropout-input", required_argument, nullptr, 207},
    {"dropout-ff", required_argument, nullptr, 208},
    {"dropout-lstm-h", required_argument, nullptr, 209},
    {"dropout-lstm-c", required_argument, nullptr, 210},
    {"dropout-lstm-i", required_argument, nullptr, 211},
    {"word-vector-init", required_argument, nullptr, 212},
    {"dropout-ff-pair", required_argument, nullptr, 213},
    {"dropout-ff-refine", required_argument, nullptr, 214},
    {"max-iterations", required_argument, nullptr, 215},
    {"loss-cluster", no_argument, &LOSS_CLUSTER, 1},
    {"dropout-textin", required_argument, nullptr, 217},
    {"dropout-contextin", required_argument, nullptr, 218},
    {"loss-no-normalize", no_argument, &LOSS_NO_NORMALIZE, 1},
    {"loss-max-hinge", no_argument, &LOSS_MAX_HINGE, 1},
    {"loss-use-multiplier", no_argument, &LOSS_USE_MULTIPLIER, 1},
    {"loss-vary-margin", no_argument, &LOSS_VARY_MARGIN, 1},
    {"loss-right-cluster", required_argument, nullptr, 219},
    {"loss-wrong-cluster", required_argument, nullptr, 220},
    {"loss-extra-none", required_argument, nullptr, 221},
    {"loss-missed-none", required_argument, nullptr, 222},
    {"clipping-threshold", required_argument, nullptr, 223},

    {"max-link-length", required_argument, nullptr, 300},

    {0, 0, 0, 0},
  };
  static auto short_opts = "";
  int option_index = 0;
  int opt;
  string data_train = "";
  string data_dev = "";
  string data_eval = "";
  string prefix = "conv-graph-model";
  string trainer = "momentum";
  string model_type = "linear";
  string word_vector_init = "-";
  string model_file = "";
  unsigned report_every_i = 100;
  unsigned no_improvement_cutoff = 5;
  unsigned dev_every_i_reports = 10;
  float learning_rate = -1.0;
  int max_iterations = 100;
  set_feature_config(feature_config.c_str()); // Set default
  while (1) {
    opt = getopt_long_only(argc, argv, short_opts, long_opts, &option_index);
    if (opt == -1) break;
    switch (opt) {
      case '?': break; // Unknown opt, error message printed by getopt_long

      case 0: break;
      case 1: DIM_LSTM_HIDDEN = stoi(string(optarg)); break;
      case 2: DIM_INPUT = stoi(string(optarg)); break;
      case 3: LAYERS_FF = stoi(string(optarg)); break;
      case 4: LAYERS_LSTM = stoi(string(optarg)); break;
      case 5: model_type = string(optarg); break;
      case 6: DIM_FF_HIDDEN = stoi(string(optarg)); break;
      case 7: model_file = string(optarg); break;
      case 8: prefix = string(optarg); break;
      case 9: set_feature_config(optarg); break;
      case 10: set_nonlinearity_type(optarg, &nonlinearity_type_pair); break;
      case 11: set_message_representation(optarg); break;
      case 13: set_nonlinearity_type(optarg, &nonlinearity_type); break;
      case 14: SELECTION_PROPORTION = stof(string(optarg)); break;
      case 15: FIXED_SELECT_DIST = stof(string(optarg)); break;
      case 16: LAYERS_FF_PAIR = stoi(string(optarg)); break;
      case 18: DIM_FF_HIDDEN_PAIR = stoi(string(optarg)); break;
      case 19: LAYERS_FF_REFINE = stoi(string(optarg)); break;
      case 20: DIM_FF_HIDDEN_REFINE = stoi(string(optarg)); break;
      case 21: set_nonlinearity_type(optarg, &nonlinearity_type_refine); break;
      case 23: CONTEXT_SIZE = stoi(string(optarg)); break;

      case 100: data_train = string(optarg); break;
      case 101: data_dev = string(optarg); break;
      case 102: data_eval = string(optarg); break;
      case 103: QUERY_SET_SIZE = stoi(optarg); break;

      case 200: trainer = string(optarg); break;
      case 201: learning_rate = stof(string(optarg)); break;
      // TODO: Change the next two to also accept floats, and in that case do
      // it that percentage through the data (e.g. 0.5 means every half
      // iteration)
      case 202: report_every_i = stoi(string(optarg)); break;
      case 203: dev_every_i_reports = stoi(string(optarg)); break;
      case 204: set_instance_type(optarg); break;
      case 205: set_loss_type(optarg); break;
      case 206: no_improvement_cutoff = stoi(string(optarg)); break;
      case 207: DROPOUT_INPUT = stof(string(optarg)); break;
      case 208: DROPOUT_FF = stof(string(optarg)); break;
      case 209: DROPOUT_LSTM_H = stof(string(optarg)); break;
      case 210: DROPOUT_LSTM_C = stof(string(optarg)); break;
      case 211: DROPOUT_LSTM_I = stof(string(optarg)); break;
      case 212: word_vector_init = string(optarg); break;
      case 213: DROPOUT_FF_PAIR = stof(string(optarg)); break;
      case 214: DROPOUT_FF_REFINE = stof(string(optarg)); break;
      case 215: max_iterations = stoi(string(optarg)); break;
      case 217: DROPOUT_TEXTIN = stof(string(optarg)); break;
      case 218: DROPOUT_CONTEXTIN = stof(string(optarg)); break;
      case 219: LOSS_RIGHT_CLUSTER = stof(string(optarg)); break;
      case 220: LOSS_WRONG_CLUSTER = stof(string(optarg)); break;
      case 221: LOSS_EXTRA_NONE = stof(string(optarg)); break;
      case 222: LOSS_MISSED_NONE = stof(string(optarg)); break;
      case 223: CLIP_THRESHOLD = stof(string(optarg)); break;

      case 300: MAX_LINK_LENGTH = stoi(string(optarg)); break;

      default: break;
    }
  }
  assert(INPUT_HAND_CRAFTED + INPUT_STRUCTURE + INPUT_TEXT > 0);
  assert(CONTEXT_SIZE < 10);
  assert(MAX_LINK_LENGTH < 1000);

  if (data_train.compare("") == 0) {
    cerr << "No training data defined" << endl;
    return 1;
  }

  // Print config
  cout << "Running with:"
    << "\n    Prefix                     " << prefix
    << "\n    Data, data-train           " << data_train
    << "\n    Data, data-dev             " << data_dev
    << "\n    Data, data-eval            " << data_eval
    << "\n    Data, query-set-size       " << QUERY_SET_SIZE 
    << "\n    Model, model               " << model_type
    << "\n    Model, model file          " << model_file
    << "\n    Model, dim-input           " << DIM_INPUT
    << "\n    Model, dim-lstm-hidden     " << DIM_LSTM_HIDDEN
    << "\n    Model, dim-ff-hidden       " << DIM_FF_HIDDEN
    << "\n    Model, dim-ff-hidden-pair  " << DIM_FF_HIDDEN_PAIR
    << "\n    Model, dim-ff-hidden-refine " << DIM_FF_HIDDEN_REFINE
    << "\n    Model, layers-ff           " << LAYERS_FF
    << "\n    Model, layers-ff-pair      " << LAYERS_FF_PAIR
    << "\n    Model, layers-ff-refine    " << LAYERS_FF_REFINE
    << "\n    Model, layers-lstm         " << LAYERS_LSTM
    << "\n    Model, input-text          " << INPUT_TEXT
    << "\n    Model, input-structure     " << INPUT_STRUCTURE
    << "\n    Model, input-hand-crafted  " << INPUT_HAND_CRAFTED
    << "\n    Model, feature-config      " << feature_config
    << "\n    Model, nonlinearity        " << nonlinearity_type
    << "\n    Model, nonlinearity-pair   " << nonlinearity_type_pair
    << "\n    Model, nonlinearity-refine " << nonlinearity_type_refine
    << "\n    Model, message-representation " << message_representation
    << "\n    Model, fixed-select        " << FIXED_SELECT
    << "\n    Model, selection-proportion " << SELECTION_PROPORTION
    << "\n    Model, fixed-select-dist   " << FIXED_SELECT_DIST
    << "\n    Model, refine-prediction   " << REFINE_PREDICTION
    << "\n    Model, subtract-av-sent    " << SUBTRACT_AV_SENT
    << "\n    Model, context-size        " << CONTEXT_SIZE
    << "\n    Model, context-text        " << CONTEXT_TEXT
    << "\n    Learning, trainer          " << trainer
    << "\n    Learning, learning-rate    " << learning_rate
    << "\n    Learning, log-freq         " << report_every_i
    << "\n    Learning, dev-freq         " << dev_every_i_reports
		<< "\n    Learning, instance-type    " << instance_type
		<< "\n    Learning, loss-type        " << loss_type
		<< "\n    Learning, no-improvement-cutoff " << no_improvement_cutoff
		<< "\n    Learning, max-iterations   " << max_iterations
		<< "\n    Learning, dropout-input    " << DROPOUT_INPUT
		<< "\n    Learning, dropout-textin   " << DROPOUT_TEXTIN
		<< "\n    Learning, dropout-contextin " << DROPOUT_CONTEXTIN
		<< "\n    Learning, dropout-ff       " << DROPOUT_FF
		<< "\n    Learning, dropout-ff-pair  " << DROPOUT_FF_PAIR
		<< "\n    Learning, dropout-ff-refine " << DROPOUT_FF_REFINE
		<< "\n    Learning, dropout-lstm-h   " << DROPOUT_LSTM_H
		<< "\n    Learning, dropout-lstm-c   " << DROPOUT_LSTM_C
		<< "\n    Learning, dropout-lstm-i   " << DROPOUT_LSTM_I
    << "\n    Learning, word-vector-init " << word_vector_init
    << "\n    Learning, loss-cluster     " << LOSS_CLUSTER
    << "\n    Learning, loss-no-normalize " << LOSS_NO_NORMALIZE
    << "\n    Learning, loss-max-hinge   " << LOSS_MAX_HINGE
    << "\n    Learning, loss-use-multiplier " << LOSS_USE_MULTIPLIER
    << "\n    Learning, loss-vary-margin " << LOSS_VARY_MARGIN
    << "\n    Learning, loss-right-cluster " << LOSS_RIGHT_CLUSTER
    << "\n    Learning, loss-wrong-cluster " << LOSS_WRONG_CLUSTER
    << "\n    Learning, loss-extra-none  " << LOSS_EXTRA_NONE
    << "\n    Learning, loss-missed-none " << LOSS_MISSED_NONE
    << "\n    Learning, clipping-threshold " << CLIP_THRESHOLD
    << "\n    Inference, max-link-length " << MAX_LINK_LENGTH 
    << endl;

  // Read data, including preparing dictionaries
  vector<IRCLog*> training_logs, dev_logs;
  vector<Instance*> training, dev;
  unordered_map<int, vector<float>> word_vectors;
  if (word_vector_init.compare("-") != 0) {
    read_word_vectors(word_vector_init, word_vectors);
  }
  read_data(training_logs, training, data_train);
  word_dict.freeze(); // no new word types allowed
  word_dict.set_unk("<unk>");
  simple_word_dict.freeze(); // no new word types allowed
  simple_word_dict.set_unk("<unk>");
  VOCAB_SIZE = word_dict.size();
  SIMPLE_VOCAB_SIZE = simple_word_dict.size();
  cerr << "Vocab sizes: " << VOCAB_SIZE << ", " << SIMPLE_VOCAB_SIZE << endl;

  read_data(dev_logs, dev, data_dev);
  // TODO: Dictionary saving, for running without reading the training data

  ostringstream os;
  time_t t = time(0);
  char timestr[100];
  strftime(timestr, sizeof(timestr), "%Y-%m-%d.%H-%M", localtime(&t));
  os << prefix << '_' << timestr << "_pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;

  ParameterCollection model;
  Trainer* sgd;
  if (trainer.compare("adadelta") == 0) {
    cout << "Using Adadelta\n";
    sgd = new AdadeltaTrainer(model);
  } else if (trainer.compare("adagrad") == 0) {
    if (learning_rate < 0.0) learning_rate = 0.1;
    cout << "Using Adagrad "<< learning_rate <<"\n";
    sgd = new AdagradTrainer(model, learning_rate);
  } else if (trainer.compare("adam") == 0) {
    if (learning_rate < 0.0) learning_rate = 0.001;
    cout << "Using Adam "<< learning_rate <<"\n";
    sgd = new AdamTrainer(model, learning_rate);
  } else if (trainer.compare("sgd") == 0) {
    if (learning_rate < 0.0) learning_rate = 0.001;
    cout << "Using Plain SGD "<< learning_rate <<"\n";
    sgd = new SimpleSGDTrainer(model, learning_rate);
  } else {
    if (trainer.compare("momentum") != 0) {
      cerr << "Unknown trainer, defaulting to momentum\n";
    }
    if (learning_rate < 0.0) learning_rate = 0.01;
    cout << "Using Momentum "<< learning_rate <<"\n";
    sgd = new MomentumSGDTrainer(model, learning_rate);
  }

///  sgd->clip_threshold = CLIP_THRESHOLD;
///  sgd->clipping_enabled = CLIP_THRESHOLD > 0.0;
///  sgd->status();

  LinkingModel* linker;
  if (
    model_type.compare("feedforward") == 0 || 
    model_type.compare("ff") == 0
  ) {
    cout << "Using Feedforward model\n";
    linker = new FeedForwardModel(model, word_vectors);
  } else {
    if (model_type.compare("linear") != 0) {
      cerr << "Unknown model_type, defaulting to Linear\n";
    }
    cout << "Using Linear model\n";
    linker = new LinearModel(model, word_vectors);
  }

  if (model_file.compare("") != 0) {
    TextFileLoader l(model_file);
    l.populate(model);
  }

  // Preprocessing
  for (auto sample : training) {
    sample->preprocess(*linker);
  }

  unsigned si = 0;
  vector<unsigned> order(training.size());
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
  int report = 0;
  unsigned lines = 0;
  double best = 0.0;
  unsigned int best_last_updated = 0;
  unsigned int current_iteration = 0;
  while (
    best_last_updated < no_improvement_cutoff &&
    current_iteration < max_iterations
  ) {
    current_iteration += 1;
    Timer iteration("completed in");
    double loss = 0;
    unsigned samples = 0;
    double correct = 0;
    Evaluator train_score;
///    cerr << "Starting" << endl;
    for (unsigned i = 0; i < report_every_i; ++i) {
      if (si == training.size()) {
        si = 0;
        cerr << "**SHUFFLE\n";
        shuffle(order.begin(), order.end(), *rndeng);
      }

      // Get instance
      auto sample = training[order[si]];
      ++si;
      // Build graph, using gold links for context
      ComputationGraph cg;
      vector<Expression> output;
///      cerr << "calling make graph" << endl;
      sample->makeGraph(cg, output, *linker, *(sample->log->links), false);
      // Get loss expression
///      cerr << "getting loss" << endl;
      Expression loss_expr = sample->getLoss(cg, output, *linker, false);
      // Evaluate graph and update loss
      loss += as_scalar(cg.incremental_forward(loss_expr));
      // Update accuracy with correct
///      cerr << "updating eval" << endl;
      sample->updateEval(cg, output, *linker, train_score);

      // Do update
      cg.backward(loss_expr);
      sgd->update();

      ++lines;
      ++samples;
    }
    sgd->status();
    cerr << endl;

    cerr << " Train"
      << " [epoch=" << (lines / (double)training.size()) << "]"
      << " L = " << (loss / samples)
      << " " << train_score.to_string();

    // show score on dev data
    report++;
    if (report % dev_every_i_reports == 0) {
      Evaluator score;
      double dloss = 0;
      unsigned dev_samples = 0;
      double dcorr = 0;
      int tp = 0;
      int fp = 0;
      int tn = 0;
      int fn = 0;
      unordered_map<string, unordered_map<unsigned int, unordered_set<unsigned int>>> links;
      for (auto sample : dev) {
        ComputationGraph cg;
        // Build graph, using links determined so far for context
        auto current_links = links[sample->log->filename];
        // TODO: determine links in the earlier part of the log
        vector<Expression> output;
        sample->makeGraph(cg, output, *linker, current_links, true);
        // Get loss expression
        Expression loss_expr = sample->getLoss(cg, output, *linker, true);
        // Evaluate graph and update loss
        dloss += as_scalar(cg.incremental_forward(loss_expr));
        // Update accuracy
        sample->updateEval(cg, output, *linker, score);
        
        ++dev_samples;
      }
      if (score.f() > best) {
        best = score.f();
        TextFileSaver s(fname);
        s.save(model);
        best_last_updated = 0;
      } else {
        best_last_updated += 1;
      }

      ostringstream to_log;
      to_log << "\n***DEV"
        << " [epoch=" << (lines / (double)training.size()) << "]"
        << " L = " << (dloss / dev_samples)
        << " " << score.to_string()
        << "\n";
      auto to_log_str = to_log.str();
      cerr << to_log_str;
      cout << to_log_str;
    }
  }
  cout << "\nFinished training.\n";
  cerr << "\nFinished training.\n";
  linker->printWeights();

  // TODO: Save model now if not already done

  if (data_eval.length() > 0) {
    vector<Instance*> eval;
    vector<IRCLog*> eval_logs;
    read_data(eval_logs, eval, data_eval);

    unordered_map<string, unordered_map<unsigned int, unordered_set<unsigned int>>> links;
    for (auto sample : eval) {
      ComputationGraph cg;
      auto current_links = links[sample->log->filename];
      vector<Expression> output;
      sample->makeGraph(cg, output, *linker, current_links, true);
      sample->printPrediction(cg, output, *linker);
    }
  }
}

