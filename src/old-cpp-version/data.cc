#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "dynet/dynet.h"
#include "dynet/dict.h"

#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>
#include <locale>
#include <iomanip>
#include <climits>

#include <data.h>
#include <models.h>

using namespace std;
using namespace dynet;

dynet::Dict word_dict;
dynet::Dict simple_word_dict;
dynet::Dict user_dict;
string kSOS_str = "<s>";
string kEOS_str = "</s>";
int kSOS = word_dict.convert(kSOS_str);
int kEOS = word_dict.convert(kEOS_str);
unsigned VOCAB_SIZE = 0;
unsigned SIMPLE_VOCAB_SIZE = 0;
unsigned int QUERY_SET_SIZE = 100; // Lines to include in an InstanceSet

// This string determines the set of input features to use. The colon separate
// base features and structural features.
string feature_config = "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy:yyyyyyyyy";
unsigned int N_FEATURES_BASE = 0;
unsigned int N_FEATURES_STRUCTURE = 0;

void set_feature_config(const char* option) {
  feature_config = string(option);
  bool past_colon = false;
  N_FEATURES_STRUCTURE = 0;
  N_FEATURES_BASE = 0;
  for (char& c : feature_config) {
    if (c == ':') past_colon = true;
    else if (c == 'y') {
      if (past_colon) N_FEATURES_STRUCTURE += 1;
      else N_FEATURES_BASE += 1;
    }
  }
  cout <<"Set feature config with: "<< N_FEATURES_BASE <<" "<< N_FEATURES_STRUCTURE <<"\n";
}

InstanceType instance_type = kSelection;

ostream& operator<<(ostream& os, InstanceType c) {
  switch (c) {
    case kFile: os << "kFile"; break;
    case kSelection: os << "kSelection"; break;
    case kPair: os << "kPair"; break;
    default: os.setstate(std::ios_base::failbit);
  }
  return os;
}

void set_instance_type(char* option) {
  if (strcmp(option, "file") ==  0 ||
      strcmp(option, "kFile") == 0) {
    instance_type = kFile;
  } else if (strcmp(option, "pair") == 0 ||
      strcmp(option, "kPair") == 0) {
    instance_type = kPair;
  } else {
    instance_type = kSelection;
  }
}

IRCMessage::IRCMessage(
  const string& text,
  const string& simple_text,
  const string& filename,
  const tm& last_time
) {
  istringstream parts(text);
  istringstream simple_parts(simple_text);

  string start;
  parts >> start;
  string username;
  parts >> username;

  // Set the message (including targets intentionally)
  int consumed = start.length() + username.length() + 2;
  if (consumed > text.length()) consumed = text.length();
  message = kSOS_str +" "+ text.substr(consumed) +" "+ kEOS_str;
  tokens = read_sentence(message, word_dict);
  simple_tokens = read_sentence(simple_text, simple_word_dict);

  // If there is a filename at the start, strip it out
  bool is_second_12 = false;
  string year;
  string month;
  string day;
  string hour = to_string(last_time.tm_hour);
  string minute = to_string(last_time.tm_min);
  if (start.find("/") != string::npos) {
    year = start.substr(0, 4);
    month = start.substr(5, 2);
    day = start.substr(8, 2);
    start = start.substr(start.find(":")+1);
  } else {
    // Prepare time based on filename and last_time
    is_second_12 = (filename.find("second") != string::npos);
    string file_ending = filename.substr(filename.rfind("/") + 1);
    istringstream datestream(file_ending);
    getline(datestream, year, '-');
    getline(datestream, month, '-');
    getline(datestream, day, '-');
  }

  is_normal_message = (start.compare("===") != 0);

  // Set the time
  if (is_normal_message) {
    minute = start.substr(4, 2);
    hour = start.substr(1, 2);
    if (is_second_12 && hour.compare("12") != 0)
      hour = to_string(stoi(hour) + 12);
  }
  istringstream date(year +" "+ month +" "+ day +" "+ hour +" "+ minute);
  date >> get_time(&time, "%Y %m %d %H %M");
  time_in_sec = mktime(&time);

  // Remove angle brackets from the username if needed
  if (is_normal_message) {
    if (username[0] == '<') {
      username = username.substr(1, username.length() - 2);
    } else {
      parts >> username;
      consumed += username.length() + 1;
      if (consumed > text.length()) consumed = text.length();
      message = "* "+ text.substr(consumed);
    }
  }
  user = user_dict.convert(username);
}

void IRCMessage::add_targets(
  const unordered_set<int>& seen_users
) {
  istringstream parts(message);
  string token;
  // We consider every token because sometimes there are multiple targets,
  // or the name is mentioned later in the message.
  while (parts >> token) {
    string ltoken(token);
    transform(ltoken.begin(), ltoken.end(), ltoken.begin(), ::tolower);
    // Rather than modifying the token to get a name, just see if any name
    // is in it.
    for (const auto& target : seen_users) {
      string ltarget(user_dict.convert(target));
      transform(ltarget.begin(), ltarget.end(), ltarget.begin(), ::tolower);
      size_t pos = ltoken.find(ltarget);
      // Avoid 1 character names (e.g. "a")
      if (pos != string::npos && ltarget.length() > 1) {
        // Check that all other characters aren't name-like
        // We will still get some false positives, like "default" or
        // "root", but we can live with that.
        string rest = ltoken.substr(0, pos) +
          ltoken.substr(pos + ltarget.length(), ltoken.length());
        if (none_of(rest.begin(), rest.end(), ::isalnum)) {
          targets.insert(target);
        }
      }
    }
  }
}

// Do an approimate union-find, based on the assumption that every
// conversation has a single root message that all others have a path to.
// That may not be true (for example if message 2 response to 0 and 1, which
// are not linked). However, it is good enough for now.
unordered_set<unsigned int>* add_to_cluster(
  const unsigned int current, 
  unordered_map<unsigned int, unordered_set<unsigned int>>& links,
  unordered_map<unsigned int, unordered_set<unsigned int>*>& clusters
) {
  // If this already has its set determined, return it
  auto c_cluster = clusters.find(current);
  if (c_cluster != clusters.end()) {
    return c_cluster->second;
  }

  // Otherwise, recurse to get the ultimate source
  unordered_set<unsigned int>* cluster;
  auto c_links = links.find(current);
  bool found = false;
  if (c_links != links.end()) {
    for (unsigned int num : c_links->second) {
      if (num != current) {
        cluster = add_to_cluster(num, links, clusters);
        found = true;
      }
    }
  }

  if (! found) {
    cluster = new unordered_set<unsigned int>();
  }
  cluster->insert(current);
  clusters.insert({current, cluster});
  return cluster;
}

IRCLog::IRCLog(
  string in_filename, vector<IRCMessage>* in_messages,
  unsigned int in_min_source, unsigned int in_max_source,
  unordered_map<unsigned int, unordered_set<unsigned int>>* in_links
) :
    filename(in_filename),
    messages(in_messages),
    min_source(in_min_source),
    max_source(in_max_source),
    links(in_links)
{
  clusters = new unordered_map<unsigned int, unordered_set<unsigned int>*>();
  // Based on links, derive sets
  for (int current = messages->size() - 1; current >= 0; current--) {
    add_to_cluster(current, *links, *clusters);
  }
}

unsigned int IRCLog::pair_to_position(
  unsigned int src_num, unsigned int target_num
) {
  // For each src we need space for every pair that could be asked about,
  // which is bounded.
  unsigned int space_per_src = MAX_LINK_LENGTH + CONTEXT_SIZE * 2 + 1;
  unsigned int shifted_target = (src_num + CONTEXT_SIZE) - target_num;
  unsigned int pos = src_num * space_per_src + shifted_target;
  return pos;
}

const vector<float>* IRCLog::get_features(
  unsigned int src_num, unsigned int target_num,
  unordered_map<unsigned int, unordered_set<unsigned int>>& links,
  vector<float>& contextual_feats
) {
  // TODO:
  // - Add features indicating whether the speaker, parent, etc is a bot
  // - Thread level: is this user in this conversation so far? Are they in a
  // different conversaion?

  // Maximum distance to look forward / back in loops
  const int MAX_CHECK_DIST = 500;

  // Variables used by both feature generation processes
  const float y = 1.0;
  const float n = 0.0;
  const auto src = &(messages->at(src_num));
  const bool has_target = (target_num != src_num);
  const auto target = (has_target ? &(messages->at(target_num)) : nullptr);

  unsigned int earlier_num = target_num;
  unsigned int later_num = src_num;
  if (src_num < target_num) {
    earlier_num = src_num;
    later_num = target_num;
  }

  // Context insensitive features (i.e. ignoring the provided links)
  unsigned int pos = pair_to_position(src_num, target_num);
  while (feature_cache.size() <= pos) feature_cache.push_back(nullptr);
  vector<float>* features = feature_cache[pos];
  if (features == nullptr) {
    vector<float> new_feats;

    // Features that depend on source only
    float src_normal = (src->is_normal_message ? y : n);
    new_feats.push_back(src_normal);

    int src_hour = src->time.tm_hour;
    new_feats.push_back(src_hour - 12.0);

    bool src_has_targets = false;
    if (src->targets.size() > 0) src_has_targets = true;
    new_feats.push_back(src_has_targets ? y : n);

    int src_last = -1;
    int loop_end = src_num - MAX_CHECK_DIST;
    bool prev_src_has_targets = false;
    if (loop_end < 0) loop_end = 0;
    for (int i = src_num - 1; i >= loop_end; i--) {
      if (messages->at(i).user == src->user) {
        if (messages->at(i).targets.size() > 0) {
          prev_src_has_targets = true;
        }
        src_last = i;
      }
    }
    new_feats.push_back(prev_src_has_targets ? y : n);
    int src_gap = src_num - src_last;
    new_feats.push_back(src_gap == 1 ? y : n);
    new_feats.push_back(src_gap > 1 && src_gap <= 5 ? y : n);
    new_feats.push_back(src_gap > 5 && src_gap <= 20 ? y : n);
    new_feats.push_back(src_gap > 20 && src_gap > src_num ? y : n);
    new_feats.push_back(src_gap > src_num ? y : n); // ie, no previous message

    // Features that depend on target only
    float target_normal = (has_target && target->is_normal_message ? y : n);
    new_feats.push_back(target_normal);

    int target_hour = src_hour;
    if (has_target) target_hour = target->time.tm_hour;
    new_feats.push_back(target_hour - 12.0);

    float previous_is_same_speaker = n;
    if (has_target && target_num > 0) {
      if (target->user == messages->at(target_num - 1).user)
        previous_is_same_speaker = y;
    }
    new_feats.push_back(previous_is_same_speaker);

    float next_is_same_speaker = n;
    if (has_target && target_num + 1 < messages->size()) {
      if (target->user == messages->at(target_num + 1).user)
        next_is_same_speaker = y;
    }
    new_feats.push_back(next_is_same_speaker);

    float target_has_targets = n;
    if (has_target && target->targets.size() > 0) target_has_targets = y;
    new_feats.push_back(target_has_targets);

    int target_last = -1;
    bool prev_target_has_targets = false;
    if (has_target) {
      int loop_end = target_num - MAX_CHECK_DIST;
      if (loop_end < 0) loop_end = 0;
      for (int i = target_num - 1; i >= loop_end; i--) {
        if (messages->at(i).user == target->user) {
          if (messages->at(i).targets.size() > 0) {
            prev_target_has_targets = true;
          }
          target_last = i;
        }
      }
    } else {
      target_last = target_num;
    }
    new_feats.push_back(prev_target_has_targets ? y : n);
    int target_gap = target_num - target_last;
    new_feats.push_back(target_gap == 1 ? y : n);
    new_feats.push_back(target_gap > 1 && target_gap <= 5 ? y : n);
    new_feats.push_back(target_gap > 5 && target_gap <= 20 ? y : n);
    new_feats.push_back(target_gap > 20 && target_gap > target_num ? y : n);
    new_feats.push_back(target_gap > target_num ? y : n); // ie, no previous

    // Features that depend on both
    new_feats.push_back(has_target ? y : n);

    int msg_diff = 0;
    if (has_target) {
      msg_diff = later_num - earlier_num;
    }
    new_feats.push_back(msg_diff - 50.0);
    new_feats.push_back(msg_diff > 1 ? y : n);
    new_feats.push_back(msg_diff > 2 ? y : n);

    float time_diff = 0;
    if (has_target) {
      if (src_num > target_num)
        time_diff = difftime(src->time_in_sec, target->time_in_sec) / 60.0;
      else
        time_diff = difftime(target->time_in_sec, src->time_in_sec) / 60.0;
    }
    new_feats.push_back(time_diff - 100.0);

    float src_target_match = n;
    if (has_target && target->targets.count(src->user) > 0) {
      src_target_match = y;
    }
    new_feats.push_back(src_target_match);

    float target_src_match = n;
    if (has_target && src->targets.count(target->user) > 0) {
      target_src_match = y;
    }
    new_feats.push_back(target_src_match);

    bool none_in_between_from_target = true;
    bool none_in_between_from_src = true;
    bool previously_src_addressed_target = false;
    bool previously_target_addressed_src = false;
    bool future_src_addressed_target = false;
    bool future_target_addressed_src = false;
    if (has_target) {
      for (unsigned int i = earlier_num + 1; i < later_num; i++) {
        if (messages->at(i).user == target->user) {
          none_in_between_from_target = false;
        } else if (messages->at(i).user == src->user) {
          none_in_between_from_src = false;
        }
        if (
          (! none_in_between_from_target) &&
          (! none_in_between_from_src)
        ) break;
      }

      int loop_end = src_num - MAX_CHECK_DIST;
      if (loop_end < 0) loop_end = 0;
      for (int i = src_num; i >= loop_end; i--) {
        auto m = messages->at(i);
        if (m.user == target->user && m.targets.count(src->user) > 0) {
          previously_target_addressed_src = true;
        }
        if (m.user == src->user && m.targets.count(target->user) > 0) {
          previously_src_addressed_target = true;
        }
      }
///      loop_end = src_num + MAX_CHECK_DIST;
///      if (loop_end > messages->size()) loop_end = messages->size();
///      for (int i = src_num + 1; i < loop_end; i++) {
///        auto m = messages->at(i);
///        if (m.user == target->user && m.targets.count(src->user) > 0) {
///          future_target_addressed_src = true;
///        }
///        if (m.user == src->user && m.targets.count(target->user) > 0) {
///          future_src_addressed_target = true;
///        }
///      }
    }
    new_feats.push_back(none_in_between_from_src ? y : n);
    new_feats.push_back(none_in_between_from_target ? y : n);
    new_feats.push_back(previously_src_addressed_target ? y : n);
    new_feats.push_back(previously_target_addressed_src ? y : n);
    new_feats.push_back(future_src_addressed_target ? y : n);
    new_feats.push_back(future_target_addressed_src ? y : n);

    bool same_speaker = (has_target && target->user == src->user);
    new_feats.push_back(same_speaker ? y : n);

    bool same_target = false;
    if (has_target) {
      for (auto t : target->targets) {
        if (src->targets.count(t) == 1) same_target = true;
      }
    }
    new_feats.push_back(same_target ? y : n);

    int common_words = 0;
    float total_src_words = 0.0001;
    float total_target_words = 0.0001;
    if (has_target) {
      for (int tok : src->tokens) {
        bool found = false;
        total_src_words += 1.0;
        for (int otok : target->tokens) {
          if (otok == tok) {
            found = true;
            break;
          }
        }
        if (found) common_words += 1;
      }
      for (int otok : target->tokens) total_target_words += 1.0;
    }
    new_feats.push_back(common_words * 2 / (total_src_words + total_target_words));
    new_feats.push_back(common_words / total_src_words);
    new_feats.push_back(common_words / total_target_words);
    new_feats.push_back(common_words == 0 ? y : n);
    new_feats.push_back(common_words == 1 ? y : n);
    new_feats.push_back(common_words > 1 ? y : n);
    new_feats.push_back(common_words > 5 ? y : n);

    // Filter the features based on the feature_config string
///    cout <<"Source: "<< to_string(*src) <<"\n";
///    if (has_target) cout <<"Target: "<< to_string(*target) <<"\n";

    features = new vector<float>();
    feature_cache[pos] = features;
    unsigned int feat = 0;
    for (char& c : feature_config) {
      if (c == ':') break;
      else if (c == 'y' && feat < new_feats.size()) {
        features->push_back(new_feats[feat]);
///        cout << " " << new_feats[feat];
      }
      feat += 1;
    }
///    cout << "\n";

    assert(N_FEATURES_BASE == features->size());
  }

  // Structural features, these must be created every time as they are
  // distinct
  if (INPUT_STRUCTURE) {
    unsigned int feat = 0;
    while (feature_config[feat] != ':') feat += 1;

    // In these 's' is source, and 't' is target, so 'tt' is the target's
    // target.

    bool author_s_is_author_tt = false;
    bool author_s_is_author_ttt = false;
    bool t_was_new_topic = true;
    unsigned int responses_to_t = 0;
    if (has_target) {
      auto tlink = links.find(target_num);
      if (tlink != links.end()) {
        for (auto& tt : tlink->second) {
          if (target_num != tt)
            t_was_new_topic = false;
          if (messages->at(tt).user == src->user)
            author_s_is_author_tt = true;

          auto ttlink = links.find(tt);
          if (ttlink != links.end()) {
            for (auto& ttt : tlink->second) {
              if (messages->at(ttt).user == src->user)
                author_s_is_author_ttt = true;
            }
          }
        }
      }
      if (src_num > target_num) {
        for (unsigned int i = target_num + 1; i < src_num; i++) {
          auto olink = links.find(i);
          if (olink != links.end())
            responses_to_t += olink->second.count(target_num);
        }
      }
    }

    // Are any previous messages from this source user directed at the target
    // user, or vice versa
    bool previous_st_link = false;
    bool previous_ts_link = false;
    bool t_has_response = false;
    if (has_target) {
      int loop_start = src_num - 1000;
      if (loop_start < 0) loop_start = 0;
      for (int i = loop_start ; i < src_num; i++) {
        if (messages->at(i).user == src->user) {
          auto link = links.find(i);
          if (link != links.end()) {
            for (auto& other : link->second) {
              if (messages->at(other).user == target->user)
                previous_st_link = true;
            }
          }
        } else if (messages->at(i).user == target->user) {
          auto link = links.find(i);
          if (link != links.end()) {
            for (auto& other : link->second) {
              if (messages->at(other).user == src->user)
                previous_ts_link = true;
            }
          }
        } else if (i > target_num) {
          auto link = links.find(i);
          if (link != links.end()) {
            for (auto& other : link->second) {
              if (other == target_num) t_has_response = true;
            }
          }
        }
      }
    }

    feat += 1;
    if (feature_config[feat] == 'y')
      contextual_feats.push_back(author_s_is_author_tt ? y : n);
    feat += 1;
    if (feature_config[feat] == 'y')
      contextual_feats.push_back(author_s_is_author_ttt ? y : n);
    feat += 1;
    if (feature_config[feat] == 'y')
      contextual_feats.push_back(t_was_new_topic ? y : n);
    feat += 1;
    if (feature_config[feat] == 'y')
      contextual_feats.push_back(responses_to_t == 0 ? y : n);
    feat += 1;
    if (feature_config[feat] == 'y')
      contextual_feats.push_back(responses_to_t == 1 ? y : n);
    feat += 1;
    if (feature_config[feat] == 'y')
      contextual_feats.push_back(responses_to_t > 1 ? y : n);
    feat += 1;
    if (feature_config[feat] == 'y')
      contextual_feats.push_back(previous_st_link ? y : n);
    feat += 1;
    if (feature_config[feat] == 'y')
      contextual_feats.push_back(previous_ts_link ? y : n);
    feat += 1;
    if (feature_config[feat] == 'y')
      contextual_feats.push_back(t_has_response ? y : n);
  }

  return features;
}

bool IRCLog::are_linked(
  unsigned int source, unsigned int target
) const {
  auto& set = links->find(source)->second;
  return set.find(target) != set.end();
}

ostream& operator<<(ostream& stream, const IRCMessage& m) {
  return stream << to_string(m);
}

string to_string(const IRCMessage& m) {
  std::ostringstream output;
  string time = string(asctime(&(m.time)));
  output << time.substr(0, time.length() - 1);
  output << " <" << m.user << ">";
  output << " [";
  bool first = true;
  for (const auto& target : m.targets) {
    if (! first) output << ",";
    output << target;
    first = false;
  }
  output << "]";
  output << " " << m.message;
  return output.str();
}

ostream& operator<<(ostream& stream, const InstancePair instance) {
  return stream << instance.to_string();
}
ostream& operator<<(ostream& stream, const InstanceSelection instance) {
  return stream << instance.to_string();
}
ostream& operator<<(ostream& stream, const InstanceSet instance) {
  return stream << instance.to_string();
}

void InstancePair::preprocess(LinkingModel& model) {
  model.preprocessPair(log, src_num, target_num);
}
void InstancePair::makeGraph(
  ComputationGraph& cg, vector<Expression>& output, LinkingModel& model,
  unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
  bool eval
) {
  model.makeGraph(cg, output, log, src_num, target_num, current_links, eval);
}
Expression InstancePair::getLoss(
  ComputationGraph& cg, vector<Expression>& output, LinkingModel& model,
  bool eval
) {
  bool linked = log->are_linked(src_num, target_num);
  return model.getLoss(cg, output, eval, linked);
}
void InstancePair::updateEval(
  ComputationGraph& cg, vector<Expression>& output, LinkingModel& model,
  Evaluator& score
) {
  bool linked = log->are_linked(src_num, target_num);
  return model.updateEval(cg, output, linked, score);
}
string InstancePair::to_string() const {
  std::ostringstream output;
  output << "InstancePair: ["
    << " " << log->filename
    << " " << src_num << " -"
    << " " << target_num
    << " ]";
  return output.str();
}
void InstancePair::printPrediction(
  ComputationGraph& cg, vector<Expression>& output, LinkingModel& model
) {
  if (model.getPredictionsPair(cg, output)) {
    cout << "InstancePair: ["
      << " " << log->filename
      << " " << src_num << " -";
    if (src_num != target_num) cout << target_num;
    cout << " ]\n";
  }
}

void InstanceSelection::preprocess(LinkingModel& model) {
  model.preprocessSelection(log, src_num);
}
void InstanceSelection::makeGraph(
  ComputationGraph& cg, vector<Expression>& output, LinkingModel& model,
  unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
  bool eval
) {
  model.makeGraph(cg, output, log, src_num, current_links, eval);
}
Expression InstanceSelection::getLoss(
  ComputationGraph& cg, vector<Expression>& output, LinkingModel& model,
  bool eval
) {
  const unordered_set<unsigned int>& link_set = log->links->at(src_num);
  const unordered_set<unsigned int>* cluster = log->clusters->at(src_num);
  return model.getLoss(cg, src_num, output, eval, link_set, cluster);
}
void InstanceSelection::updateEval(
  ComputationGraph& cg, vector<Expression>& output, LinkingModel& model,
  Evaluator& score
) {
  const unordered_set<unsigned int>& link_set = log->links->at(src_num);
  return model.updateEval(cg, output, src_num, link_set, score);
}
string InstanceSelection::to_string() const {
  std::ostringstream output;
  output << "InstanceSelection: ["
    << " " << log->filename
    << " " << src_num << " -";
  auto iter = log->links->find(src_num);
  if (iter != log->links->end()) {
    for (auto target : iter->second) {
      output << " " << target;
    }
  }
  output << " ]";
  return output.str();
}
void InstanceSelection::printPrediction(
  ComputationGraph& cg, vector<Expression>& output, LinkingModel& model
) {
  vector<unsigned int> targets;
  model.getPredictionsSelection(cg, src_num, output, targets);
  cout << "InstanceSelection: ["
    << " " << log->filename
    << " " << src_num << " -";
  for (auto target : targets) {
    if (target != src_num) cout << " " << target;
  }
  cout << " ]\n";
}

void InstanceSet::preprocess(LinkingModel& model) {
  model.preprocessSet(log, min_source, max_source);
}
void InstanceSet::makeGraph(
  ComputationGraph& cg, vector<Expression>& output, LinkingModel& model,
  unordered_map<unsigned int, unordered_set<unsigned int>>& current_links,
  bool eval
) {
  model.makeGraphSet(cg, output, log, min_source, max_source, current_links,
      eval);
}
Expression InstanceSet::getLoss(
  ComputationGraph& cg, vector<Expression>& output, LinkingModel& model,
  bool eval
) {
  return model.getLossSet(cg, output, eval, log, min_source, max_source);
}
void InstanceSet::updateEval(
  ComputationGraph& cg, vector<Expression>& output, LinkingModel& model,
  Evaluator& score
) {
  return model.updateEvalSet(cg, output, log, min_source, max_source, score);
}
string InstanceSet::to_string() const {
  std::ostringstream output;
  output << "InstanceSet: ["
    << " " << log->filename
    << " " << min_source
    << " " << max_source
    << " ]";
  return output.str();
}
void InstanceSet::printPrediction(
  ComputationGraph& cg, vector<Expression>& output, LinkingModel& model
) {
  unordered_map<unsigned int, unordered_set<unsigned int>> links;
  model.getPredictionsSet(cg, output, min_source, max_source, &links);

  for (unsigned int source = min_source; source < max_source; source++) {
    auto guess = links.find(source);
    cout << "InstanceSet: ["
      << " " << log->filename
      << " " << source << " -";
    if (guess != links.end()) {
      for (auto target : guess->second) cout << " " << target;
    }
    cout << " ]\n";
  }
}

IRCLog* read_file(
  const string& text_file, const string& annotation_file,
  const string& simple_text_file
) {
  // Read messages
  auto messages = new vector<IRCMessage>();
  string line;
  string simple_line;
  cerr << "Reading " << text_file << " and " << simple_text_file << endl;
  ifstream text_in(text_file);
  assert(text_in);
  ifstream simple_text_in(simple_text_file);
  assert(simple_text_in);
  time_t zero = 0;
  tm prev = *gmtime(&zero);
  unordered_set<int> seen_users;
  while (getline(text_in, line)) {
    getline(simple_text_in, simple_line);
    messages->emplace_back(line, simple_line, text_file, prev);
    seen_users.insert(messages->back().user);
    prev = messages->back().time;
  }
  // Add seen user information
  for (auto& msg : *messages) {
    msg.add_targets(seen_users);
  }

  // Read annotations
  auto links = new unordered_map<unsigned int, unordered_set<unsigned int>>();
  ifstream ann_in(annotation_file);
  assert(ann_in);
  unsigned int min_source = UINT_MAX;
  unsigned int max_source = 0;
  while (getline(ann_in, line)) {
    istringstream symbols(line);
    assert(symbols);
    string symbol;

    symbols >> symbol;
    unsigned int source = stoi(symbol);
    if (source < min_source) min_source = source;
    if (source > max_source) max_source = source;

    symbols >> symbol;
    assert(symbol.compare("-") == 0);

    bool no_links = true;
    auto iter = links->insert({source, unordered_set<unsigned int>()});
    while (symbols >> symbol) {
      no_links = false;
      unsigned int target = stoi(symbol);
      iter.first->second.insert(target);
    }
    if (no_links) {
      iter.first->second.insert(source);
    }
  }

  return new IRCLog(text_file, messages, min_source, max_source + 1, links);
}

void read_data(
  vector<IRCLog*>& storage,
  vector<Instance*>& instances,
  const string& file_list
) {
  cerr << "Reading file list from " << file_list << endl;

  ifstream in(file_list);
  assert(in);
  string line;
  unsigned pairs_count = 0;
  unsigned pair_groups_count = 0;
  while (getline(in, line)) {
    istringstream parts(line);
    string text_file;
    string annotation_file;
    string simple_text_file;
    parts >> text_file;
    parts >> annotation_file;
    parts >> simple_text_file;

    auto log = read_file(text_file, annotation_file, simple_text_file);
    storage.push_back(log);

    // Create instances
    unsigned int min_source = log->min_source;
    unsigned int max_source = log->max_source;
    cout << "Making pairs for " << text_file
      << " from " << min_source
      << " to " << max_source
      << endl;
    if (instance_type == kFile) {
      for (unsigned int source = min_source; source < max_source; source += QUERY_SET_SIZE) {
        unsigned int end = source + QUERY_SET_SIZE;
        if (end > max_source) end = max_source;
        instances.push_back(new InstanceSet(log, source, end));
      }
    } else if (instance_type == kSelection) {
      for (unsigned int source = min_source; source < max_source; source++) {
        instances.push_back(new InstanceSelection(log, source));
      }
    } else if (instance_type == kPair) {
      for (unsigned int source = min_source; source < max_source; source++) {
        unsigned int target_start = 0;
        if (source > MAX_LINK_LENGTH) target_start = source - MAX_LINK_LENGTH;
        for (unsigned int target = target_start; target <= source; target++) {
          instances.push_back(new InstancePair(log, source, target));
        }
      }
    }
  }
  cout << "Read " << instances.size() << " instances\n";
}

void read_word_vectors(
  string& filename,
  unordered_map<int, vector<float>>& word_vectors
) {
  cerr << "Reading initial word vectors from " << filename << endl;
  ifstream in(filename);
  assert(in);
  string line;
  unsigned int dim = 0;
  while (getline(in, line)) {
    istringstream parts(line);
    string word;
    parts >> word;
    int word_id = simple_word_dict.convert(word);

    vector<float> nums;
    float num;
    while (! parts.eof()) {
      parts >> num;
      nums.push_back(num);
    }

    word_vectors[word_id] = nums;
    if (dim == 0) {
      dim = nums.size();
    } else {
      assert(dim == nums.size());
    }
  }
  cerr << "Read " << word_vectors.size() << " word vectors, dimension " << dim << endl;
  assert(dim == DIM_INPUT);
}

