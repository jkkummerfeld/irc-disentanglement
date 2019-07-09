#ifndef UM_NLP_CONVGRAPH_EVAL_H_
#define UM_NLP_CONVGRAPH_EVAL_H_

#include <stdlib.h>

using namespace std;

struct Evaluator {
  unsigned int total_gold = 0;
  unsigned int total_guess = 0;
  unsigned int matched = 0;

  unsigned int starts_total_gold = 0;
  unsigned int starts_total_guess = 0;
  unsigned int starts_matched = 0;

  double p(bool start = false);
  double r(bool start = false);
  double f(bool start = false);

  // TODO: other metrics:
  // - Cluster level metrics (VoI, % perfect clusters, % all but one message)

  string to_string();
};

#endif // UM_NLP_CONVGRAPH_EVAL_H_
